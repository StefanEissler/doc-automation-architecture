import json
import logging
from typing import Dict, Optional, Tuple, List, TypedDict, Any

from pydantic import BaseModel, Field, create_model
from langchain.agents import create_agent
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.language_models.chat_models import BaseChatModel

from src.architectures.base import BaseCondition
from src.architectures.c3_ai_agent import get_document_tools
from src.data_loader import Document


# Pydantic Schemata für Planner und Validator
class PlannerOutput(BaseModel):
    reasoning: str = Field(
        description="Detaillierte Analyse des Dokuments (Layout, wo stehen Tabellen, wo Metadaten?)"
    )
    strategy: str = Field(
        description="Konkrete, schrittweise Anweisungen an den Extractor (Worauf muss er bei den spezifischen Pflichtfeldern achten?)"
    )


class ValidatorOutput(BaseModel):
    status: str = Field(description="Muss exakt 'PASSED' oder 'FAILED' sein.")
    feedback: str = Field(
        description="Detailliertes Feedback zu Halluzinationen oder fälschlicherweise ausgelassenen Werten."
    )


# State Definition
class MultiAgentState(TypedDict):
    document_content: str
    target_fields: List[str]
    planner_reasoning: str
    planner_strategy: str
    raw_extraction: Dict[str, Any]
    final_output: Dict[str, Any]
    validation_errors: str
    correction_count: int
    input_tokens: int
    output_tokens: int


class MultiAgentCondition(BaseCondition):
    def __init__(self, llm_text: BaseChatModel, llm_json: BaseChatModel):
        self.llm_text = llm_text
        self.llm_json = llm_json
        self.max_retries = 3
        self.logger = logging.getLogger(self.__class__.__name__)

        self._extractor_agent = None
        self.workflow = self._create_workflow()

    def _create_workflow(self):
        workflow = StateGraph(MultiAgentState)

        workflow.add_node("planner", self._planner_node)
        workflow.add_node("extractor", self._extractor_node)
        workflow.add_node("validator", self._validator_node)

        workflow.add_edge(START, "planner")
        workflow.add_edge("planner", "extractor")
        workflow.add_edge("extractor", "validator")

        workflow.add_conditional_edges(
            "validator",
            self._should_continue,
            {
                "continue": "extractor",
                "end": END,
            },
        )

        return workflow.compile()

    def _build_extractor_agent(self, document_content: str):
        generic_tools = get_document_tools(document_content)
        return create_agent(
            model=self.llm_text,
            tools=generic_tools,
        )

    def _planner_node(self, state: MultiAgentState) -> Dict:
        self.logger.info("C4: Planung der Extraktion...")

        prompt = [
            SystemMessage(
                content=(
                    "Du bist der Planner-Agent in einem hochpräzisen Dokumenten-Extraktions-System.\n"
                    "Deine Aufgabe ist die initiale Dokumentenanalyse. FÜHRE KEINE EXTRAKTION DURCH!\n"
                    "Analysiere die Struktur und erstelle eine narrensichere Strategie für den Executor-Agenten."
                )
            ),
            HumanMessage(
                content=f"Zielfelder: {state['target_fields']}\n\nDokument:\n<document>\n{state['document_content']}\n</document>"
            ),
        ]

        planner_result = self.llm_json.with_structured_output(PlannerOutput)

        try:
            response = planner_result.invoke(prompt)
            reasoning = response.reasoning
            strategy = response.strategy
        except Exception as e:
            self.logger.error(f"C4: Konnte Planner-JSON nicht parsen. Fehler: {e}")
            reasoning = "Parsing Error."
            strategy = "Standard-Extraktion anwenden."

        in_tok = (
            response.usage_metadata.get("input_tokens", 0)
            if hasattr(response, "usage_metadata") and response.usage_metadata
            else 0
        )
        out_tok = (
            response.usage_metadata.get("output_tokens", 0)
            if hasattr(response, "usage_metadata") and response.usage_metadata
            else 0
        )

        return {
            "planner_reasoning": reasoning,
            "planner_strategy": strategy,
            "input_tokens": state.get("input_tokens", 0) + in_tok,
            "output_tokens": state.get("output_tokens", 0) + out_tok,
        }

    def _extractor_node(self, state: MultiAgentState) -> Dict:
        self.logger.info(
            f"C4: Extractor arbeitet. Versuch: {state['correction_count'] + 1}"
        )

        target_str = ", ".join(state["target_fields"])
        feedback_str = (
            f"\n<feedback>\nFEHLER-FEEDBACK DES REVIEWERS:\n{state['validation_errors']}\nBEHEBE DIESE FEHLER ZWINGEND IN DIESEM VERSUCH!\n</feedback>"
            if state.get("validation_errors")
            else ""
        )

        system_prompt = (
            "Du bist der Executor-Agent für Datenextraktion.\n"
            f"Zielschema der Felder: {target_str}\n\n"
            "WICHTIGE REGELN (STRICT COMPLIANCE):\n"
            "1. Wenn eine Information im Dokument fehlt, setze den Wert auf null. Erfinde NIEMALS Daten!\n"
            "2. Du darfst ausschließlich Fakten verwenden, die exakt so im Text stehen.\n"
            "3. Halte dich exakt an die Anweisungen des Planners.\n\n"
            "Plan des Planners:\n"
            f"<planner_strategy>\n{state.get('planner_strategy', '')}\n</planner_strategy>\n"
            f"{feedback_str}"
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(
                content=f"Hier ist das Dokument:\n<document>\n{state['document_content']}\n</document>\n\nLies das Dokument, extrahiere die Daten und gib am Ende alle Werte als Text zurück."
            ),
        ]

        in_tok, out_tok = 0, 0
        final_state = {}

        for chunk in self._extractor_agent.stream(
            {"messages": messages}, stream_mode="values"
        ):
            final_state = chunk
            if (
                isinstance(chunk, AIMessage)
                and hasattr(chunk, "usage_metadata")
                and chunk.usage_metadata
            ):
                in_tok += chunk.usage_metadata.get("input_tokens", 0)
                out_tok += chunk.usage_metadata.get("output_tokens", 0)

        final_message = final_state["messages"][-1].content

        try:
            field_definitions = {
                field: (Optional[str], None) for field in state["target_fields"]
            }
            ExtractionSchema = create_model("ExtractionSchema", **field_definitions)
            structured_parser = self.llm_json.with_structured_output(ExtractionSchema)

            clean_prompt = f"Wandle diesen unstrukturierten Extraktionstext exakt in das geforderte JSON-Schema um. Ändere keine extrahierten Werte!\n\nText:\n{final_message}"
            clean_response = structured_parser.invoke(
                [HumanMessage(content=clean_prompt)]
            )
            data = clean_response.dict() if clean_response else {}
        except Exception as e:
            self.logger.error(
                f"C4: Konnte JSON vom LLM nicht strukturieren. Fehler: {e}"
            )
            data = {}

        return {
            "raw_extraction": data,
            "final_output": data,
            "input_tokens": state.get("input_tokens", 0) + in_tok,
            "output_tokens": state.get("output_tokens", 0) + out_tok,
        }

    def _validator_node(self, state: MultiAgentState) -> Dict:
        self.logger.info("C4: Agent 3 (Reviewer) prüft Qualität...")
        data = state.get("raw_extraction", {})

        prompt = [
            SystemMessage(
                content=(
                    "Du bist der Fact-Checking-Agent in einem audit-sicheren System.\n"
                    "Vergleiche das extrahierte JSON kritisch mit dem Original-Dokument.\n"
                    "1. Wurden Fakten halluziniert, die NICHT im Text stehen?\n"
                    "2. Wurden vorhandene Pflichtfelder übersehen und fälschlicherweise auf null gesetzt?\n"
                    "ACHTUNG: Wenn eine Information wirklich nicht im Text steht, ist es KORREKT, wenn sie im JSON null ist. "
                    "Setze 'status' nur auf 'FAILED', wenn echte Fehler oder Halluzinationen vorliegen. Ansonsten auf 'PASSED'."
                )
            ),
            HumanMessage(
                content=f"Dokument:\n<document>\n{state['document_content']}\n</document>\n\nExtrahiertes JSON:\n{json.dumps(data, indent=2)}"
            ),
        ]

        validator_result = self.llm_json.with_structured_output(ValidatorOutput)

        try:
            response = validator_result.invoke(prompt)
            status = response.status.upper()
            review = response.feedback
        except Exception as e:
            self.logger.error(f"C4: Validator Fehler: {e}")
            status = "FAILED"
            review = "Kritischer Fehler bei der Validierung."

        # BUGFIX: python_missing ist restlos entfernt. Das LLM entscheidet!
        final_errors = ""
        if status != "PASSED":
            final_errors += f"KI-Kritik (Bitte beheben!): {review}"

        return {
            "validation_errors": final_errors.strip(),
            "correction_count": state.get("correction_count", 0) + 1,
        }

    def _should_continue(self, state: MultiAgentState) -> str:
        if state["validation_errors"] and state["correction_count"] <= self.max_retries:
            self.logger.warning(
                f"C4: Fehler gefunden -> Gehe in Korrektur-Loop! (Fehler: {state['validation_errors']})"
            )
            return "continue"

        if state["validation_errors"]:
            self.logger.error("C4: Max Retries erreicht. Beende mit Fehlern.")
        else:
            self.logger.info("C4: Validierung erfolgreich! Keine Fehler gefunden.")

        return "end"

    def extract_data(self, document: Document) -> Tuple[Dict, Dict, Optional[str]]:
        initial_state = {
            "document_content": document.content,
            "target_fields": document.target_fields,
            "planner_reasoning": "",
            "planner_strategy": "",
            "raw_extraction": {},
            "final_output": {},
            "validation_errors": "",
            "correction_count": 0,
            "input_tokens": 0,
            "output_tokens": 0,
        }

        self.logger.info(f"C4: Starte Multi-Agenten Pipeline für {document.id}")
        try:
            self._extractor_agent = self._build_extractor_agent(document.content)
            result_state = self.workflow.invoke(initial_state)

            metadata = {
                "input_tokens": result_state.get("input_tokens", 0),
                "output_tokens": result_state.get("output_tokens", 0),
                "tokens": result_state.get("input_tokens", 0)
                + result_state.get("output_tokens", 0),
            }

            if result_state["correction_count"] > self.max_retries:
                return result_state["final_output"], metadata, None

            return result_state["raw_extraction"], metadata, None

        except Exception as e:
            self.logger.error(f"C4 Error: {e}")
            safe_metadata = {
                "input_tokens": initial_state.get("input_tokens", 0),
                "output_tokens": initial_state.get("output_tokens", 0),
                "tokens": initial_state.get("input_tokens", 0)
                + initial_state.get("output_tokens", 0),
            }
            return {}, safe_metadata, str(e)
