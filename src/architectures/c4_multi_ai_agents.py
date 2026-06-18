import json
import logging
import re
from typing import Dict, Tuple, List, TypedDict, Any

from langchain.agents import create_agent
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.language_models.chat_models import BaseChatModel

from src.architectures.base import BaseCondition
from src.architectures.c3_ai_agent import get_document_tools
from src.data_loader import Document


# State Definition
class MultiAgentState(TypedDict):
    document_content: str
    target_fields: List[str]
    planner_reasoning: str  # Chain of Thought des Planners
    planner_strategy: str
    raw_extraction: Dict[str, Any]
    final_output: Dict[
        str, Any
    ]  # greift, wenn maximale loops schon erreicht sind, um das fehlerhafte Ergebnis auszugeben.
    validation_errors: str
    correction_count: int
    input_tokens: int
    output_tokens: int


class MultiAgentCondition(BaseCondition):
    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        self.max_retries = 3
        self.logger = logging.getLogger(self.__class__.__name__)
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

    def _planner_node(self, state: MultiAgentState) -> Dict:
        self.logger.info("C4: Planung der Extraktion...")

        prompt = [
            SystemMessage(
                content=(
                    "Du bist der Planner-Agent für Dokumenten-Extraktion. Deine EINZIGE Aufgabe ist es, "
                    "das Dokument zu analysieren und eine Strategie für den Executor-Agenten zu erstellen.\n"
                    "FÜHRE KEINE EXTRAKTION DURCH! Nenne keine konkreten Werte aus dem Text!\n"
                    "Antworte AUSSCHLIESSLICH im folgenden JSON-Format:\n"
                    "{\n"
                    '  "reasoning": "Deine Analyse des Layouts (Wo stehen Tabellen? Wo Metadaten?)",\n'
                    '  "strategy": "Deine konkreten Anweisungen an den Extractor (Worauf muss er achten?)"\n'
                    "}"
                )
            ),
            HumanMessage(
                content=f"Zielfelder, die später extrahiert werden sollen: {state['target_fields']}\n\nDokument:\n{state['document_content']}"
            ),
        ]

        response = self.llm.invoke(prompt)

        try:
            clean_text = re.sub(r"```json\n|\n```", "", response.content).strip()
            json_match = re.search(r"\{.*\}", clean_text, re.DOTALL)
            plan_data = json.loads(json_match.group(0)) if json_match else {}
            reasoning = plan_data.get("reasoning", "Keine Analyse verfügbar.")
            strategy = plan_data.get("strategy", "Standard-Extraktion anwenden.")
        except Exception:
            self.logger.error("C4: Konnte Planner-JSON nicht parsen.")
            reasoning = "Parsing Error."
            strategy = response.content.strip()

        in_tok = (
            response.usage_metadata.get("input_tokens", 0)
            if response.usage_metadata
            else 0
        )
        out_tok = (
            response.usage_metadata.get("output_tokens", 0)
            if response.usage_metadata
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

        doc_tools = get_document_tools(state["document_content"])
        target_str = ", ".join(state["target_fields"])

        feedback_str = ""
        if state.get("validation_errors"):
            feedback_str = f"\n<feedback>\nFEHLER-FEEDBACK DES REVIEWERS:\n{state['validation_errors']}\nBEHEBE DIESE FEHLER ZWINGEND IN DIESEM VERSUCH!\n</feedback>"

        system_prompt = (
            "Du bist der Executor-Agent. Gib am Ende AUSSCHLIESSLICH ein JSON-Objekt aus.\n"
            f"Zwingendes Zielschema (Keys): {target_str}\n\n"
            "WICHTIGE REGELN:\n"
            "1. Wenn eine Information im Dokument fehlt, setze den Wert auf null.\n"
            "2. Erfinde NIEMALS Daten! Nutze das Tool 'verify_exact_match' bei Unsicherheiten.\n"
            "3. Halte dich exakt an die Anweisung des Planners.\n\n"
            "Hier ist der Plan des Planners:\n"
            f"<planner_strategy>\n{state.get('planner_strategy', '')}\n</planner_strategy>\n"
            f"{feedback_str}"
        )

        agent = create_agent(
            model=self.llm, tools=doc_tools, system_prompt=system_prompt
        )

        in_tok = 0
        out_tok = 0

        final_state = {}
        for chunk in agent.stream(
            {
                "messages": [
                    HumanMessage(
                        content="Lies das Dokument und extrahiere die Daten. Nutze bei Bedarf Tools."
                    )
                ]
            },
            stream_mode="values",
        ):
            final_state = chunk
            if (
                isinstance(chunk, AIMessage)
                and hasattr(chunk, "usage_metadata")
                and chunk.usage_metadata
            ):
                in_tok += chunk.usage_metadata.get("input_tokens", 0)
                out_tok += chunk.usage_metadata.get("output_tokens", 0)

        final_message = final_state["messages"][-1]  # Das ist ein Message-Objekt
        final_output = final_message.content

        try:
            clean_text = re.sub(r"```json\n|\n```", "", final_output).strip()
            json_match = re.search(r"\{.*\}", clean_text, re.DOTALL)
            data = json.loads(json_match.group(0)) if json_match else {}
            self.logger.info(data)
            # BUGFIX für Llama3.1: Entpacken des halluzinierten Tool-Calls
            if (
                "name" in data
                and data["name"] == "extract_data"
                and "parameters" in data
            ):
                inner_json_str = data["parameters"].get("json_string", "{}")
                data = json.loads(inner_json_str)

        except Exception:
            self.logger.error("C4: Konnte JSON vom LLM nicht parsen.")
            data = {}

        return {
            "raw_extraction": data,
            "current_output": data,
            "input_tokens": state.get("input_tokens", 0) + in_tok,
            "output_tokens": state.get("output_tokens", 0) + out_tok,
        }

    def _validator_node(self, state: MultiAgentState) -> Dict:
        self.logger.info("C4: Agent 3 (Reviewer) prüft Qualität...")
        data = state.get("raw_extraction", {})

        # Validator überprüft fehlerhafte Einträge.
        prompt = [
            SystemMessage(
                content=(
                    "Du bist der Fact-Checking-Agent. "
                    "Vergleiche das extrahierte JSON mit dem Original-Dokument.\n"
                    "1. Wurden Fakten halluziniert (z.B. falsche Firmennamen), die NICHT im Text stehen?\n"
                    "2. Wurden offensichtliche Beträge oder Namen übersehen?\n"
                    "Antworte AUSSCHLIESSLICH im JSON-Format mit den Keys 'status' und 'feedback'.\n"
                    'Beispiel FAILED: {"status": "FAILED", "feedback": "Der Advertiser \'ABC\' steht nicht im Text."}\n'
                    'Beispiel PASSED: {"status": "PASSED", "feedback": "Alles korrekt extrahiert."}'
                )
            ),
            HumanMessage(
                content=f"Dokument:\n{state['document_content']}\n\nExtrahiertes JSON:\n{json.dumps(data, indent=2)}"
            ),
        ]

        response = self.llm.invoke(prompt)

        try:
            clean_text = re.sub(r"```json\n|\n```", "", response.content).strip()
            json_match = re.search(r"\{.*\}", clean_text, re.DOTALL)
            review_data = json.loads(json_match.group(0)) if json_match else {}
            status = review_data.get("status", "FAILED")
            review = review_data.get("feedback", "Konnte Feedback nicht parsen.")
        except Exception:
            status = "FAILED"
            review = response.content.strip()

        # Pyhton warnt, erzwingt aber keinen Fehler, wenn das LLM "PASSED" sagt.
        python_missing = [
            f
            for f in state["target_fields"]
            if f not in data or data[f] == "" or data[f] is None
        ]

        final_errors = ""
        if status != "PASSED":
            final_errors += "KI-Kritik (Bitte beheben!): " + review + "\n"
            if python_missing:
                final_errors += f"Zusätzlich fehlen diese Keys, prüfe ob sie im Text stehen: {', '.join(python_missing)}"

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

    def extract_data(self, document: Document) -> Tuple[Dict, Dict]:
        initial_state = {
            "document_content": document.content,
            "target_fields": document.target_fields,
            "planner_plan": "",
            "raw_extraction": {},
            "final_output": {},
            "validation_errors": "",
            "correction_count": 0,
            "input_tokens": 0,
            "output_tokens": 0,
        }

        self.logger.info(f"C4: Starte Multi-Agenten Pipeline für {document.id}")
        result_state = self.workflow.invoke(initial_state)

        metadata = {
            "input_tokens": result_state["input_tokens"],
            "output_tokens": result_state["output_tokens"],
            "tokens": result_state["input_tokens"] + result_state["output_tokens"],
        }

        if result_state["correction_count"] == 3:
            return result_state["final_output"], metadata

        return result_state["raw_extraction"], metadata
