import json
import logging
from typing import Dict, Tuple

from langchain.agents import AgentState
from langgraph.graph import StateGraph

from src.architectures.base import BaseCondition
from src.data_loader import Document


class MultiAgentCondition(BaseCondition):
    def __init__(self, llm):
        self.llm = llm
        self.max_retries = 2
        self.workflow = self._create_workflow()

    def _create_workflow(self):
        """Ertellt die LangGraph-Pipline gemäß dem Forschungsdesign"""
        workflow = StateGraph()

        workflow.add_node("scanner", self._scanner_agent)
        workflow.add_node("extractor", self._extractor_agent)
        workflow.add_node("validator", self._validator_agent)

        workflow.set_entry_point("scanner")
        workflow.add_edge(
            "scanner", "extractor", condition=self._scanner_to_extractor_condition
        )
        workflow.add_edge(
            "extractor", "validator", condition=self._extractor_to_validator_condition
        )
        workflow.add_edge(
            "validator", "extractor", condition=self._validator_to_extractor_condition
        )

        workflow.add_conditional_edges(
            "validator",
            self._should_continue,
            {"continue": "extractor", "finish": None},
        )

        return workflow.compile()

    def _scanner_agent(self, state: AgentState):
        """Scanner-Agent: Initialanalyse und OCR-Interpretation."""
        # Hier könnte eine Vor-Klassifikation oder Textbereinigung stattfinden
        logging.info("C4: Scanner-Agent analysiert Dokument...")
        return {"correction_count": 0}

    def _extraction_agent(self, state: AgentState):
        """Extraktions-Agent: Identifikation von Pflichtfeldern[cite: 374]."""
        logging.info("C4: Extraktions-Agent identifiziert Felder...")
        prompt = f"Extrahiere aus folgendem Text: {state['document_content']}. Felder: {state['target_fields']}"
        # Simulierter LLM-Aufruf
        response = self.llm.invoke(prompt)
        try:
            # Annahme: LLM liefert JSON
            data = json.loads(response.content)
        except:
            data = {f: "error" for f in state["target_fields"]}

        return {"raw_extraction": data}

    def _validator_agent(self, state: AgentState):
        """Validator-Agent: Prüft Ergebnis gegen Regelkatalog."""
        logging.info("C4: Validator-Agent prüft Daten...")
        data = state["raw_extraction"]
        errors = {}

        # Beispielhafte Validierung (z.B. Datumsformat oder Vollständigkeit)
        for field in state["target_fields"]:
            if not data.get(field) or data.get(field) == "error":
                errors[field] = "Field missing or invalid"

        return {"validation_results": errors, "final_output": data}

    def _should_continue(self, state: AgentState):
        """Entscheidungslogik für den Self-Correction-Loop (H4)."""
        if state["validation_results"] and state["correction_count"] < self.max_retries:
            logging.warning(
                f"C4: Fehler gefunden. Starte Korrektur-Loop {state['correction_count'] + 1}"
            )
            return "correct"
        return "end"

    def extract_data(self, document: Document) -> Tuple[Dict, int]:
        initial_state = {
            "document_content": document.content,
            "target_fields": document.target_fields,
            "raw_extraction": {},
            "validation_results": {},
            "correction_count": 0,
            "final_output": {},
            "used_tokens": 0,
        }

        result = self.workflow.invoke(initial_state)
        return result["final_output"], result["used_tokens"]
