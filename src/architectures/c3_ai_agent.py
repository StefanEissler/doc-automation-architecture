import json
import logging
from typing import Dict, Tuple

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.language_models.chat_models import BaseChatModel

from src.architectures.base import BaseCondition
from src.data_loader import Document


# Definition des Tools für den ReAct-Agenten
@tool
def search_in_document(query: str, document_text: str) -> str:
    """
    Sucht nach einem spezifischen Begriff oder Kontext im Dokumententext.
    Nützlich, wenn das Dokument sehr lang ist und der Agent gezielt nach
    einem bestimmten Feld (z.B. 'Rechnungsnummer') suchen muss.
    """
    # Eine simple textbasierte Suche (könnte später durch echtes RAG ersetzt werden)
    lines = document_text.splitlines()
    results = [line for line in lines if query.lower() in line.lower()]
    if results:
        return "\n".join(results[:5])
    return "Kein relevanter Kontext gefunden."


class SingleAgentCondition(BaseCondition):
    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        self.tools = [search_in_document]
        system_prompt = (
            "Du bist ein intelligenter Datenextraktions-Agent für Geschäftsdokumente.\n"
            "Nutze deine Werkzeuge, um das Dokument schrittweise zu analysieren.\n"
            "Wenn du alle nötigen Informationen gesammelt hast, antworte AUSSCHLIESSLICH "
            "mit einem validen JSON-Objekt. Die Keys müssen exakt den geforderten Feldern entsprechen."
        )
        self.agent = create_agent(
            model=self.llm, tools=self.tools, system_prompt=system_prompt
        )

    def extract_data(self, document: Document) -> Tuple[Dict, int]:
        fields_str = ", ".join(document.target_fields)

        user_prompt = (
            f"Extrahiere die folgenden Felder: {fields_str}\n\n"
            f"Hier ist das vollständige Dokument:\n{document.content}"
        )

        logging.info(f"C3: Starte ReAct-Loop für Dokument {document.id}")
        result = self.agent.invoke(
            {"messages": [{"role": "user", "content": user_prompt}]}
        )

        # Die finale Antwort ist die letzte Nachricht in der Liste
        final_message = result["messages"][-1].content

        # Robusteres JSON-Parsing statt simplem String-Split
        extracted_data = {}
        try:
            clean_text = final_message.replace("```json", "").replace("```", "").strip()
            extracted_data = json.loads(clean_text)
        except json.JSONDecodeError:
            logging.error(f"C3: JSON Parsing fehlgeschlagen für Dokument {document.id}")
            extracted_data = {field: "" for field in document.target_fields}

        # TODO: LangChain Callback Handler für Token-Tracking implementieren
        used_tokens = 0

        return extracted_data, used_tokens
