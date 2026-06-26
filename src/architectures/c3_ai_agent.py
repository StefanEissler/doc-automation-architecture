import logging
from typing import Any, Dict, List, Optional, Tuple

from langchain.messages import AIMessage, HumanMessage
from langchain.agents import create_agent
from langchain.tools import tool
from langchain.chat_models import BaseChatModel
from pydantic import Field, create_model

from src.architectures.base import BaseCondition
from src.data_loader import Document


# Definition des Tools für den ReAct-Agenten
def get_document_tools(document_content: str):

    @tool
    def calculate_sum(value1: float, value2: float) -> float:
        """
        Addiert zwei Zahlenwerte. Nutze dies zwingend, um zu überprüfen,
        ob Teilbeträge (Sub-Amounts) den Gesamtbetrag (Gross Amount) ergeben.
        """
        return value1 + value2

    @tool
    def verify_exact_match(extracted_value: str) -> str:
        """
        Prüft (Faktencheck), ob ein extrahierter Text-Wert EXAKT so im Original-Dokument vorkommt.
        Nutze dieses Tool, um sicherzustellen, dass du keine Wörter oder Zahlen halluzinierst!
        Gibt 'True' zurück, wenn der Wert existiert, sonst 'False'.
        """
        # Ein einfacher, aber harter Substring-Match gegen den echten OCR Text
        if extracted_value.lower() in document_content.lower():
            return f"True: '{extracted_value}' existiert im Dokument."
        else:
            return f"False: '{extracted_value}' wurde nicht gefunden! Bitte lies den Text nochmal genau."

    @tool
    def clean_and_format_date(raw_date_string: str) -> str:
        """
        Nimmt einen unsauberen Datums-String aus dem OCR-Text (z.B. '12/24/19' oder 'Dec 24 2019')
        und versucht, ihn in ein sauberes, einheitliches Format zu übersetzen.
        """
        # AgenticIE nutzt solche Sanitizer, um die JSON-Qualität zu sichern
        import dateutil.parser

        try:
            parsed_date = dateutil.parser.parse(raw_date_string)
            return parsed_date.strftime("%Y-%m-%d")
        except Exception:
            return (
                "Fehler: Datum konnte nicht geparst werden. Behalte den Originalwert."
            )

    return [calculate_sum, verify_exact_match, clean_and_format_date]


class SingleAgentCondition(BaseCondition):
    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        self.logger = logging.getLogger(self.__class__.__name__)

    def extract_data(self, document: Document) -> Tuple[Dict, Dict, Optional[str]]:
        ExtractionSchema = document.schema_class

        doc_tools = get_document_tools(document.content)

        target_fields_str = ", ".join(document.target_fields)
        system_prompt = (
            "Du bist ein autonomer Extraktionsagent für Geschäftsdaten.\n"
            f"Deine Aufgabe ist die Extraktion folgender Pflichtfelder: {target_fields_str}.\n"
            "Nutze Tools, um Fakten zu prüfen.\n"
            "WICHTIG: Erfinde keine Werte; übernehme Texte exakt aus dem Dokument.\n"
            "Feldtypen:\n"
            "- Alle Felder außer 'line_items': einfacher String (z.B. '476.00' oder 'Friends of Jeff')\n"
            "- 'line_items': Liste von Strings, ein Eintrag pro Zeile\n"
            "NIEMALS Dictionaries verwenden."
        )

        agent = create_agent(
            model=self.llm,
            tools=doc_tools,
            system_prompt=system_prompt,
            response_format=ExtractionSchema,
        )

        task_prompt = (
            f"### New Documents ###\n\n"
            f"(<Document>)\n"
            f"{document.content}\n"
            f"(</Document>)\n\n"
            f"(<Task>)\n"
            f"Extrahiere die geforderten Werte aus dem Dokument. Achte extrem genau auf den Kontext, um Halluzinationen zu vermeiden. "
            f"Werte müssen exakt aus dem Text übernommen werden.\n"
            f"Ziel-Schema:\n{target_fields_str}\n"
            f"(</Task>)\n\n"
            f"Gebe als Antwort ausschließlich das Pydantic-Objekt im geforderten Schema zurück."
        )

        self.logger.info(f"C3 Starte Single Agent für Dokument {document.id}")

        input_tokens, output_tokens = 0, 0
        used_tools = []
        extracted_data = {}

        try:
            for chunk in agent.stream(
                {"messages": [HumanMessage(content=task_prompt)]},
                stream_mode="updates",
                version="v2",
            ):
                self.logger.debug(f"STREAM CHUNK: {chunk}")
                if chunk["type"] == "updates":
                    for node_name, node_output in chunk["data"].items():
                        if node_name == "tools":
                            for msg in node_output.get("messages", []):
                                used_tools.append(msg.name)
                                self.logger.info(
                                    f"Tool {msg.name} wurde erfolgreich ausgeführt."
                                )

                        if "messages" in node_output:
                            for msg in node_output["messages"]:

                                # Token Tracking
                                if isinstance(msg, AIMessage):
                                    usage = getattr(msg, "usage_metadata", None)
                                    if not usage and isinstance(msg, dict):
                                        usage = msg.get("usage_metadata")
                                    if usage:
                                        input_tokens += usage.get("input_tokens", 0)
                                        output_tokens += usage.get("output_tokens", 0)

                                # Daten Extraktion
                                if isinstance(msg, AIMessage) and hasattr(
                                    msg, "tool_calls"
                                ):
                                    for tc in msg.tool_calls:
                                        if tc["name"] == "ExtractionSchema":
                                            extracted_data = tc["args"]

            used_tools = list(dict.fromkeys(used_tools))

            metadata = {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "used_tools": used_tools,
            }

            self.logger.debug("C3 DEBUG OUTPUT:")
            self.logger.debug(f"Structured Result:\n{extracted_data}")
            self.logger.debug(f"Token Metadata:\n{metadata}")
            self.logger.info(f"Finished C3 for: {document.id}")

            return extracted_data, metadata, None

        except Exception as e:
            self.logger.error(f"C3 Error: {e}")
            safe_metadata = {
                "agent_input_tokens": input_tokens,
                "agent_output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "used_tools": used_tools,
            }

            return {}, safe_metadata, str(e)
