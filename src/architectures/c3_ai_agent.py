import logging
from typing import Dict, Optional, Tuple

from langchain.agents import create_agent
from langchain.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool, StructuredTool
from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import create_model

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
        doc_tools = get_document_tools(document.content)
        target_fields_str = ", ".join(document.target_fields)

        field_definitions = {
            field: (Optional[str], None) for field in document.target_fields
        }
        ExtractionSchema = create_model("ExtractionSchema", **field_definitions)

        def submit_data(**kwargs):
            return "Daten erfolgreich eingereicht. Beende den Task."

        submit_tool = StructuredTool.from_function(
            func=submit_data,
            name="submit_extracted_data",
            description="NUTZE DIESES TOOL ZWINGEND ALS ALLERLETZTE AKTION. Übergib hier die finalen, extrahierten Daten.",
            args_schema=ExtractionSchema,
        )
        doc_tools.append(submit_tool)

        system_prompt = (
            "Du bist ein autonomer Extraktionsagent für Geschäftsdaten.\n"
            f"Deine Aufgabe ist die Extraktion folgender Pflichtfelder: {target_fields_str}.\n"
            "Nutze Tools, um Fakten zu prüfen.\n"
            "WICHTIG: Wenn du alle Daten gefunden hast, MUSST du als allerletzten Schritt das Tool "
            "'submit_extracted_data' aufrufen, um deine Ergebnisse abzugeben. Erfinde keine Werte."
        )

        agent = create_agent(
            model=self.llm,
            tools=doc_tools,
        )

        self.logger.info(f"C3 Starte echten Single Agent für Dokument {document.id}")

        input_tokens, output_tokens = 0, 0
        extracted_data = {}

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(
                content=f"DOKUMENTENTEXT:\n<document>\n{document.content}\n</document>\n\n"
                "Starte jetzt die Extraktion."
            ),
        ]

        try:
            for chunk in agent.stream({"messages": messages}, stream_mode="values"):
                # Token Tracking
                if (
                    isinstance(chunk, AIMessage)
                    and hasattr(chunk, "usage_metadata")
                    and chunk.usage_metadata
                ):
                    input_tokens += chunk.usage_metadata.get("input_tokens", 0)
                    output_tokens += chunk.usage_metadata.get("output_tokens", 0)

                latest_message = chunk["messages"][-1]
                if isinstance(latest_message, AIMessage) and hasattr(
                    latest_message, "tool_calls"
                ):
                    for tc in latest_message.tool_calls:
                        if tc["name"] == "submit_extracted_data":
                            extracted_data = tc["args"]
                            break

                if extracted_data:
                    break

            metadata = {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "tokens": input_tokens + output_tokens,
            }

            self.logger.info(f"Finished C3 for: {document.id}")
            return extracted_data, metadata, None

        except Exception as e:
            self.logger.error(f"C3 Error: {e}")
            safe_metadata = {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "tokens": input_tokens + output_tokens,
            }
            return {}, safe_metadata, str(e)
