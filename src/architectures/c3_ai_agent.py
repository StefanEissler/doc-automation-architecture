import json
import logging
import re
from typing import Dict, Optional, Tuple

from langchain.agents import create_agent
from langchain.messages import AIMessage, HumanMessage
from langchain.tools import tool
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.checkpoint.memory import InMemorySaver

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

        # system_prompt = PromptTemplate(
        #     template=(
        #         "Du bist ein Datenextraktions-Experte für Geschäftsdokumente.\n"
        #         "Extrahiere die gewünschten Felder aus dem Dokumententext.\n\n"
        #         "{format_instructions}\n\n"
        #         "Dokumententext:\n{document_content}\n"
        #     ),
        #     input_variables=["document_content"],
        #     partial_variables={
        #         "format_instructions": self.parser.get_format_instructions()
        #     },
        # )

    def extract_data(self, document: Document) -> Tuple[Dict, Dict, Optional[str]]:
        doc_tools = get_document_tools(document.content)
        target_fields_str = ", ".join(document.target_fields)

        system_prompt = (
            "Du bist ein autonomer Extraktions-Agent für Geschäftsdaten.\n"
            f"Deine Aufgabe ist es, folgende Pflichtfelder aus dem Dokumententext zu extrahieren: {target_fields_str}.\n"
            "Nutze deine Tools (z.B. calculate_sum), falls du Beträge mathematisch überprüfen musst, um die Validität zu erhöhen.\n"
            "Gib am Ende AUSSCHLIESSLICH ein valides JSON-Objekt zurück, das die extrahierten Daten enthält. "
            "Erfinde keine Daten. Wenn ein Feld nicht existiert, setze es auf null.\n\n"
            f"DOKUMENTENTEXT:\n{document.content}"
        )

        checkpointer = InMemorySaver()

        agent = create_agent(
            model=self.llm,
            tools=doc_tools,
            system_prompt=system_prompt,
            checkpointer=checkpointer,
        )

        self.logger.info(f"C3: Starte Agenten-Loop für Dokument {document.id}")

        input_tokens = 0
        output_tokens = 0

        try:
            thread_config = {
                "configurable": {"thread_id": str(document.id.strip(".pdf"))}
            }

            final_state = None
            for chunk in agent.stream(
                {
                    "messages": [
                        HumanMessage(
                            content="Starte jetzt die Suche und Extraktion für die geforderten Felder."
                        )
                    ]
                },
                config=thread_config,
                stream_mode="values",
            ):
                final_state = chunk
                latest_message = chunk["messages"][-1]
                if isinstance(latest_message, AIMessage) and getattr(
                    latest_message, "tool_calls", None
                ):
                    tool_names = [tc["name"] for tc in latest_message.tool_calls]
                    self.logger.info(f"C3: Agent nutzt Tools -> {tool_names}")

            if not final_state or "messages" not in final_state:
                raise ValueError("Agent hat keine gültige Antwort generiert.")

            all_messages = final_state["messages"]

            tool_call_count = sum(
                len(msg.tool_calls)
                for msg in all_messages
                if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None)
            )

            self.logger.info(
                f"C3 erfolgreich. Tool Calls: {tool_call_count} | Nachrichten insgesamt: {len(all_messages)}"
            )

            self.logger.info(all_messages)
            final_message = all_messages[-1]
            final_output = final_message.content

            # Token aggregieren
            for msg in all_messages:
                if (
                    isinstance(msg, AIMessage)
                    and hasattr(msg, "usage_metadata")
                    and msg.usage_metadata
                ):
                    input_tokens += msg.usage_metadata.get("input_tokens", 0)
                    output_tokens += msg.usage_metadata.get("output_tokens", 0)

            clean_text = re.sub(r"```json\n|\n```", "", final_output).strip()
            json_match = re.search(r"\{.*\}", clean_text, re.DOTALL)
            extracted_data = json.loads(json_match.group(0)) if json_match else {}

            metadata = {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "tokens": input_tokens + output_tokens,
            }

            self.logger.info(f"Finished C3 for: {document.id}")
            return extracted_data, metadata, None
        except Exception as e:
            self.logger.error(f"C3 Error: {e}")
            safe_metadata = locals().get(
                "metadata",
                {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "tokens": input_tokens + output_tokens,
                },
            )
            return {}, safe_metadata, str(e)
