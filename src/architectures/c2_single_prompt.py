import logging
from typing import Dict, Optional, Tuple

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import create_model

from src.architectures.base import BaseCondition
from src.data_loader import Document


class SinglePromptCondition(BaseCondition):
    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        self.logger = logging.getLogger(self.__class__.__name__)

    def extract_data(self, document: Document) -> Tuple[Dict, Dict, Optional[str]]:
        # Dynamisches Pydantic-Schema basierend auf den Ziel-Feldern generieren
        field_definitions = {
            field: (Optional[str], None) for field in document.target_fields
        }
        ExtractionSchema = create_model("ExtractionSchema", **field_definitions)

        target_fields_str = ", ".join(document.target_fields)

        system_msg = SystemMessage(
            content=(
                "Du bist ein Experte für die Extraktion von Informationen aus Geschäftsdokumenten.\n"
                f"Extrahiere EXAKT diese Pflichtfelder: {target_fields_str}.\n"
                "Erfinde keine Daten. Setze Felder auf null, wenn sie fehlen."
            )
        )
        human_msg = HumanMessage(content=f"Dokumententext:\n{document.content}")

        # Strukturierten Output erzwingen und Rohdaten für Token-Tracking anfordern
        structured_llm = self.llm.with_structured_output(
            ExtractionSchema, include_raw=True
        )

        input_tokens = 0
        output_tokens = 0
        extracted_data = {}
        error_msg = None

        self.logger.info(f"C2: Starte Single Prompt Extraktion für {document.id}")

        try:
            response = structured_llm.invoke([system_msg, human_msg])

            parsed_data = response.get("parsed")
            raw_message = response.get("raw")

            if parsed_data:
                extracted_data = parsed_data.dict()
            else:
                error_msg = "LLM hat kein valides Schema zurückgegeben."

            if (
                raw_message
                and hasattr(raw_message, "usage_metadata")
                and raw_message.usage_metadata
            ):
                input_tokens = raw_message.usage_metadata.get("input_tokens", 0)
                output_tokens = raw_message.usage_metadata.get("output_tokens", 0)

        except Exception as e:
            self.logger.error(f"C2 Error: {e}")
            error_msg = str(e)

        metadata = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "tokens": input_tokens + output_tokens,
        }

        self.logger.debug("C2 DEBUG OUTPUT:")
        self.logger.debug(f"Extracted Data:\n{extracted_data}")
        self.logger.debug(f"Token Metadata:\n{metadata}")
        if error_msg:
            self.logger.debug(f"Error Message: {error_msg}")

        self.logger.info(f"Finished C2 for: {document.id}")
        return extracted_data, metadata, error_msg
