import logging
from typing import Dict, Optional, Tuple

from langchain.messages import HumanMessage, SystemMessage
from langchain.chat_models import BaseChatModel
from pydantic import Field, create_model

from src.architectures.base import BaseCondition
from src.data_loader import Document


class SinglePromptCondition(BaseCondition):
    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        self.logger = logging.getLogger(self.__class__.__name__)

    def extract_data(self, document: Document) -> Tuple[Dict, Dict, Optional[str]]:
        ExtractionSchema = document.schema_class
        self.logger.debug(document.schema_class)

        if not ExtractionSchema:
            self.logger.error(f"Doc {document.id} has no valid Schema.")
            return {}, {}, "No schema available"

        target_fields_str = ", ".join(document.target_fields)

        system_msg = SystemMessage(
            content=(
                "You are an expert data extraction assistant specialized in B2B ad-buy forms and invoices.\n"
                "Your task is to extract exact values from the provided OCR text according to the required schema.\n"
                f"These are the target fields to extract: {target_fields_str}.\n"
                "You MUST extract all tabular rows into the line_items array.\n"
                "If a value is missing, return null. Do NOT create or hallucinate any data!\n\n"
                "CRITICAL INSTRUCTIONS FOR JSON VALUES:\n"
                "1. OUTPUT STRICTLY THE FINAL EXTRACTED VALUE ONLY.\n"
                "2. NO EXPLANATIONS. NO REASONING. NO CHAIN OF THOUGHT.\n"
                "3. DO NOT write full sentences inside the fields (e.g., write '$560.00' and NEVER 'The gross amount is $560.00').\n"
                "4. Be absolutely concise."
            )
        )
        human_msg = HumanMessage(
            content=f"Extract the data of this document content:\n{document.content}"
        )

        # Strukturierten Output erzwingen und Rohdaten für Token-Tracking anfordern
        structured_llm = self.llm.with_structured_output(
            ExtractionSchema, include_raw=True
        )

        input_tokens = 0
        output_tokens = 0
        extracted_data = {}
        error_msg = None

        self.logger.info(f"C2: Starting Single Prompt Extraction for {document.id}")

        try:
            response = structured_llm.invoke([system_msg, human_msg])

            parsed_data = response.get("parsed")
            raw_message = response.get("raw")

            if parsed_data:
                extracted_data = parsed_data.dict()
            else:
                error_msg = "LLM has no returned a valid Schema."

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
