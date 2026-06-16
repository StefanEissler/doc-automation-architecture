import json
import logging
import re
from typing import Tuple, Dict

from langchain.messages import HumanMessage, SystemMessage
from src.architectures.base import BaseCondition
from src.data_loader import Document

from langchain_ollama import ChatOllama


class SinglePromptCondition(BaseCondition):

    def __init__(self, llm: ChatOllama):
        self.llm = llm
        self.logger = logging.getLogger(self.__class__.__name__)

    def extract_data(self, document: Document) -> Tuple[Dict, int]:

        format_instructions = "\n".join(
            [f"- {field}" for field in document.target_fields]
        )

        system_msg = SystemMessage(
            content=(
                "Du bist ein Experte für die Extraktion von Informationen aus Geschäftsdokumenten.\n"
                "Extrahiere die gewünschten Felder aus dem vom User bereitgestellten Text.\n"
                "Antworte AUSSCHLIESSLICH im validen JSON-Format. Erfinde keine Daten.\n\n"
                f"Zielschema (Felder):\n{format_instructions}"
            )
        )
        human_msg = HumanMessage(content=f"Dokumententext:\n{document.content}")

        prompts = [system_msg, human_msg]
        response = self.llm.invoke(prompts)

        print(prompts)

        print(response)

        input_tokens = 0
        output_tokens = 0
        if response.usage_metadata or response.response_metadata:
            input_tokens = response.usage_metadata.get("input_tokens", 0)
            output_tokens = response.usage_metadata.get("output_tokens", 0)
            all_tokens = response.usage_metadata.get("total_tokens", 0)
            duration = response.response_metadata.get("total_duration", 0)

        metadata = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "tokens": all_tokens,
            "duration": duration,
        }

        try:
            match = re.search(r"\{.*\}", response.content, re.DOTALL)

            if match:
                clean_text = match.group(0)
                extracted_data = json.loads(clean_text)
            else:
                raise ValueError("Kein JSON-Objekt in der Antwort gefunden.")

        except (json.JSONDecodeError, ValueError) as je:
            self.logger.exception("JSON Decode Error: %s", je)
            extracted_data = {}
        self.logger.info(f"Finished C2 for: ${document.id}")
        return extracted_data, metadata
