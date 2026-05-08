from typing import Dict, Tuple

from src.architectures.base import BaseCondition
from src.data_loader import Document


class SingleAgentCondition(BaseCondition):
    def __init__(self, llm):
        self.llm = llm

    def extract_data(self, document: Document) -> Tuple[Dict, int]:
        prompt = f"""
        Extrahiere die folgenden Felder aus dem Dokument:\n
        {', '.join(document.target_fields)}\n
        Dokumentinhalt:\n
        {document.content}
        """

        response = self.llm(prompt)
        extracted_data = self.parse_response(response)

        # Hier noch richtig extrahieren.
        token_count = len(response) // 4

        return extracted_data, token_count

    def parse_response(self, response: str) -> Dict:
        extracted_data = {}
        lines = response.splitlines()
        for line in lines:
            if ":" in line:
                key, value = line.split(":", 1)
                extracted_data[key.strip()] = value.strip()
        return extracted_data
