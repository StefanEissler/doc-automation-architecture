import json
from typing import Tuple, Dict
from langchain_google_vertexai import ChatVertexAI
from langchain.prompts import PromptTemplate
from src.architectures.base import BaseCondition
from src.data_loader import Document


class SinglePromptCondition(BaseCondition):

    def __init__(self, llm):
        self.llm = llm
        self.prompt = PromptTemplate.from_template(
            "Du bist ein Experte für die Extraktion von Informationen aus Geschäftsdokumenten.\n"
            "Extrahiere die gewünschten Felder aus dem folgenden Text. \n"
            "Antworte AUSSCHLIESSLICH im validen JSON-Format. Erfinde keine Daten.\n\n"
            "Dokumententext:\n{document_content}\n\n"
            "Zielschema (Felder):\n{format_instructions}"
        )

    def extract_data(self, document: Document) -> Tuple[Dict, int]:
        format_instructions = "\n".join(
            [f"- {field}" for field in document.target_fields]
        )

        chain = self.prompt | self.llm

        response = chain.invoke(
            {
                "document_content": document.content,
                "format_instructions": format_instructions,
            }
        )

        # TODO: LangChain Token-Tracking Callback hier integrieren
        used_tokens = 0

        try:
            clean_text = (
                response.content.replace("```json", "").replace("```", "").strip()
            )
            extracted_data = json.loads(clean_text)
        except json.JSONDecodeError:
            extracted_data = {}

        return extracted_data, used_tokens
