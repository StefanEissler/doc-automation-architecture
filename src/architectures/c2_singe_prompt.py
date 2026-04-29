from langchain.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field


def run_condition_c2():
    print("Running C2 single prompt LLM")
    # Placeholder for the actual implementation of the single prompt experiment


# Definition der Zielannotationen
class ExtractionResult(BaseModel):
    registration_num: str = Field(description="Die Registrierungsnummer des Dokuments")
    date: str = Field(description="Das Ausstellungsdatum im Format YYYY-MM-DD")


# Der Prompt
prompt_template = """
Du bist ein Experte für die Extraktion von Informationen aus Geschäftsdokumenten.
Extrahiere die gewünschten Felder aus dem folgenden Text. 
Antworte AUSSCHLIESSLICH im JSON-Format. Erfinde keine Daten.

Dokumententext:
{document_content}

{format_instructions}
"""
