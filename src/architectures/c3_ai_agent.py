import logging
from typing import Dict, Optional, Tuple

from langchain.messages import AIMessage, HumanMessage
from langchain.agents import create_agent
from langchain.tools import tool
from langchain.chat_models import BaseChatModel

from src.architectures.base import BaseCondition
from src.data_loader import Document


# Definition des Tools für den ReAct-Agenten
def get_document_tools(document_content: str):

    @tool
    def calculate_sum(value1: float, value2: float) -> float:
        """
        Adds two numerical values together. Mandatory tool to verify
        whether sub-amounts add up to the total gross amount.
        """
        return value1 + value2

    @tool
    def verify_exact_match(extracted_value: str) -> str:
        """
        Performs a fact check to verify if an extracted text value exists EXACTLY as specified in the original document.
        Use this tool to ensure you do not hallucinate words or numbers!
        Returns 'True' if the value exists, otherwise 'False'.
        """
        # A simple but strict substring match against the actual OCR text
        if extracted_value.lower() in document_content.lower():
            return f"True: '{extracted_value}' exists in the document."
        else:
            return f"False: '{extracted_value}' was not found! Please re-read the text carefully."

    @tool
    def clean_and_format_date(raw_date_string: str) -> str:
        """
        Takes an unformatted or messy date string from the OCR text (e.g., '12/24/19' or 'Dec 24 2019')
        and attempts to normalize it into a clean, standardized format.
        """
        # AgenticIE utilizes such sanitizers to ensure JSON quality
        import dateutil.parser

        try:
            parsed_date = dateutil.parser.parse(raw_date_string)
            return parsed_date.strftime("%Y-%m-%d")
        except Exception:
            return "Error: Date could not be parsed. Retain the original value."

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
            "You are an autonomous business data extraction agent.\n"
            f"Your task is to extract the following mandatory fields: {target_fields_str}.\n"
            "Utilize the provided tools to verify facts.\n"
            "CRITICAL: Do not fabricate values; extract texts exactly as they appear in the document.\n"
            "Field Types:\n"
            "  All fields except 'line_items': Simple string (e.g., '476.00' or 'Friends of Jeff')\n"
            "  'line_items': List of strings, one entry per item line\n"
            "NEVER use dictionaries."
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
