import json
import logging
import re
from typing import Dict, Optional, Tuple

from langchain.messages import AIMessage, HumanMessage
from langchain.agents import create_agent
from langchain.tools import tool
from langchain.chat_models import BaseChatModel
from langchain_core.output_parsers import (
    JsonOutputParser,
)

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
            f"Your task is to extract the following mandatory fields: {target_fields_str}.\n\n"
            "MANDATORY STEPS (YOU MUST FOLLOW THIS EXACTLY):\n"
            "STEP 1 (VERIFY): Before you do anything else, you MUST use the 'verify_exact_match' tool "
            "to verify the exact spelling of the Advertiser and the Gross Amount.\n"
            "STEP 2 (CALCULATE): If there are line items, you MUST use the 'calculate_sum' tool to check if they match the total.\n"
            "STEP 3 (EXTRACT): ONLY AFTER you have received the tool observations from Step 1 and 2, "
            "you are allowed to output the final data using the required ExtractionSchema.\n\n"
            "CRITICAL RULES:\n"
            "- You MUST use at least one tool before submitting the final schema.\n"
            "- You MUST output the final ExtractionSchema in your last response.\n"
            "- Do not fabricate values; extract texts exactly as they appear in the document.\n"
            "- ALL field values must match the schema types exactly. NEVER invent nested dicts for plain string fields.\n"
            "- DO NOT use markdown, backticks, or code blocks in your final output.\n"
            "- OUTPUT STRICTLY VALID JSON ONLY in your final response.\n"
            "- DO NOT ADD ANY CONVERSATIONAL TEXT, PREAMBLES, OR EXPLANATIONS BEFORE OR AFTER THE JSON.\n"
        )

        agent = create_agent(
            model=self.llm,
            tools=doc_tools,
            system_prompt=system_prompt,
            response_format=ExtractionSchema,
        )

        task_prompt = (
            "### NEW EXTRACTION TASK ###\n\n"
            "<DOCUMENT>\n"
            f"{document.content}\n"
            "</DOCUMENT>\n\n"
            "<TASK_REQUIREMENTS>\n"
            f"Extract the following fields: {target_fields_str}.\n"
            "1. Analyze the document context carefully to avoid hallucinations.\n"
            "2. If a field value is not explicitly present, return 'null'.\n"
            "3. Use the required tools for fact-checking before submitting your final structured answer.\n"
            "</TASK_REQUIREMENTS>\n\n"
            "Execute the steps now. Your last output message MUST be ONLY the raw, **valid JSON** object matching the ExtractionSchema. No other text."
        )

        self.logger.info(f"C3 Starte Single Agent für Dokument {document.id}")

        input_tokens, output_tokens = 0, 0
        used_tools = []
        extracted_data = {}
        extraction_retry = None

        try:
            result = agent.invoke({"messages": [HumanMessage(content=task_prompt)]})
            self.logger.debug(result)

            last_content = result["messages"][-1].content if result["messages"] else ""

            if isinstance(last_content, list):
                text_parts = []
                for part in last_content:
                    if isinstance(part, dict) and "text" in part:
                        text_parts.append(part["text"])
                    elif isinstance(part, str):
                        text_parts.append(part)
                last_content = " ".join(text_parts)

            if last_content:
                parser = JsonOutputParser()
                try:
                    # Standard-Parser to validate and parse the JSON output
                    extracted_data = parser.invoke(last_content)
                except Exception as parse_e:
                    self.logger.warning(
                        "C3: Standard-Parser failed, trying Regex-Fallback."
                    )
                    try:
                        # Try with regex fallback to extract JSON block
                        match = re.search(r"\{.*\}", last_content, re.DOTALL)
                        if match:
                            extraction_retry = 1
                            json_str = match.group(0)
                            extracted_data = json.loads(json_str)
                            self.logger.info("C3: Regex-Fallback erfolgreich.")
                        else:
                            raise ValueError("Kein JSON-Block in der Antwort gefunden.")
                    except Exception as regex_e:
                        self.logger.error(
                            f"C3: Schema-Validation final failed! Parser: {parse_e}, Regex: {regex_e}. Content was: {last_content[:100]}..."
                        )
                        extracted_data = {}

            # Token-Tracking and Tool-Tracking from Message-History
            for msg in result.get("messages", []):
                if isinstance(msg, AIMessage):
                    usage = getattr(msg, "usage_metadata", None) or {}
                    input_tokens += usage.get("input_tokens", 0)
                    output_tokens += usage.get("output_tokens", 0)
                    for tc in getattr(msg, "tool_calls", []):
                        name = (
                            tc.get("name")
                            if isinstance(tc, dict)
                            else getattr(tc, "name", None)
                        )
                        if name and name != ExtractionSchema.__name__:
                            used_tools.append(name)

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
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "used_tools": used_tools,
                "extraction_retry": extraction_retry,
            }

            return {}, safe_metadata, str(e)
