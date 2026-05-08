import re
from typing import Tuple, Dict
from src.architectures.base import BaseCondition
from src.data_loader import Document


class RuleBasedCondition(BaseCondition):

    def extract_data(self, document: Document) -> Tuple[Dict, int]:
        text = document.content
        extracted_data = {}

        regex_patterns = document.metadata.get("regex_patterns", {})

        for field in document.target_fields:
            pattern = regex_patterns.get(field)
            if pattern:
                match = re.search(pattern, text)
                extracted_data[field] = match.group(1).strip() if match else ""
            else:
                extracted_data[field] = ""

        return extracted_data, 0
