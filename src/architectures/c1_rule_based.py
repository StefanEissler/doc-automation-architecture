import re
from typing import Tuple, Dict
from src.architectures.base import BaseCondition
from src.data_loader import Document

import logging


class RuleBasedCondition(BaseCondition):

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def extract_data(self, document: Document) -> Tuple[Dict, int]:
        try:
            text = document.content
            text_lower = text.lower()
            extracted_data = {}

            for field in document.target_fields:
                field_lower = field.lower()

                variants = [field_lower]
                if "_" in field_lower:
                    variants.append(field_lower.replace("_", " "))
                if "-" in field_lower:
                    variants.append(field_lower.replace("-", " "))

                match_index = -1
                matched_variant_len = 0

                for variant in variants:
                    idx = text_lower.find(variant)
                    if idx != -1 and (
                        match_index == -1 or len(variant) > matched_variant_len
                    ):
                        match_index = idx
                        matched_variant_len = len(variant)
                        break

                if match_index != -1:
                    start_pos = match_index + matched_variant_len
                    remaining_text = text[start_pos:].strip()

                    remaining_text = re.sub(r"^[\s:\-]+", "", remaining_text)

                    words = remaining_text.split()
                    extracted_data[field] = words[0] if words else ""
                else:
                    extracted_data[field] = ""

            return extracted_data, 0
        except Exception as e:
            self.logger.exception(e)
