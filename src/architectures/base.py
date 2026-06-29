from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple, Dict
from src.data_loader import Document


class BaseCondition(ABC):
    @abstractmethod
    def extract_data(
        self, document: Document
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Optional[str]]:
        """
        Gives back 3-Tupels:
        1. extracted_data – Extracted JSON-Dict, or {} with error
        2. metadata – Aggregated Token-Counter (input_tokens, output_tokens, tokens, duration)
        3. error_msg – Error Message when Modell-/Parsing-Errors, else None
        """
        pass
