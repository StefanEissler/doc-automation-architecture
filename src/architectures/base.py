from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple, Dict
from src.data_loader import Document


class BaseCondition(ABC):
    @abstractmethod
    def extract_data(
        self, document: Document
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Optional[str]]:
        """
        Gibt ein 3-Tupel zurück:
        1. extracted_data – Extrahiertes JSON-Dict, oder {} bei Fehler
        2. metadata – Aggregierte Token-Zähler (input_tokens, output_tokens, tokens, duration)
        3. error_msg – Fehlermeldung bei Modell-/Parsing-Fehler, sonst None
        """
        pass
