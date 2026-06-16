from abc import ABC, abstractmethod
from copyreg import constructor
from typing import Any, Tuple, Dict
from src.data_loader import Document


class BaseCondition(ABC):
    @abstractmethod
    def extract_data(self, document: Document) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Gibt ein Tupel zurück:
        1. Extrahiertes JSON-Dict
        2. Metadaten-Dict (input_tokens, output_tokens, etc.)
        """
        pass
