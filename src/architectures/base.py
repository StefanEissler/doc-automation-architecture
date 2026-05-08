from abc import ABC, abstractmethod
from copyreg import constructor
from typing import Tuple, Dict
from src.data_loader import Document


class BaseCondition(ABC):
    @abstractmethod
    def extract_data(self, document: Document) -> Tuple[Dict, int]:
        pass
