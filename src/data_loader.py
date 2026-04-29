# src/data_loader.py
import json
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Document:
    """Represents a single Document of the experiment."""

    id: str
    complexity: str
    content: str  # Raw OCR-Text or structured Data
    metadata: dict  # defines type, source, etc.


class DataLoader:
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            logging.error(f"Data directory not found: {self.data_dir}")
            raise FileNotFoundError(f"Directory {self.data_dir} does not exist.")

    def load_docs(
        self, complexity: str = "all", limit: Optional[int] = None
    ) -> List[Document]:
        """
        Loads data of used complexity.
        """
        documents = []

        corpus_file = self.data_dir / "corpus.json"

        try:
            with open(corpus_file, "r", encoding="utf-8") as f:
                raw_data = json.load(f)

            for item in raw_data:
                # Filter for complexity level if specified
                if complexity != "all" and item.get("complexity") != complexity:
                    continue

                doc = Document(
                    id=item["id"],
                    complexity=item["complexity"],
                    content=item["content"],
                    metadata=item.get("metadata", {}),
                )
                documents.append(doc)

                if limit and len(documents) >= limit:
                    break

        except Exception as e:
            logging.error(f"Failed to load documents: {e}")
            raise e

        logging.info(
            f"Loaded {len(documents)} documents for complexity '{complexity}'."
        )
        return documents
