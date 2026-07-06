import json
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Type

from dataclasses_json import dataclass_json
from pydantic import BaseModel

from src.schemas import DocileBaseSchema, VRDUBaseSchema


@dataclass_json
@dataclass
class Document:
    id: str
    content: str
    complexity: str
    target_fields: List[str]  # Bleibt eine Liste für die Prompts
    ground_truth: Dict
    schema_class: Optional[Type[BaseModel]] = None
    metadata: Optional[Dict] = field(default_factory=dict)
    pdf_path: Optional[Path] = None
    source: Optional[str] = None


class DataLoader:
    def __init__(self, data_dir: str, experiment: str = "A"):
        self.data_dir = Path(data_dir)
        self.experiment = experiment
        self.logger = logging.getLogger(self.__class__.__name__)

        self.corpus_file = (
            self.data_dir / f"corpus_experiment_{self.experiment}_internal.jsonl"
        )

        if not self.corpus_file.exists():
            raise FileNotFoundError(
                f"Corpus-Datei fehlt: {self.corpus_file}. Bitte Pfad prüfen!"
            )

    def load_docs(
        self, complexity: list | str = "all", limit: Optional[int] = None
    ) -> List[Document]:

        if isinstance(complexity, str):
            complexity = [complexity]

        documents = []
        with open(self.corpus_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                doc_complexity = item.get("complexity", "L1")
                doc_id = item.get("id")

                if "all" not in complexity and doc_complexity not in complexity:
                    continue

                ground_truth = item.get("ground_truth", {})

                # Das rohe Dict aus dem JSON! WICHTIG: Nicht sofort in list() casten!
                target_fields_raw = item.get("target_fields", {})
                source: str = item.get("source", "")

                if source.startswith("VRDU"):
                    base_schema = VRDUBaseSchema
                elif source == "docile":
                    base_schema = DocileBaseSchema
                else:
                    base_schema = VRDUBaseSchema

                schema = None
                try:
                    schema = base_schema.filter_schema(target_fields_raw)
                except ValueError as e:
                    self.logger.warning(f"Warning for Document: {doc_id}: {e}")
                    # Fallback
                    common_fields = [
                        f for f in target_fields_raw if f in base_schema.model_fields
                    ]
                    schema = (
                        base_schema.filter_schema(common_fields)
                        if common_fields
                        else None
                    )

                self._log_schema_details(doc_id, target_fields_raw, schema)

                target_fields_flat = (
                    list(target_fields_raw.keys())
                    if isinstance(target_fields_raw, dict)
                    else target_fields_raw
                )

                doc = Document(
                    id=doc_id,
                    complexity=doc_complexity,
                    content=item.get("content", ""),
                    target_fields=target_fields_flat,
                    ground_truth=ground_truth,
                    pdf_path=item.get("pdf_path", ""),
                    schema_class=schema,
                    source=source,
                    metadata=item.get("metadata", {}),
                )
                documents.append(doc)

                if limit and len(documents) >= limit:
                    break

        return documents
