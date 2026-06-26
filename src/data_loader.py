from curses import meta
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
    target_fields: List[str]
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

        # Leite den Dateinamen aus dem Experiment-Parameter ab
        self.corpus_file = (
            self.data_dir / f"corpus_experiment_{self.experiment}_internal.jsonl"
        )

        if not self.corpus_file.exists():
            raise FileNotFoundError(
                f"Corpus-Datei fehlt: {self.corpus_file}. Bitte Pfad prüfen!"
            )

    def load_docs(
        self, complexity: str = "all", limit: Optional[int] = None
    ) -> List[Document]:
        documents = []

        with open(self.corpus_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                doc_complexity = item.get("complexity", "L1")
                doc_id = item.get("id")

                if complexity != "all" and doc_complexity != complexity:
                    continue

                ground_truth = item.get("ground_truth", {})
                target_fields = item.get("target_fields", [])

                source: str = item.get("source", "")
                schema = None

                if source.startswith("VRDU"):
                    base_schema = VRDUBaseSchema
                elif source == "docile":
                    base_schema = DocileBaseSchema
                else:
                    base_schema = VRDUBaseSchema

                try:
                    schema = base_schema.filter_schema(target_fields)
                except ValueError as e:
                    self.logger.log(f"WARNUNG bei Doc {item.get("id")}: {e}")
                    # Fallback: Nur die Felder, die sicher da sind, oder leeres Schema
                    # Besser: Crash vermeiden, leere Liste oder Basis-Filterung
                    common_fields = [
                        f for f in target_fields if f in VRDUBaseSchema.model_fields
                    ]
                    schema = (
                        VRDUBaseSchema.filter_schema(common_fields)
                        if common_fields
                        else None
                    )

                doc = Document(
                    id=doc_id,
                    complexity=doc_complexity,
                    content=item.get("content", ""),
                    target_fields=target_fields,
                    ground_truth=ground_truth,
                    pdf_path=item.get("pdf_path", ""),
                    schema_class=schema,
                )
                documents.append(doc)

                if limit and len(documents) >= limit:
                    break

        return documents
