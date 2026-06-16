import json
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Dict


@dataclass
class Document:
    id: str
    complexity: str
    content: str
    target_fields: List[str]
    ground_truth: Dict
    pdf_path: Optional[Path] = None


class DataLoader:
    def __init__(self, data_dir: str, experiment: str = "A"):
        self.data_dir = Path(data_dir)
        self.experiment = experiment

        # Leite den Dateinamen aus dem Experiment-Parameter ab
        self.corpus_file = (
            self.data_dir / f"corpus_experiment_{self.experiment}_internal.json"
        )

        if not self.corpus_file.exists():
            raise FileNotFoundError(
                f"Corpus-Datei fehlt: {self.corpus_file}. Bitte Pfad prüfen!"
            )

    def load_docs(
        self, complexity: str = "all", limit: Optional[int] = None
    ) -> List[Document]:
        with open(self.corpus_file, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        documents = []
        for item in raw_data:
            doc_complexity = item.get("complexity", "L1")

            if complexity != "all" and doc_complexity != complexity:
                continue

            ground_truth = item.get("ground_truth", {})
            target_fields = list(ground_truth.keys())

            # Pfad zur PDF-Datei konstruieren (falls später benötigt)
            doc_id = item.get("doc_id")
            pdf_path = (
                self.data_dir
                / "pdfs"
                / f"experiment_{self.experiment}"
                / doc_complexity.lower()
                / doc_id
            )

            doc = Document(
                id=doc_id,
                complexity=doc_complexity,
                content=item.get("ocr_text", ""),
                target_fields=target_fields,
                ground_truth=ground_truth,
                pdf_path=pdf_path if pdf_path.exists() else None,
            )
            documents.append(doc)

            if limit and len(documents) >= limit:
                break

        return documents
