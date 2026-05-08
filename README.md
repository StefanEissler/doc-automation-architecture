# KI-basierte Dokumenten-Automation im Unternehmenskontext
## Ein kontrollierter Architekturvergleich zu Extraktionsgenauigkeit und Betriebskosten

**Masterarbeit** · Universität Hohenheim · Institut für Wirtschaftsinformatik (580A)
**Autor:** Stefan Eißler
**Daten:** April–September 2026

---

## Forschungskontext

Die automatisierte Verarbeitung unstrukturierter Unternehmensdokumente stellt Organisationen vor erhebliche Herausforderungen. Eingehende B2B-Geschäftsdokumente – Angebote, Lieferscheine, Mahnungen, Auftragsbestätigungen – erreichen Unternehmen über heterogene Kanäle ohne einheitliche Struktur und müssen manuell in ERP- und Buchhaltungssysteme übertragen werden. Regelbasierte Automatisierungsansätze (RPA) scheitern an dieser Aufgabe, da sie deterministisch arbeiten und Kontext nicht interpretieren können.

Diese Arbeit untersucht empirisch, ob und unter welchen Bedingungen Multi-Agenten-Systeme gegenüber einfacheren KI-Architekturen überlegene Ergebnisse bei der strukturierten Informationsextraktion aus heterogenen Dokumenten erzielen und zu welchem Kosten-Nutzen-Verhältnis.

---

## Experimentelles Design

Das Experiment folgt einem kontrollierten Between-Conditions Benchmark-Design mit vier Systemkonfigurationen:

| Bedingung | Bezeichnung | Implementierung |
|---|---|---|
| C1 | Regelbasiert | Regex-Pattern-Matching pro Zielfeld (`RuleBasedCondition`) |
| C2 | Single-Prompt LLM | Einmaliger strukturierter LLM-Call ohne Tool-Use (`SinglePromptCondition`) |
| C3 | Single ReAct Agent | Iteratives Reasoning mit Tool-Use via `langchain.agents.create_agent` (`SingleAgentCondition`) (vgl. Yao et al., 2023) |
| C4 | Multi-Agenten-System | Scanner → Extractor → Validator-Pipeline mit Self-Correction-Loop via LangGraph (`MultiAgentCondition`) |

Drei Komplexitätsstufen operationalisieren die Dokumentheterogenität und sind als CLI-Filter (`--complexity`) im Datenlader hinterlegt:

- **L1** – Vollständig strukturierte Dokumente (z.B. standardisierte Auftragsbestätigung)
- **L2** – Moderat komplexe Dokumente (z.B. mehrseitige Angebote, Mahnungen mit Zahlungshistorie)
- **L3** – Hochkomplexe Fälle (schlecht gescannte PDFs, gemischtsprachige Dokumente, fehlende Pflichtfelder)

Pro Dokument und Bedingung werden Vorhersage, Token-Verbrauch und Laufzeit (`perf_counter`-Delta um `extract_data`) erfasst und über den `BenchmarkEvaluator` als CSV nach `results/` geschrieben.

---

## Repository-Struktur

```
doc-automation-architecture/
│
├── data/
│   ├── docile/                      # via scripts/download_dataset_docile.sh
│   ├── vrdu/                        # via scripts/download_dataset_vrdu.sh
│   └── corpus/
│       └── corpus.json              # vom DataLoader erwartetes Eingabeformat
│
├── src/
│   ├── main.py                      # CLI-Einstieg, Experiment-Runner
│   ├── data_loader.py               # Document-Dataclass + Korpus-Loader
│   ├── evaluation.py                # BenchmarkEvaluator, CSV-Export
│   ├── constants.py                 # PROJECT_ROOT
│   │
│   ├── architectures/
│   │   ├── base.py                  # BaseCondition (ABC)
│   │   ├── c1_rule_based.py         # C1: Regex-Extraktion
│   │   ├── c2_singe_prompt.py       # C2: Single-Prompt LLM
│   │   ├── c3_ai_agent.py           # C3: ReAct-Agent mit Suchwerkzeug
│   │   └── c4_multi_ai_agents.py    # C4: LangGraph Multi-Agenten-Workflow
│   │
│   └── preprocessing/
│       └── pdf_extractor.py         # (Platzhalter)
│
├── notebooks/
│   └── data_exploration.ipynb       # Datensatz-Analyse
│
├── scripts/
│   ├── download_dataset_docile.sh   # DocILE-Download (Token-pflichtig)
│   ├── download_dataset_vrdu.sh     # VRDU-Sparse-Checkout via git
│   └── zip-data.sh                  # Backup ohne Roh-Datasets
│
├── results/                         # CSV-Ausgaben des BenchmarkEvaluators
├── tests/
├── main.py                          # Wrapper auf src.main
├── pyproject.toml                   # uv-Projektkonfiguration
├── .env.example
└── README.md
```

---

## Technische Initialisierung

### Voraussetzungen

- Python ≥ 3.13
- [uv](https://docs.astral.sh/uv/)
- Lokale Ollama-Installation mit `llama3`-Modell (Default-Provider)

### Setup

```bash
# uv installieren (einmalig)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Repository klonen und Abhängigkeiten installieren
git clone https://github.com/<username>/doc-automation-architecture.git
cd doc-automation-architecture
uv sync

# Umgebungsvariablen konfigurieren
cp .env.example .env
```

Kerndependencies (siehe `pyproject.toml`): `langchain`, `langgraph`, `langchain-anthropic`, `pdfplumber`, `python-docx`, `pytesseract`, `spacy` (`en_core_web_lg`), `pandas`, `scipy`, `scikit-learn`, `seaborn`, `matplotlib`, `reportlab`, `faker`, `jupyter`.

---

## Datensätze

### DocILE Benchmark Dataset

Token muss über das offizielle Formular angefordert werden: https://docs.google.com/forms/u/0/d/e/1FAIpQLSeYaPkF_BOeD2GwBGueVbprESD7Mys-hMAiUj8oVKBmBGnJUw

```bash
chmod +x ./scripts/download_dataset_docile.sh
./scripts/download_dataset_docile.sh TOKEN annotated-trainval data/docile --unzip
```

### VRDU Dataset

```bash
chmod +x ./scripts/download_dataset_vrdu.sh
./scripts/download_dataset_vrdu.sh
```

Klont per Sparse-Checkout `registration-form/` und `ad-buy-form/` aus `google-research-datasets/vrdu` nach `data/vrdu/` und entpackt die `dataset.jsonl.gz`-Dateien.

### Backup ohne Roh-Datasets

```bash
./scripts/zip-data.sh
```

Erzeugt `data_backup_x_origin.zip` mit allen `data/`-Inhalten **außer** `data/docile/*` und `data/vrdu/*`.

---

## Experiment ausführen

```bash
uv run python -m src.main [OPTIONS]
```

### CLI-Argumente

| Flag | Werte | Default | Beschreibung |
|---|---|---|---|
| `--config` | `all`, `c1`, `c2`, `c3`, `c4` | `all` | Wählt eine einzelne Bedingung oder den vollständigen Architekturvergleich. |
| `--complexity` | `all`, `L1`, `L2`, `L3` | `all` | Filtert das Korpus auf eine Komplexitätsstufe. |
| `--limit` | `int` | `None` | Begrenzt die Zahl geladener Dokumente (z.B. für Pilot-Läufe). |
| `--provider` | `vertex`, `ollama` | `ollama` | LLM-Backend. Aktuell ist nur `ollama` (Modell `llama3`, `temperature=0.2`) aktiv verdrahtet. |

### Beispielaufrufe

```bash
# Vollständiger Lauf: alle 4 Bedingungen über alle Komplexitätsstufen
uv run python -m src.main

# Pilot-Lauf: nur C2 auf 10 L1-Dokumenten
uv run python -m src.main --config c2 --complexity L1 --limit 10

# Multi-Agenten-System auf hochkomplexen Dokumenten
uv run python -m src.main --config c4 --complexity L3
```

### Ergebnisausgabe

Pro Lauf schreibt `BenchmarkEvaluator.save_to_csv` eine Datei `results/evaluation_<timestamp>_<config>_<complexity>.csv` mit den Spalten `condition, complexity, doc_id, f1_score, is_hallucination, token_cost, duration`.

---

## Quellen

Šimsa, Š., Šulc, M., Uřičář, M., Patel, Y., Hamdi, A., Kocián, M., Skalický, M., Matas, J., Doucet, A., Coustaty, M., & Karatzas, D. (2023, February 11). DocILE Benchmark for Document Information Localization and Extraction. arXiv.Org. https://arxiv.org/abs/2302.05658v2

Wang, Z., Zhou, Y., Wei, W., Lee, C.-Y., & Tata, S. (2023). VRDU: A Benchmark for Visually-rich Document Understanding. Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, 5184–5193. https://doi.org/10.1145/3580305.3599929

Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K., & Cao, Y. (2022). ReAct: Synergizing Reasoning and Acting in Language Models (Version 3). arXiv. https://doi.org/10.48550/ARXIV.2210.03629

---

> Dieses Repository enthält keine echten Unternehmensdaten.
> Alle verwendeten Dokumente stammen aus öffentlichen Datensätzen (DocILE, VRDU).
> API-Keys werden ausschließlich lokal via `.env` verwaltet und sind nicht Teil des Repositories.
