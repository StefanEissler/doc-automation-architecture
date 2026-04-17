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

Das Experiment folgt einem kontrollierten Between-Conditions Benchmark-Design mit vier aktiven Systemkonfigurationen und einer manuellen Basisbedingung:

| Bedingung | Bezeichnung | Beschreibung |
|---|---|---|
| C1 | Ground Truth; Regelbasiert | Template-Matching + Regex-Skripte |
| C2 | Single-Prompt LLM | Einmaliger strukturierter LLM-Call, kein Tool-Use |
| C3 | Single ReAct Agent | Iteratives Reasoning mit Tool-Use (Yao et al., 2023) |
| C4 | Multi-Agenten-System | Classifier → Extractor → Validator mit Self-Correction-Loop |

Drei Komplexitätsstufen operationalisieren die Dokumentheterogenität:

- **L1** – Vollständig strukturierte Dokumente (z.B. standardisierte Auftragsbestätigung)
- **L2** – Moderat komplexe Dokumente (z.B. mehrseitige Angebote, Mahnungen mit Zahlungshistorie)
- **L3** – Hochkomplexe Fälle (schlecht gescannte PDFs, gemischtsprachige Dokumente, fehlende Pflichtfelder)

**Abhängige Variablen:** F1-Score je Pflichtfeld · Halluzinationsrate · Token-Cost · Time-to-Completion · Automatisierungsgrad

**Auswertung:** Zweifaktorielle ANOVA (Systemkonfiguration × Komplexitätsstufe) · Tukey-HSD Post-hoc · Regressionsanalyse Break-even

---

## Repository-Struktur

```
doc-automation-benchmark/
│
├── data/
│   ├── raw/                    # Rohdaten (RVL-CDIP, HuggingFace-Quellen)
│   ├── processed/              # Vorverarbeiteter Text nach OCR/Parsing
│   ├── annotated/              # Ground-Truth-Annotationen (JSON)
│   └── synthetic/              # Synthetisch generierte Dokumente (L3)
│
├── src/
│   ├── preprocessing/
│   │   ├── pdf_extractor.py    # pdfplumber, python-docx, email-Parser
│   │   └── ocr.py              # Tesseract-Wrapper für Scans
│   │
│   ├── systems/
│   │   ├── c1_regex.py         # Regelbasiertes System (C1)
│   │   ├── c2_single_prompt.py # Single-Prompt LLM (C2)
│   │   ├── c3_react_agent.py   # ReAct Single Agent via LangGraph (C3)
│   │   └── c4_multi_agent.py   # Multi-Agenten-Pipeline (C4)
│   │
│   ├── evaluation/
│   │   ├── metrics.py          # F1, Halluzinationsrate, Token-Cost, Time
│   │   ├── runner.py           # Automatisierter Durchlauf aller Systeme
│   │   └── ground_truth.py     # Ground-Truth-Loader und Feldvergleich
│   │
│   └── data_generation/
│       └── synthetic_docs.py   # reportlab + faker für L3-Dokumente
│
├── notebooks/
│   ├── 01_data_exploration.ipynb   # Datensatz-Analyse und Verteilung
│   ├── 02_pilot_test.ipynb         # Pilottest mit 20 Dokumenten
│   └── 03_results_analysis.ipynb   # ANOVA, Visualisierungen, Break-even
│
├── results/
│   ├── raw_outputs/            # JSON-Outputs jedes Systems je Dokument
│   └── statistics/             # Auswertungsergebnisse, Plots
│
├── tests/
│   └── test_metrics.py         # Unit-Tests für Evaluationsmetriken
│
├── pyproject.toml              # uv Projektkonfiguration + Abhängigkeiten
├── .env.example                # API-Key-Template (nie .env committen)
├── .gitignore
└── README.md
```

---

## Technische Initialisierung

### Voraussetzungen

- Python ≥ 3.11
- [uv](https://docs.astral.sh/uv/) (empfohlen) oder pip
- Tesseract OCR (`brew install tesseract` / `apt install tesseract-ocr`)
- Anthropic API Key

### Setup mit uv (empfohlen)

```bash
# uv installieren (einmalig)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Repository klonen und Abhängigkeiten installieren
git clone https://github.com/<username>/doc-automation-benchmark.git
cd doc-automation-benchmark
uv sync

# Umgebungsvariablen konfigurieren
cp .env.example .env
# ANTHROPIC_API_KEY=sk-... in .env eintragen
```

### Setup mit venv (alternativ)

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

## Daten Initalisierung

### Download des DocILE Benchmark Dataset

Token muss über: https://docs.google.com/forms/u/0/d/e/1FAIpQLSeYaPkF_BOeD2GwBGueVbprESD7Mys-hMAiUj8oVKBmBGnJUw angefordert werden.

```bash
sudo chmod +x ./scripts/download_dataset.sh
./scripts/download_dataset.sh TOKEN annotated-trainval data/docile --unzip
```

### Kerndependencies

```toml
# pyproject.toml (Auszug)
[project]
name = "doc-automation-benchmark"
requires-python = ">=3.11"

dependencies = [
    "langchain>=0.3",
    "langgraph>=0.2",
    "langchain-anthropic>=0.3",
    "pdfplumber>=0.11",
    "python-docx>=1.1",
    "pytesseract>=0.3",
    "pillow>=10.0",
    "pandas>=2.0",
    "scipy>=1.12",
    "pingouin>=0.5",          # ANOVA und Post-hoc
    "scikit-learn>=1.4",
    "reportlab>=4.0",         # Synthetische PDF-Generierung
    "faker>=24.0",
    "mlflow>=2.0",            # Experiment-Tracking
    "python-dotenv>=1.0",
    "jupyter>=1.0",
]
```

---

## Experiment ausführen

```bash
# Pilottest mit 20 Dokumenten (alle Systeme)
uv run python src/evaluation/runner.py --pilot

# Hauptexperiment (alle 200 Dokumente)
uv run python src/evaluation/runner.py --full

# Einzelnes System testen
uv run python src/evaluation/runner.py --system c4 --docs data/processed/

# Ergebnisse auswerten
uv run jupyter notebook notebooks/03_results_analysis.ipynb
```

---

## Ground-Truth-Format

Jedes Dokument wird als JSON annotiert:

```json
{
  "doc_id": "inv_042",
  "doc_type": "invoice",
  "complexity": "L2",
  "source": "rvl_cdip",
  "annotator_1": "se",
  "annotator_2": "xx",
  "fields": {
    "amount":    "1.234,56",
    "tax":       "19%",
    "date":      "2024-01-15",
    "supplier":  "Mustermann GmbH",
    "account":   null
  }
}
```

`null` bedeutet: Feld im Dokument nicht vorhanden (relevant für Halluzinationsmessung bei L3).

---

## Literatur (Auswahl)

- Yao, S. et al. (2023). ReAct: Synergizing Reasoning and Acting in Language Models. *ICLR 2023*
- Průcha, P. et al. (2025). Are LLM Agents the New RPA? *arXiv:2509.04198*
- Zhu, K. et al. (2025). MultiAgentBench. *arXiv:2503.01935*
- Yehudai, A. et al. (2025). Survey on Evaluation of LLM-based Agents. *arXiv:2503.16416*

---

> Dieses Repository enthält keine echten Unternehmensdaten.  
> Alle verwendeten Dokumente sind entweder öffentliche Datensätze (RVL-CDIP, CORD)  
> oder synthetisch generiert. API-Keys werden ausschließlich lokal via `.env` verwaltet  
> und sind nicht Teil des Repositories.