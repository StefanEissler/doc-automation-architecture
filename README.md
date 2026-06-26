# AI-Based Document Automation in a Corporate Context

## A Controlled Architecture Comparison of Extraction Accuracy and Operational Costs

**Master's Thesis** · University of Hohenheim · Institute of Information Systems (580A)

**Author:** Stefan Eißler

**Date:** April–September 2026

---

## Research Context

The automated processing of unstructured corporate documents poses significant challenges for organizations. Incoming B2B business documents—such as quotes, delivery notes, payment reminders, and order confirmations—reach companies through heterogeneous channels without a uniform structure. They typically require manual transfer into ERP and accounting systems. Rule-based automation approaches (RPA) fail at this task because they operate deterministically and cannot interpret context.

This thesis empirically investigates whether and under what conditions multi-agent systems deliver superior results compared to simpler AI architectures for structured information extraction from heterogeneous documents, and evaluates the corresponding cost-benefit ratio.

---

## Experimental Design

The experiment follows a controlled between-conditions benchmark design featuring four system configurations:

| Condition | Designation | Implementation |
| --- | --- | --- |
| **C1** | Rule-Based | Regex pattern matching per target field (`RuleBasedCondition`) |
| **C2** | Single-Prompt LLM | One-time structured LLM call without tool use (`SinglePromptCondition`) |
| **C3** | Single ReAct Agent | Iterative reasoning with tool use via `langchain.agents.create_agent` (`SingleAgentCondition`) (cf. Yao et al., 2023) |
| **C4** | Multi-Agent System | Planner → Extractor → Validator pipeline with a self-correction loop via LangGraph (`MultiAgentCondition`) |

Three complexity levels operationalize document heterogeneity and are integrated as CLI filters (`--complexity`) within the data loader:

* **L1** – Fully structured documents (e.g., standardized order confirmations)
* **L2** – Moderately complex documents (e.g., multi-page quotes, payment reminders with payment history)
* **L3** – Highly complex cases (poorly scanned PDFs, mixed-language documents, missing mandatory fields)

For each document and condition, the prediction, token consumption, and runtime (`perf_counter` delta around `extract_data`) are captured and written to a CSV file in `results/` via the `BenchmarkEvaluator`.

---

## Repository Structure

```
doc-automation-architecture/
│
├── data/
│   ├── raw/                      # via scripts/download_dataset_docile.sh scripts/download_dataset_vrdu.sh
│   ├── processed/                # processed data
│
├── src/
│   ├── main.py                      # main method, Experiment-Runner
│   ├── data_loader.py               # Document-Dataclass + Korpus-Loader
│   ├── evaluation.py                # BenchmarkEvaluator, CSV-Export
│   ├── constants.py                 # PROJECT_ROOT
│   │
│   ├── architectures/
│   │   ├── base.py                  # BaseCondition (ABC)
│   │   ├── c1_rule_based.py         # C1: Regex extraction
│   │   ├── c2_singe_prompt.py       # C2: Single-Prompt LLM
│   │   ├── c3_ai_agent.py           # C3: ReAct Agent with search tool
│   │   └── c4_multi_ai_agents.py    # C4: LangGraph Multi-Agent Workflow
│
├── notebooks/
│   └── data_exploration.ipynb       # Dataset analysis
│
├── scripts/
│   ├── download_dataset_docile.sh   # DocILE download (Token required)
│   ├── download_dataset_vrdu.sh     # VRDU sparse checkout via git
│   └── zip-data.sh                  # Backup without raw datasets
│
├── results/                         # CSV outputs of the BenchmarkEvaluator
├── tests/                           # Testing scripts
├── main.py                          # Wrapper for src.main
├── pyproject.toml                   # uv project configuration
├── .env.example
└── README.md

```

---

## Technical Initialization

### Prerequisites

* Python ≥ 3.13
* [uv](https://docs.astral.sh/uv/)
* Local Ollama installation with the `llama3` model (default provider)

### Setup

```bash
# Install uv (one-time setup)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone repository and install dependencies
git clone https://github.com/StefanEissler/doc-automation-architecture.git
cd doc-automation-architecture
uv sync

# Configure environment variables
cp .env.example .env

```

**Core Dependencies** (see `pyproject.toml`): `langchain`, `langgraph`, `langchain-anthropic`, `pdfplumber`, `python-docx`, `pytesseract`, `spacy` (`en_core_web_lg`), `pandas`, `scipy`, `scikit-learn`, `seaborn`, `matplotlib`, `reportlab`, `faker`, `jupyter`.

---

## Datasets

### DocILE Benchmark Dataset

A token must be requested via the official form: [https://docs.google.com/forms/u/0/d/e/1FAIpQLSeYaPkF_BOeD2GwBGueVbprESD7Mys-hMAiUj8oVKBmBGnJUw](https://docs.google.com/forms/u/0/d/e/1FAIpQLSeYaPkF_BOeD2GwBGueVbprESD7Mys-hMAiUj8oVKBmBGnJUw)

```bash
chmod +x ./scripts/download_dataset_docile.sh
./scripts/download_dataset_docile.sh TOKEN annotated-trainval data/docile --unzip

```

### VRDU Dataset

```bash
chmod +x ./scripts/download_dataset_vrdu.sh
./scripts/download_dataset_vrdu.sh

```

Performs a sparse checkout of `registration-form/` and `ad-buy-form/` from `google-research-datasets/vrdu` into `data/vrdu/` and extracts the `dataset.jsonl.gz` files.

### Backup Without Raw Datasets

```bash
./scripts/zip-data.sh

```

Generates a `data_backup_x_origin.zip` file containing all contents of `data/` **except** `data/docile/*` and `data/vrdu/*`.

### Installing the Ollama Model

Comprehensive documentation: [https://docs.langchain.com/oss/python/integrations/llms/ollama](https://docs.langchain.com/oss/python/integrations/llms/ollama)

**Installation:**

* **Mac:**
```bash
brew install ollama

```


* **Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh

```



---

## Running the Experiment

```bash
uv run python -m main [OPTIONS]

```

### CLI Arguments

| Flag | Values | Default | Description |
| --- | --- | --- | --- |
| `--condition` | `all`, `C1`, `C2`, `C3`, `C4` | `all` | Selects a single condition or executes the complete architectural comparison. |
| `--complexity` | `all`, `L1`, `L2`, `L3` | `all` | Filters the corpus to a specific complexity level. |
| `--limit` | `int` | `None` | Limits the number of loaded documents (e.g., for pilot runs). |
| `--provider` | `openai`, `ollama` | `ollama` | LLM provider. ⚠️ *Only Ollama is currently implemented.* |
| `--model` | `str` | `llama3.3` | Specific model name for the selected provider (e.g., `llama3.3`, `gpt-4o-mini`, `gpt-4o`). |
| `--log` | `str` | `INFO` | Configuration of the logger (set to `DEBUG` for verbose output or `INFO` as default). |

### Example Commands

```bash
# Full execution: run all 4 conditions across all complexity levels for partial experiment A
uv run python -m main

# Testing a single condition:
uv run python -m main --condition C2 --experiment A --limit 1 --model llama3.1:8b

# Pilot run: only C2 on 10 L1 documents
uv run python -m main --condition C2 --complexity L1 --limit 10

# Multi-agent system executed on highly complex documents
uv run python -m main --condition C4 --complexity L3

# Running the experiment on the cluster with corresponding execution script
chmod +x scripts/run_benchmark_experiment.sh
./scripts/run_benchmark_experiment.sh

```

### Evaluation Output

For each execution, `BenchmarkEvaluator.save_to_csv` writes a file named `results/evaluation_<timestamp>_<config>_<complexity>.csv` containing the following columns: `condition, complexity, doc_id, f1_score, is_hallucination, token_cost, duration`.

---

## References

Šimsa, Š., Šulc, M., Uřičář, M., Patel, Y., Hamdi, A., Kocián, M., Skalický, M., Matas, J., Doucet, A., Coustaty, M., & Karatzas, D. (2023, February 11). DocILE Benchmark for Document Information Localization and Extraction. arXiv.Org. [https://arxiv.org/abs/2302.05658v2](https://arxiv.org/abs/2302.05658v2)

Wang, Z., Zhou, Y., Wei, W., Lee, C.-Y., & Tata, S. (2023). VRDU: A Benchmark for Visually-rich Document Understanding. Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, 5184–5193. [https://doi.org/10.1145/3580305.3599929](https://doi.org/10.1145/3580305.3599929)

Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K., & Cao, Y. (2022). ReAct: Synergizing Reasoning and Acting in Language Models (Version 3). arXiv. [https://doi.org/10.48550/ARXIV.2210.03629](https://doi.org/10.48550/ARXIV.2210.03629)

---

> This repository does not contain real corporate data.
> All documents utilized are sourced from public benchmarks (DocILE, VRDU).
> API keys are managed exclusively via local `.env` configuration and are excluded from the repository tracking.