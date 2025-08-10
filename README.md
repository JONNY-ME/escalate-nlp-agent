
# 🚀 Eskalate NLP Agent

A modular, end-to-end **Natural Language Processing (NLP) agent** for information retrieval, named entity extraction, and summarization — designed to be **config-driven** and **dataset-agnostic**.

This project demonstrates:
- Dataset ingestion (Reuters via NLTK, Amazon Polarity via Hugging Face Datasets)
- Preprocessing & Exploratory Data Analysis (EDA)
- Rule-based and ML-based Named Entity Recognition (NER)
- Extractive summarization (TextRank)
- A configurable **agent** that retrieves, extracts, summarizes, and answers user queries.

---

## 📂 Project Structure

```

escalate-nlp-agent/
├─ README.md
├─ LICENSE
├─ requirements.txt
├─ configs/
│  ├─ config.yaml                  # Master config
│  ├─ dataset/
│  │  ├─ reuters.yaml
│  │  ├─ amazon\_polarity.yaml
│  ├─ extract/
│  │  ├─ rule\_based.yaml
│  │  └─ spacy\_ner.yaml
│  ├─ summarize/
│  │  ├─ textrank.yaml
│  └─ eval/
│     └─ rouge.yaml
├─ data/                           # (gitignored)
│  ├─ raw/
│  ├─ interim/
│  └─ processed/
├─ models/
│  ├─ spacy/
│  └─ hf/
├─ notebooks/
│  ├─ 01\_data\_prep\_eda.ipynb
│  ├─ 02\_extraction\_demos.ipynb
│  ├─ 03\_summarization\_demos.ipynb
│  └─ 04\_agent\_walkthrough.ipynb
├─ reports/
│  ├─ figures/
│  └─ brief\_report.pdf
├─ src/
│  └─ escalate\_nlp\_agent/
│     ├─ text\_prep/
│     ├─ eda/
│     ├─ extract/
│     ├─ summarize/
│     ├─ evaluate/
│     ├─ agent/
│     └─ pipeline.py
├─ scripts/
│  ├─ download\_data.py
│  ├─ run\_pipeline.py
│  ├─ run\_extraction.py
│  ├─ run\_summarization.py
│  └─ run\_agent\_demo.py
└─ tests/

````

---

## 🧠 Features

- **Dataset ingestion** via configs (`configs/dataset/*.yaml`)
- **Preprocessing**: lowercase, punctuation removal, Unicode normalization, stopword removal
- **EDA**: token statistics, vocabulary frequency, NER counts
- **Extraction**:
  - Rule-based regex patterns for dates, numbers, emails, URLs
  - spaCy NER for PERSON, ORG, GPE, DATE, MONEY, PERCENT
- **Summarization**: Extractive summarization with TextRank
- **Agent**:
  - Retrieves top-k relevant documents for a query
  - Runs extractions & summarizations on retrieved docs
  - Returns synthesized final answers

---

## 📦 Installation

### 1️⃣ Clone the repo
```bash
git clone https://github.com/yourusername/escalate-nlp-agent.git
cd escalate-nlp-agent
````

### 2️⃣ Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

---

## 📊 Datasets

### Reuters (NLTK)

```bash
python scripts/download_data.py --dataset reuters
```

Stored under `data/raw/reuters/reuters_full.parquet`.

### Amazon Polarity (Hugging Face)

```bash
python scripts/download_data.py --dataset amazon_polarity
```

Stored under `data/raw/amazon_polarity/amazon_polarity_full.parquet`.

---

## ⚙️ Configuration

All dataset and pipeline settings are in `configs/`:

* `configs/dataset/reuters.yaml` – Reuters dataset setup
* `configs/extract/rule_based.yaml` – Regex extraction patterns
* `configs/extract/spacy_ner.yaml` – spaCy model + entity labels
* `configs/summarize/textrank.yaml` – TextRank parameters

---

## ▶️ Usage

### 1) Run full preprocessing + EDA

```powershell
$env:PYTHONPATH="src"
python scripts/run_pipeline.py --config configs/config.yaml
```

### 2) Run extraction

```powershell
python scripts/run_extraction.py --dataset_config configs/dataset/reuters.yaml --extract_config configs/extract/rule_based.yaml
python scripts/run_extraction.py --dataset_config configs/dataset/reuters.yaml --extract_config configs/extract/spacy_ner.yaml
```

### 3) Run summarization

```powershell
python scripts/run_summarization.py --dataset_config configs/dataset/reuters.yaml --summarize_config configs/summarize/textrank.yaml
```

### 4) Run the agent demo

```powershell
python scripts/run_agent_demo.py --dataset_config configs/dataset/reuters.yaml --agent_config configs/agent/news_aggregator.yaml --query "What did the company report about profits?"
```

---

## 📓 Notebooks

* **01\_data\_prep\_eda.ipynb** – Dataset loading, cleaning, token stats
* **02\_extraction\_demos.ipynb** – Rule-based vs spaCy NER examples
* **03\_summarization\_demos.ipynb** – Extractive summaries on sample docs
* **04\_agent\_walkthrough.ipynb** – End-to-end query demonstration

---

## 📄 Example Agent Output

**Query:**

```
What did the company report about profits?
```

**Answer:**

```
craftmatic/contour industries inc said it would report substantial profits for Q1 1987...
booker plc said 1987 had started well... pretax profits rose from 46.5 mln to 54.6 mln.
```

---

## 🛠 Development

### Run tests

```bash
pytest tests/
```

### Lint and format

```bash
black src/ scripts/
ruff check src/ scripts/
```

---

## 🔮 Next Steps

* Add abstractive summarization (BART, PEGASUS)
* Improve retrieval with semantic search (e.g., SBERT embeddings)
* Deploy as an API with FastAPI/Streamlit

---

## 📜 License

MIT License — see [LICENSE](LICENSE) file.
