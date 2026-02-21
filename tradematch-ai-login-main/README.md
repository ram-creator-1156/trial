# 🌐 TradeMatch AI — Swipe to Export Intelligent Matchmaking

An AI-powered platform that intelligently matches **Exporters** with **Importers** using live trade signals, global news sentiment, and machine-learning scoring — all wrapped in an intuitive swipe-based UI.

## 🏗️ Project Structure

```
tradematch-ai-login-main/
├── backend/                  # FastAPI backend
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes/
│   │       ├── __init__.py
│   │       ├── match.py      # Matchmaking endpoints
│   │       ├── exporter.py   # Exporter CRUD & queries
│   │       └── importer.py   # Importer CRUD & queries
│   ├── core/
│   │   ├── __init__.py
│   │   └── config.py         # App settings & env vars
│   ├── services/
│   │   ├── __init__.py
│   │   ├── data_loader.py    # Excel / data ingestion
│   │   ├── matching.py       # Orchestrates the algorithm
│   │   └── news_signal.py    # Global news signal processing
│   ├── __init__.py
│   └── main.py               # FastAPI app entry point
│
├── frontend/                 # Streamlit frontend
│   ├── __init__.py
│   ├── app.py                # Streamlit entry point
│   ├── pages/
│   │   ├── __init__.py
│   │   ├── dashboard.py      # Overview dashboard
│   │   ├── swipe.py          # Swipe-to-match UI
│   │   └── results.py        # Match results & export
│   └── components/
│       ├── __init__.py
│       ├── cards.py          # Exporter/Importer cards
│       └── filters.py       # Filter & search widgets
│
├── models/                   # ML models & logic
│   ├── __init__.py
│   ├── matchmaker.py         # Core matchmaking algorithm
│   ├── feature_engineer.py   # Feature engineering pipeline
│   ├── scorer.py             # Similarity / compatibility scoring
│   └── saved_models/         # Serialised model artifacts
│       └── .gitkeep
│
├── utils/                    # Shared utilities
│   ├── __init__.py
│   ├── constants.py          # Project-wide constants
│   ├── helpers.py            # Misc helper functions
│   └── logger.py             # Logging configuration
│
├── data/                     # Data directory
│   ├── .gitkeep
│   └── EXIM_DatasetAlgo_Hackathon.xlsx  ← place your file here
│
├── .env.example              # Environment variable template
├── .gitignore
├── requirements.txt
└── README.md
```

## ⚡ Quick Start

```bash
# 1. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Copy env template and set your keys
copy .env.example .env       # then edit .env

# 4. Place your dataset
#    Put EXIM_DatasetAlgo_Hackathon.xlsx inside the data/ folder

# 5. Start the FastAPI backend
uvicorn backend.main:app --reload --port 8000

# 6. Start the Streamlit frontend (in a new terminal)
streamlit run frontend/app.py
```

## 📊 Dataset Sheets

| Sheet Name                          | Description                     |
| ----------------------------------- | ------------------------------- |
| `Exporter_LiveSignals_v5_Updated`   | Exporter profiles & signals     |
| `Global_News_LiveSignals_Updated`   | Global trade news & sentiment   |
| `Importer_LiveSignals_v5_Updated`   | Importer profiles & signals     |

## 🛠️ Tech Stack

- **Backend:** FastAPI, Uvicorn
- **Frontend:** Streamlit
- **ML / Data:** Pandas, NumPy, Scikit-learn
- **AI / NLP:** LangChain, OpenAI
- **Data Format:** OpenPyXL (Excel)
