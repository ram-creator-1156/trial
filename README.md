# TradeMatch LOC
### Swipe-to-Export Intelligent Matchmaking Algorithm
> **Hackathon 2026** — Full-stack Python application

---

## 📁 Project Structure

```
Hackathon/
├── data/
│   └── EXIM_DatasetAlgo_Hackathon.xlsx   ← Place your Excel here
├── backend/
│   ├── main.py                            ← FastAPI entry point
│   ├── schemas.py                         ← Pydantic models
│   └── routers/
│       └── match.py                       ← Match / swipe endpoints
├── frontend/
│   ├── app.py                             ← Streamlit swipe UI
│   └── components/                        ← Reusable UI components
├── models/
│   ├── preprocessor.py                    ← Feature engineering
│   └── matchmaker.py                      ← Core scoring engine
├── utils/
│   ├── data_loader.py                     ← Excel sheet loader
│   ├── news_signals.py                    ← News sentiment processor
│   └── logger.py                          ← Loguru logger
├── config/
│   └── settings.py                        ← Pydantic-settings config
├── logs/                                  ← Auto-generated log files
├── .env.example                           ← Environment variable template
├── requirements.txt
└── README.md
```

---

## ⚡ Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure environment
```bash
copy .env.example .env
# Edit .env to set your OPENAI_API_KEY if needed
```

### 3. Place the Excel file
```
data\EXIM_DatasetAlgo_Hackathon.xlsx
```

### 4. Start the FastAPI backend
```bash
uvicorn backend.main:app --reload --port 8000
```
- Swagger UI → http://localhost:8000/docs
- Health check → http://localhost:8000/health

### 5. Start the Streamlit frontend (new terminal)
```bash
streamlit run frontend/app.py
```
- UI → http://localhost:8501

---

## 🧠 Algorithm Overview

The matching score is a **weighted sum of 5 signals**:

| Signal | Weight | Description |
|---|---|---|
| Text Similarity | 40% | TF-IDF cosine similarity on product descriptions |
| HS Code Match | 25% | Exact / prefix-level Harmonised System code alignment |
| Numeric Fit | 20% | Capacity vs demand, price vs budget compatibility |
| News Sentiment | 10% | HS-level sentiment from Global News sheet |
| Compliance | 5% | Compliance score alignment between parties |

A **penalty** is applied when the importer's country appears in news as sanctioned/banned.

---

## 🔌 API Endpoints

| Method | URL | Description |
|---|---|---|
| `GET` | `/health` | System health + loaded row counts |
| `GET` | `/api/matches` | Top-K matches (optional importer filter) |
| `POST` | `/api/matches` | Same, via JSON body |
| `POST` | `/api/swipe` | Record like / dislike / superlike |
| `GET` | `/api/swipes` | Retrieve swipe history |
| `GET` | `/api/exporters` | Paginated list of exporters |
| `GET` | `/api/importers` | Paginated list of importers |

---

## 🛠️ Tech Stack

- **Backend**: FastAPI + Uvicorn
- **Frontend**: Streamlit
- **ML Core**: Scikit-learn (TF-IDF, MinMaxScaler, cosine similarity)
- **Data**: Pandas + OpenPyXL
- **NLP/LLM**: LangChain + Sentence-Transformers (optional enrichment)
- **Config**: Pydantic-Settings + Python-dotenv
- **Logging**: Loguru
