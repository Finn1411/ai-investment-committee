# Finance Agent — AI Investment Research & Forecasting System

> **Private research and decision support tool.**  
> **Does NOT replace your own verification of data or professional investment advice.**

![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)
![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-green.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-teal.svg)

A **data-driven investment research system** that evaluates probabilities, risks, and scenarios — not a stock price predictor.

**Design Principle:**  
*"Based on the data available at the time of analysis, the estimated probability of outperformance against Benchmark X over 12 months is Y%. The investment thesis is based on A, B, and C. It becomes invalid if D, E, or F occur."*

---

## 🌟 Key Features

* **🧠 Multi-Agent Committee**: 7 independent LLM agents (Fundamental, Value, Growth, Earnings, Risk, Bear, and Committee Lead) analyze the data, debate, and form a consensus rating (BUY / HOLD / AVOID).
* **📊 Deterministic Quant Engine**: LLMs *never* do math. A Python quant engine calculates 45+ metrics (DCF, Piotroski F-Score, Altman Z-Score, Scenarios, etc.) before the agents ever see the data.
* **🌐 RAG News Ingestion**: Automatically scrapes real-time news from Yahoo Finance, extracts factual claims using Gemini structured outputs, stores them in a local ChromaDB, and injects them into the agents' context to prevent hallucinations.
* **🌍 Macro & Peer Awareness**: Dynamically detects the market regime (Risk-On/Off via VIX and 10Y Yield) and benchmarks the stock's relative strength against direct competitors.
* **📓 Prediction Journal**: Every decision is permanently logged to a local SQLite database for future backtesting, accountability, and accuracy calibration.
* **💻 Premium Web Dashboard**: A stunning glassmorphic dark-mode web UI to run analyses, view streaming agent thoughts, and track historical predictions.
* **🚨 Automated Desktop Alerts**: A background monitor script that scans your database daily and triggers native Windows Desktop Notifications if a stock is downgraded.

---

## 🛠 Tech Stack

* **Backend**: FastAPI, Python 3.11+
* **LLM**: Google Gemini API (`gemini-3.5-flash`, `gemini-3.6-flash`, `gemini-embedding-2`)
* **Vector DB**: ChromaDB (Local)
* **Relational DB**: SQLite + SQLAlchemy ORM
* **Frontend**: Vanilla JavaScript (ES6), HTML5, Custom CSS
* **Automation**: `win10toast` for native desktop alerts

---

## 🚀 Quick Start

### 1. Clone & Environment Setup
```bash
git clone https://github.com/yourusername/finance-agent.git
cd finance-agent
python -m venv .venv

# Windows
.venv\Scripts\activate
# Mac/Linux
source .venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Secrets
Create an `.env` file from the example:
```bash
cp .env.example .env
```
Edit `.env` and add your Gemini API Key:
```env
GEMINI_API_KEY="your_api_key_here"
```

### 4. Initialize the Database
```bash
python -c "from finance_agent.database.db import init_db; init_db()"
```

---

## 💻 Usage

### Launch the Web Dashboard (Recommended)
Start the FastAPI server:
```bash
python -m uvicorn finance_agent.api.main:app --reload
```
Navigate to `http://localhost:8000` in your browser.

### Run Automated Watchlist Monitor
To scan your saved stocks and get Desktop Alerts for downgrades:
```bash
python -m finance_agent.automation.monitor
```

---

## 🏗 Architecture

```text
Data Sources (YFinance, News)
       ↓
Data Validation Agent (Cleans & Checks)
       ↓
Feature / Quant Engine (DCF, Scenarios, Macro, Peers)
       ↓
RAG Engine (ChromaDB + Gemini Embeddings)
       ↓
Fundamental / Growth / Value / Earnings Agents
       ↓
Risk Manager + Bear Agent (Stress tests the bullish thesis)
       ↓
Committee Engine (Final Verdict & Invalidation Criteria)
       ↓
Prediction Journal (SQLite)
       ↓
Web Dashboard / Desktop Alerts
```

---

## 📜 Principles

1. **LLMs never do math.** All calculations are deterministic.
2. **Data is validated before agents see it.**
3. **Probabilistic outputs.** We use Bear/Base/Bull scenarios, not point predictions.
4. **Accountability.** Every prediction is logged immutably.
5. **Traceability.** Agents must cite their sources.
