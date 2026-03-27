# Arxis — AI Research Assistant

A full-stack research workspace powered by a LangGraph agent, TF-IDF RAG, and Groq inference. Search arXiv, chat with your papers, and take notes — all in one place.

---

## Features

- **Agentic RAG** — LangGraph agent with two tools: fetch papers from arXiv and search indexed chunks via TF-IDF
- **No GPU required** — in-memory TF-IDF index (~5MB for 50 papers, 150-250 chunks), no embeddings model needed
- **Fast inference** — Groq (Llama 3.1 8B) at 500+ tok/s, sub-3s average response latency
- **3-panel workspace** — Chat, Papers, and Notes panels in a single-page interface
- **Live paper discovery** — papers found during chat are automatically saved to the project with arXiv links
- **Research notes** — markdown editor with auto-save (3s debounce) persisted to the database
- **Multi-turn memory** — per-project conversation history maintained across requests
- **Resilient API calls** — exponential backoff retry on arXiv 429s and Groq tool-call failures

---

## Tech Stack

| Layer | Tech |
|---|---|
| Backend | Django 5.2, SQLite |
| Agent | LangGraph, langchain-groq |
| LLM | Groq — Llama 3.1 8B Instant |
| Retrieval | scikit-learn TF-IDF |
| Paper source | arXiv API (`arxiv` package) |
| Frontend | Vanilla JS, AJAX, marked.js |

---

## Project Structure

```
researcher/          # Django project config
Home/                # Projects model, home + project list views
workplace/           # Workspace views (chat, notes, papers)
agent/
  app.py             # LangGraph agent, TF-IDF store, tools
db.sqlite3
```

---

## Setup

**1. Clone and install dependencies**

```bash
pip install django langchain-groq langchain-community langchain-core langgraph \
            langchain-text-splitters arxiv scikit-learn numpy python-dotenv
```

**2. Set your Groq API key**

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_key_here
```

Get a free key at [console.groq.com](https://console.groq.com).

**3. Run migrations and start the server**

```bash
python manage.py migrate
python manage.py runserver
```

**4. Open the app**

Go to `http://127.0.0.1:8000`, create a project, and start researching.

---

## How it works

1. You type a question in the Chat panel
2. The LangGraph agent decides whether to fetch papers from arXiv, search the TF-IDF index, or answer directly
3. Fetched papers are chunked and indexed in RAM for the session
4. Papers discovered during chat are saved to the project's DB record with their arXiv links
5. The answer is rendered with full markdown support in the chat interface

---

## Environment Variables

| Variable | Description |
|---|---|
| `GROQ_API_KEY` | Required. Your Groq API key. |
