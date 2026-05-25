# Chat With PDF RAG

Retrieval-Augmented Generation project for loading documents, creating embeddings, storing chunks in ChromaDB, and answering questions through a configurable LLM provider.

## Features

- CLI workflow for document ingestion, document Q&A, summarization, resume ATS analysis, database stats, and database reset.
- ChromaDB persistent vector store.
- Ollama embeddings with `nomic-embed-text`.
- Switchable LLM provider through environment variables.
- Gemini 2.5 Flash support through `langchain-google-genai`.
- MLflow tracking for ingestion, query, summary, and resume analysis runs.
- Docker and Docker Compose setup.
- GitHub Actions checks for Python compilation and Docker image builds.

## Project Structure

```text
.
├── app.py
├── cli.py
├── rag_agent_orchestrator.py
├── core/
│   ├── config.py
│   ├── document_loader.py
│   ├── embedding_manager.py
│   ├── logger_config.py
│   ├── mlflow_tracker.py
│   ├── rag_engine.py
│   ├── semantic_chunker.py
│   └── vector_db_manager.py
├── news_articles/
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── .github/workflows/ci.yml
```

## Setup

Create and activate a virtual environment.

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies.

```bash
pip install -r requirements.txt
```

Update `.env` with your provider settings. The repo includes `.env.example` and a placeholder `.env`.

```env
LLM_PROVIDER=gemini
LLM_MODEL=gemini-2.5-flash
GOOGLE_API_KEY=your_api_key_here
```

The app still uses Ollama for embeddings by default, so keep Ollama running with the embedding model available.

```bash
ollama pull nomic-embed-text
```

For local Ollama generation instead of Gemini:

```env
LLM_PROVIDER=ollama
LLM_MODEL=llama3
OLLAMA_BASE_URL=http://localhost:11434
```

## Run

Start the CLI.

```bash
python app.py
```

Run the programmatic example.

```bash
python app.py --programmatic
```

## Docker

Build and run with Docker Compose.

```bash
docker compose up --build
```

The compose file mounts these local folders into the container:

- `news_articles`
- `chroma_persistent_storage`
- `logs`
- `mlruns`

When using Ollama from Docker on Windows or macOS, set:

```env
OLLAMA_BASE_URL=http://host.docker.internal:11434
```

## MLflow

MLflow is enabled by default.

```env
ENABLE_MLFLOW=true
MLFLOW_TRACKING_URI=file:./mlruns
MLFLOW_EXPERIMENT_NAME=chat-with-pdf-rag
```

Open the local MLflow UI after running the app.

```bash
mlflow ui --backend-store-uri ./mlruns
```

## GitHub Actions

The CI workflow runs on push to `main` and on pull requests. It performs:

- Dependency installation.
- Python compile checks.
- Docker image build.

## Environment Variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `LLM_PROVIDER` | `ollama` | `gemini` or `ollama`. |
| `LLM_MODEL` | `llama3` | Chat model name. Use `gemini-2.5-flash` for Gemini. |
| `GOOGLE_API_KEY` | empty | Required when `LLM_PROVIDER=gemini`. |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API URL. |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Ollama embedding model. |
| `CHROMA_PATH` | `chroma_persistent_storage` | Chroma persistence directory. |
| `CHROMA_COLLECTION` | `rag_documents` | Chroma collection name. |
| `LOG_LEVEL` | `INFO` | Application log level. |
| `LOG_FILE` | `rag_agent.log` | Log filename under `logs/`. |
| `ENABLE_MLFLOW` | `true` | Enables MLflow tracking. |
| `MLFLOW_TRACKING_URI` | `file:./mlruns` | MLflow tracking store. |
| `MLFLOW_EXPERIMENT_NAME` | `chat-with-pdf-rag` | MLflow experiment name. |
