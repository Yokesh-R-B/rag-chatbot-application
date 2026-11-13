# RAG Chatbot

A document question-answering application built with FastAPI, Streamlit,
ChromaDB, and Groq LLaMA. Users can upload PDF, DOCX, and HTML files, ask
questions, and continue conversations with session history.

## Configuration

Copy `.env.example` to `.env` and set `GROQ_API_KEY`. The application stores
SQLite and ChromaDB data in `DATA_DIR`; use a persistent volume for deployments.
`MAX_UPLOAD_BYTES` defaults to 25 MiB. Set `ALLOWED_ORIGINS` and `API_BASE_URL`
for the URLs used by your deployment.

## Local development

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Start the API:

```powershell
Push-Location api
python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
Pop-Location
```

In another terminal, start the UI:

```powershell
$env:API_BASE_URL = "http://localhost:8000"
python -m streamlit run app/streamlit_app.py
```

The API health endpoint is `http://localhost:8000/health` and interactive
documentation is available at `/docs`.

## Docker deployment

The included Compose file runs both services and persists application data in
the `rag-data` volume:

```bash
docker compose up --build -d
```

Open the UI at `http://localhost:8501`. For production, put the UI and API
behind a TLS-terminating reverse proxy, provide secrets through the platform's
secret manager, and back up the persistent `rag-data` volume. The default API
container uses one worker because the local SQLite/Chroma storage is
file-backed; scale out only after moving those stores to shared services.

## API endpoints

* `GET /health` - liveness check
* `POST /chat` - answer a question using the configured RAG model
* `POST /upload-doc` - upload and index a supported document
* `GET /list-docs` - list indexed documents
* `POST /delete-doc` - remove a document and its vectors
