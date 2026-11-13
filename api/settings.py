import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
configured_data_dir = Path(os.getenv("DATA_DIR", str(BASE_DIR)))
DATA_DIR = (
    configured_data_dir
    if configured_data_dir.is_absolute()
    else (BASE_DIR.parent / configured_data_dir).resolve()
)
DATA_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = DATA_DIR / os.getenv("DB_FILENAME", "rag_app.db")
CHROMA_PATH = DATA_DIR / os.getenv("CHROMA_DIRNAME", "chroma_db")
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(25 * 1024 * 1024)))
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv("ALLOWED_ORIGINS", "http://localhost:8501").split(",")
    if origin.strip()
]
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
