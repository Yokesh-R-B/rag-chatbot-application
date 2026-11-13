from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic_models import QueryInput, QueryResponse, DocumentInfo, DeleteFileRequest
from langchain_utils import get_rag_chain
from db_utils import (
    insert_application_logs,
    get_chat_history,
    get_all_documents,
    get_document_ids,
    insert_document_record,
    delete_document_record,
)
from chroma_utils import index_document_to_chroma, delete_doc_from_chroma, remove_orphaned_vectors
import os
import uuid
import logging
import tempfile
from pathlib import Path
from settings import ALLOWED_ORIGINS, GROQ_API_KEY, LOG_LEVEL, MAX_UPLOAD_BYTES

# Set up logging
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="RAG Chatbot API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

if not GROQ_API_KEY:
    logger.warning("GROQ_API_KEY is not configured; chat requests will fail until it is set")

remove_orphaned_vectors(get_document_ids())

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat", response_model=QueryResponse)
def chat(query_input: QueryInput):
    session_id = query_input.session_id or str(uuid.uuid4())
    logger.info("Chat request received for session %s", session_id)

    try:
        chat_history = get_chat_history(session_id)
        rag_chain = get_rag_chain(query_input.model.value)
        answer = rag_chain.invoke({
            "input": query_input.question,
            "chat_history": chat_history
        })
        insert_application_logs(session_id, query_input.question, answer, query_input.model.value)
    except Exception:
        logger.exception("Chat request failed for session %s", session_id)
        raise HTTPException(status_code=502, detail="The language model service is unavailable.")

    return QueryResponse(answer=answer, session_id=session_id, model=query_input.model)

@app.post("/upload-doc")
def upload_and_index_document(file: UploadFile = File(...)):
    allowed_extensions = ['.pdf', '.docx', '.html']
    filename = Path(file.filename or "upload").name
    file_extension = Path(filename).suffix.lower()
    if file_extension not in allowed_extensions:
        raise HTTPException(status_code=400, detail="Unsupported file type.")

    temp_file_path = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix="rag-upload-", suffix=file_extension, delete=False
        ) as buffer:
            temp_file_path = buffer.name
            total_bytes = 0
            while chunk := file.file.read(1024 * 1024):
                total_bytes += len(chunk)
                if total_bytes > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="Uploaded file is too large.")
                buffer.write(chunk)

        file_id = insert_document_record(filename)
        success = index_document_to_chroma(temp_file_path, file_id)

        if success:
            return {"message": f"File {filename} has been successfully uploaded and indexed.", "file_id": file_id}
        else:
            delete_document_record(file_id)
            raise HTTPException(status_code=500, detail=f"Failed to index {filename}.")
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@app.get("/list-docs", response_model=list[DocumentInfo])
def list_documents():
    return get_all_documents()

@app.post("/delete-doc")
def delete_document(request: DeleteFileRequest):
    chroma_delete_success = delete_doc_from_chroma(request.file_id)

    if chroma_delete_success:
        db_delete_success = delete_document_record(request.file_id)
        if db_delete_success:
            return {"message": f"Successfully deleted document with file_id {request.file_id} from the system."}
        else:
            return {"error": f"Deleted from Chroma but failed to delete document with file_id {request.file_id} from the database."}
    else:
        return {"error": f"Failed to delete document with file_id {request.file_id} from Chroma."}
