from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredHTMLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from typing import List
from langchain_core.documents import Document
import logging
from settings import CHROMA_PATH

logger = logging.getLogger(__name__)


embedding_function = HuggingFaceEmbeddings(
    model_name="intfloat/e5-large-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)


# Text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
)

# Chroma vector store
vectorstore = Chroma(
    persist_directory=str(CHROMA_PATH),
    embedding_function=embedding_function
)

def load_and_split_document(file_path: str) -> List[Document]:
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".docx"):
        loader = Docx2txtLoader(file_path)
    elif file_path.endswith(".html"):
        loader = UnstructuredHTMLLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path}")

    documents = loader.load()
    return text_splitter.split_documents(documents)

def index_document_to_chroma(file_path: str, file_id: int) -> bool:
    try:
        splits = load_and_split_document(file_path)

        for split in splits:
            split.metadata["file_id"] = file_id

        vectorstore.add_documents(splits)
        return True
    except Exception:
        logger.exception("Error indexing document %s", file_path)
        try:
            vectorstore._collection.delete(where={"file_id": file_id})
        except Exception:
            logger.exception("Failed to clean up partial vectors for document %s", file_id)
        return False

def delete_doc_from_chroma(file_id: int):
    try:
        vectorstore._collection.delete(where={"file_id": file_id})
        return True
    except Exception:
        logger.exception("Error deleting vectors for document %s", file_id)
        return False

def remove_orphaned_vectors(document_ids: set[int]) -> None:
    try:
        stored = vectorstore._collection.get(include=["metadatas"])
        orphaned_ids = {
            metadata.get("file_id")
            for metadata in stored.get("metadatas", [])
            if metadata and metadata.get("file_id") not in document_ids
        }
        for file_id in orphaned_ids:
            if file_id is not None:
                delete_doc_from_chroma(file_id)
                logger.info("Removed orphaned vectors for document %s", file_id)
    except Exception:
        logger.exception("Failed to reconcile orphaned document vectors")

retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
