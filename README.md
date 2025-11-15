# RAG Chatbot (FastAPI + Streamlit)

This is a simple RAG (Retrieval-Augmented Generation) chatbot where users can upload documents (PDF, DOCX, HTML) and ask questions from them.
The backend is built using **FastAPI**, and the frontend uses **Streamlit**.
ChromaDB is used to store embeddings, and Groq LLaMA 3.1 is used for answering questions.

## Features

* Upload documents and store them in a local ChromaDB
* Ask questions based on uploaded content
* Follow-up questions supported
* Chat history stored in SQLite
* API routes built using FastAPI
* Simple Streamlit UI

## Setup Instructions

### **1. Clone the repository**

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
```

### **2. Create virtual environment**

```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

### **3. Install dependencies**

```bash
pip install -r requirements.txt
```

### **4. Add your API keys**

Create a `.env` file:

```
GROQ_API_KEY=your_groq_key
LANGCHAIN_API_KEY=your_langchain_key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=RAG_Chatbot
```

## Running the Project

### **Start Backend**

```bash
cd api
python -m uvicorn main:app --reload
```

Backend runs at:

```
http://127.0.0.1:8000/docs
```

### **Start Frontend**

Open a new terminal:

```bash
cd frontend
streamlit run app.py
```

## What I Learned

* How RAG works end-to-end
* Using different LLM models with LangChain
* Storing embeddings using ChromaDB
* Building endpoints with FastAPI
* Creating a frontend using Streamlit

