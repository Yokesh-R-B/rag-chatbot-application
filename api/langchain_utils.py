from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from operator import itemgetter
from chroma_utils import vectorstore

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 6, "fetch_k": 30, "lambda_mult": 0.5},
)

# -------------------------
# 1. Standalone Question Rewriter
# -------------------------
contextualize_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "reformulate it into a standalone question. "
    "Do NOT answer the question."
)

contextualize_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

# -------------------------
# 2. RAG QA Prompt
# -------------------------
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use ONLY the provided context."),
    ("system", "Context:\n{context}"),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

# -------------------------
# 3. Build the Chain (LCEL)
# -------------------------
def format_docs(docs):
    return "\n\n".join(
        f"[Page {doc.metadata.get('page_label', doc.metadata.get('page', '?'))}]\n{doc.page_content}"
        for doc in docs
    )

def retrieve_context(question):
    normalized = question.lower().strip()
    overview_query = any(
        phrase in normalized
        for phrase in (
            "what is it about",
            "what is this about",
            "what is the thesis about",
            "summarize the thesis",
            "summary of the thesis",
            "give me an overview",
        )
    )
    if not overview_query:
        return retriever.invoke(question)

    queries = (
        question,
        "thesis title contents introduction problem statement objectives and scope",
        "thesis abstract contribution significance conclusion key outcomes",
    )
    documents = []
    seen = set()
    for query in queries:
        for document in vectorstore.max_marginal_relevance_search(
            query, k=3, fetch_k=30, lambda_mult=0.5
        ):
            key = (document.metadata.get("file_id"), document.metadata.get("page"))
            if key not in seen:
                seen.add(key)
                documents.append(document)
    return documents[:8]

def get_rag_chain(model="openai/gpt-oss-20b"):
    llm = ChatGroq(model=model)

    # Reformulate Question
    standalone_question = contextualize_prompt | llm | StrOutputParser()

    # Retrieve docs based on standalone question
    retriever_chain = standalone_question | RunnableLambda(retrieve_context) | format_docs

    # Final QA
    rag_chain = (
        {
            "context": retriever_chain,
            "input": itemgetter("input"),
            "chat_history": itemgetter("chat_history")
        }
        | qa_prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain
