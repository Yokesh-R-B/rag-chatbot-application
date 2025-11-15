from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter
from chroma_utils import vectorstore

retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

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
    return "\n\n".join([doc.page_content for doc in docs])

def get_rag_chain(model="llama-3.1-8b-instant"):
    llm = ChatGroq(model=model)

    # Reformulate Question
    standalone_question = contextualize_prompt | llm | StrOutputParser()

    # Retrieve docs based on standalone question
    retriever_chain = standalone_question | retriever | format_docs

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
