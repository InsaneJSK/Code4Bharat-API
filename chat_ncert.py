import os
import requests
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_qdrant import QdrantVectorStore
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from qdrant_client import QdrantClient
from embedder import LocalMiniLMEmbedder
from langdetect import detect

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
EMBEDDER_API_URL = os.getenv("EMBEDDER_API_URL")  # just for clarity

CONDENSE_PROMPT = ChatPromptTemplate.from_template("""
Given the chat history and the latest user question, rewrite the question to be a standalone question.

Chat History:
{chat_history}

Follow Up Input: {input}
Standalone question:
""")

CHAT_PROMPT = ChatPromptTemplate.from_template("""
Use the following context to answer the question. If the context is insufficient, say you don't know.
Never mention the context as context, say NCERT book instead. If possible, tell about the figure, page number, etc.
Context:
{context}

Question:
{input}
""")

def create_chatbot():
    print("🔌 Connecting to Qdrant...")
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

    print("🧠 Loading embedder...")
    embedder = LocalMiniLMEmbedder()

    print("🗂️ Initializing vector store...")
    db = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embedder,
        content_payload_key="text"  # IMPORTANT for retrieving actual text
    )

    print("🔍 Building retriever...")
    basic_retriever = db.as_retriever(search_kwargs={"k": 20}, search_type="similarity")

    print("🤖 Loading LLMs from Groq...")
    llm_main = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama-3.3-70b-versatile")
    llm_light = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama-3.1-8b-instant")

    retriever_with_memory = create_history_aware_retriever(
        llm=llm_light,
        retriever=basic_retriever,
        prompt=CONDENSE_PROMPT
    )

    document_chain = create_stuff_documents_chain(llm_main, CHAT_PROMPT)
    retrieval_chain = create_retrieval_chain(retriever_with_memory, document_chain)

    print("✅ Retrieval chain ready")
    return retrieval_chain

def run_chatbot(messages, user_input, N_TURNS=3):
    try:
        lang = detect(user_input)
        print(f"🌐 Detected language: {lang}")
        if lang == "hi":
            answer = "❗ Hindi support is currently in development. Please ask in English."
            messages.append({"role": "user", "content": user_input})
            messages.append({"role": "assistant", "content": answer})
            return answer
    except Exception as e:
        answer = f"❌ Language detection failed: {str(e)}"
        messages.append({"role": "user", "content": user_input})
        messages.append({"role": "assistant", "content": answer})
        return answer

    chain = create_chatbot()

    history_pairs = [
        (messages[i]["content"], messages[i + 1]["content"])
        for i in range(len(messages) - 2, -1, -2)
    ][::-1][:N_TURNS]

    chat_history_text = "\n".join(
        f"User: {u}\nAssistant: {a}" for u, a in history_pairs
    )

    messages.append({"role": "user", "content": user_input})
    try:
        print("💬 Invoking retrieval chain...")
        response = chain.invoke({
            "input": user_input,
            "chat_history": chat_history_text
        })

        docs = response.get("context", [])
        print(f"📄 Retrieved {len(docs)} docs")
        for i, doc in enumerate(docs):
            print(f"--- Doc {i+1} ---\n{doc.page_content[:300]}\n")

        answer = response["answer"]
    except Exception as e:
        answer = f"❌ Error: {str(e)}"

    messages.append({"role": "assistant", "content": answer})
    return answer


if __name__ == "__main__":
    print("💬 NCERT Chatbot Ready! Type your question or 'exit' to quit.")
    chat_history = []
    while True:
        query = input("👤 You: ")
        if query.lower() in ["exit", "quit"]:
            print("👋 Bye!")
            break
        response = run_chatbot(chat_history, query)
        print("🤖 Bot:", response)
