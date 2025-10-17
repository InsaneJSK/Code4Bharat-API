import os
from langchain_groq import ChatGroq
from langchain_qdrant import QdrantVectorStore
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from qdrant_client import QdrantClient
from embedder import LocalMiniLMEmbedder
from langdetect import detect
import re

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

CONDENSE_PROMPT = ChatPromptTemplate.from_template("""
Given the chat history and the latest user question, rewrite the question to be a standalone question.
Chat History:
{chat_history}
Follow Up Input: {input}
Standalone question:
""")

CHAT_PROMPT = ChatPromptTemplate.from_template("""
You are a teaching assistant helping students by answering their questions using only the NCERT book content provided below.
Never refer to the text as "context" — always call it the "NCERT book."
If the question is factual and the answer is clearly present in the NCERT book, respond accurately and concisely. Mention relevant page numbers, figures, or sections wherever possible.
If the question is open-ended, literary, or inferential, you may attempt an answer, but clearly state that this goes beyond what is directly written in the NCERT. Make sure your response still aligns with the level, tone, and theme of the NCERT content and remains age-appropriate.
If the NCERT book does not contain information required to answer the question, or the question is completely irrelevant to the context topics, clearly say:  
**"This answer is not available in the NCERT book."**
Context (NCERT book content):  
{context}
Question:  
{input}
""")

def create_chatbot(cid: str):
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

    embedder = LocalMiniLMEmbedder()

    db = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embedder,
        content_payload_key="text"  # IMPORTANT for retrieving actual text
    )

    basic_retriever = db.as_retriever(
        search_kwargs={
            "k": 5,
            "filter": {
            "must": [
                {"key": "cid", "match": {"value": cid}}
            ]
        }
        },
        search_type="mmr"
    )

    llm_main = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="deepseek-r1-distill-llama-70b")
    llm_light = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="deepseek-r1-distill-llama-70b")

    retriever_with_memory = create_history_aware_retriever(
        llm=llm_light,
        retriever=basic_retriever,
        prompt=CONDENSE_PROMPT
    )

    document_chain = create_stuff_documents_chain(llm_main, CHAT_PROMPT)
    retrieval_chain = create_retrieval_chain(retriever_with_memory, document_chain)

    return retrieval_chain

def run_chatbot(messages, user_input, cid, N_TURNS=3):
    chain = create_chatbot(cid)

    history_pairs = [
        (messages[i]["content"], messages[i + 1]["content"])
        for i in range(len(messages) - 2, -1, -2)
    ][::-1][:N_TURNS]

    chat_history_text = "\n".join(
        f"User: {u}\nAssistant: {a}" for u, a in history_pairs
    )

    messages.append({"role": "user", "content": user_input})
    try:
        response = chain.invoke({
            "input": user_input,
            "chat_history": chat_history_text
        })
        docs = response.get("context", [])
        print(f"📄 Retrieved {len(docs)} documents")
        for i, doc in enumerate(docs):
            snippet = doc.page_content[:200].replace("\n", " ")
            print(f"--- Doc {i+1} ---\n{snippet}\n")

        def extract_final_answer(text: str) -> str:
            return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()

        answer = extract_final_answer(response["answer"])

    except Exception as e:
        answer = f"❌ Error: {str(e)}"
        docs=[]

    messages.append({"role": "assistant", "content": answer})
    return answer, docs


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
