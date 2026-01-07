import os
import re
from typing import Optional

from langchain_groq import ChatGroq
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from embedder import LocalMiniLMEmbedder

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


def format_profile_context(profile: Optional[dict]) -> str:
    """
    Convert onboarding profile into a small text block that will be prepended
    to the LLM prompt (only at the generation stage).
    """
    if not profile:
        return ""

    eq_score = profile.get("eq_score", "N/A")
    eq_level = profile.get("eq_level", "Unknown")
    learning = profile.get("learning_style", {})

    if isinstance(eq_level, str):
        eq_level_l = eq_level.lower()
    else:
        eq_level_l = "unknown"

    if eq_level_l == "low":
        eq_support_msg = "User requires more emotional support and encouragement."
    elif eq_level_l == "moderate":
        eq_support_msg = "User requires some emotional support during learning."
    elif eq_level_l == "high":
        eq_support_msg = "User requires minimal emotional support."
    else:
        eq_support_msg = "EQ support needs unknown."

    learning_desc = []
    for key, value in learning.items():
        learning_desc.append(f"{key.capitalize()}: {value}")

    return (
        f"User Profile:\n"
        f"- EQ Score: {eq_score} ({eq_level}) → {eq_support_msg}\n"
        f"- Learning Style:\n  " + "\n  ".join(learning_desc) + "\n\n"
    )


def create_chatbot_components(cid: str):
    """
    Create retriever and LLM component for a given cid.
    We return the retriever and llm_main so run_chatbot can:
      1) retrieve using raw user_input
      2) then call LLM with retrieved docs + profile
    """
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    embedder = LocalMiniLMEmbedder()

    db = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embedder,
        content_payload_key="text",
    )

    retriever = db.as_retriever(
        search_kwargs={
            "k": 5,
            "filter": {
                "must": [
                    {"key": "cid", "match": {"value": cid}}
                ]
            }
        },
        search_type="mmr",
    )

    llm_main = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="groq/compound")

    return retriever, llm_main


def run_chatbot(messages: list, user_input: str, cid: str, profile: Optional[dict] = None, N_TURNS: int = 3):
    """
    1) Retrieve relevant NCERT docs using the RAW user_input (no profile injected here).
    2) If docs found: build final prompt including profile_context and docs, then call LLM.
    3) If no docs found: return the required message "This answer is not available in the NCERT book."
    """
    retriever, llm_main = create_chatbot_components(cid)
    profile_context = format_profile_context(profile)

    # Build recent chat history (if any) for context in the final prompt
    history_pairs = []
    for i in range(len(messages) - 2, -1, -2):
        if i + 1 < len(messages):
            history_pairs.append((messages[i]["content"], messages[i + 1]["content"]))
    history_pairs = history_pairs[::-1][:N_TURNS]

    chat_history_text = "\n".join(f"User: {u}\nAssistant: {a}" for u, a in history_pairs)

    # Append user message to the chat history list (for local state)
    messages.append({"role": "user", "content": user_input})

    # ---------- Retrieval (ONLY on raw user_input) ----------
    try:
        docs = retriever.invoke(user_input)
    except Exception as e:
        # retrieval failed — surface an error but avoid silent failure
        answer = f"❌ Retrieval Error: {e}"
        print(answer)
        return answer, []

    # If retrieval returns 0 documents -> obey your instruction and return the standard message
    if not docs:
        answer = "This answer is not available in the NCERT book."
        messages.append({"role": "assistant", "content": answer})
        print("📄 Retrieved 0 documents; returning NCERT-not-available message.")
        return answer, []

    # Debug: show retrieved doc count & short snippets
    print(f"📄 Retrieved {len(docs)} documents")
    for i, doc in enumerate(docs):
        snippet = getattr(doc, "page_content", str(doc))[:200].replace("\n", " ")
        print(f"--- Doc {i+1} ---\n{snippet}\n")

    # Compose the NCERT context from retrieved docs
    combined_context = "\n\n".join(getattr(d, "page_content", str(d)) for d in docs)

    # ---------- Final prompt for the LLM (profile injected here) ----------
    final_prompt = f"""
You are a teaching assistant helping students by answering their questions using only the NCERT book content provided below.

{profile_context}
Never refer to the text as "context" — always call it the "NCERT book."
If the question is factual and the answer is clearly present in the NCERT book, respond accurately and concisely. Mention relevant page numbers, figures, or sections wherever possible.
If the question is open-ended, literary, or inferential, you may attempt an answer, but clearly state that this goes beyond what is directly written in the NCERT. Make sure your response still aligns with the level, tone, and theme of the NCERT content and remains age-appropriate.
If the NCERT book does not contain information required to answer the question, or the question is completely irrelevant to the context topics, clearly say:
"This answer is not available in the NCERT book."

Chat history (most recent turns):
{chat_history_text}

NCERT Book Content (retrieved):
{combined_context}

Question:
{user_input}

Provide the final answer only (no extra commentary). If you need to show step-wise reasoning, enclose it inside <think>...</think> tags so the caller can strip it.
"""

    # Call the LLM with the final prompt
    try:
        llm_response = llm_main.invoke(final_prompt)
        raw_text = getattr(llm_response, "content", str(llm_response)).strip()

        # Remove any <think> inner blocks before returning as final answer (as you used earlier)
        def extract_final_answer(text: str) -> str:
            return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()

        answer = extract_final_answer(raw_text)

    except Exception as e:
        answer = f"❌ Model Error: {str(e)}"
        docs = []
        print(answer)

    # Save assistant response to chat history
    messages.append({"role": "assistant", "content": answer})
    return answer, docs


# Quick CLI test helper
if __name__ == "__main__":
    print("💬 NCERT Chatbot Ready! Type your question or 'exit' to quit.")
    chat_history = []
    while True:
        query = input("👤 You: ")
        if query.lower() in ["exit", "quit"]:
            print("👋 Bye!")
            break
        # example: run with profile=None locally
        answer, docs = run_chatbot(chat_history, query, cid="class10_science_chapter_11", profile=None)
        print("🤖 Bot:", answer)
