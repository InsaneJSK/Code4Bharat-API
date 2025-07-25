import json
import os
import requests
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

load_dotenv()

# Use langchain_groq LLM
llm = ChatGroq(
    model_name="meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=0
)

# Constants
EVAL_DATA_PATH = "evals/redo.json"
CACHE_PATH = "evals/cached_factual_answers.json"
HF_SPACE_URL = "https://InsaneJSK-Code4Bharat-API.hf.space/chat-ncert"

# Load eval data (your ground truth questions, etc.)
with open(EVAL_DATA_PATH, "r", encoding="utf-8") as f:
    eval_data = json.load(f)

# Load cache if it exists
if os.path.exists(CACHE_PATH):
    with open(CACHE_PATH, "r", encoding="utf-8") as f:
        cached_answers = json.load(f)
else:
    cached_answers = {}

# Lists for dataset construction
questions, answers, contexts, ground_truths = [], [], [], []
new_cache = {}

for sample in eval_data:
    ques = sample["question"]

    if ques in cached_answers:
        print(f"[✓] Using cached response for: {ques}")
        ans = cached_answers[ques]["answer"]
        ctx = cached_answers[ques]["contexts"]
    else:
        print(f"[→] Querying LLM for: {ques}")
        payload = {
            "messages": [],
            "user_input": ques,
            "cid": sample.get("cid", "")
        }

        try:
            response = requests.post(HF_SPACE_URL, json=payload).json()
            ans = response.get("response", '')
            docs = response.get("docs", [])
            ctx = [doc.get("page_content", "") for doc in docs]

        except Exception as e:
            print(f"[!] Error during API call: {e}")
            continue

        # Save to new cache
        new_cache[ques] = {
            "answer": ans,
            "contexts": ctx
        }

    # Add to final dataset lists
    questions.append(ques)
    answers.append(ans)
    contexts.append(ctx)
    ground_truths.append(sample["ground_truth"])

# Update and save cache
cached_answers.update(new_cache)
with open(CACHE_PATH, "w", encoding="utf-8") as f:
    json.dump(cached_answers, f, ensure_ascii=False, indent=2)

# Construct HuggingFace Dataset
dataset = Dataset.from_dict({
    "question": questions,
    "answer": answers,
    "contexts": contexts,
    "ground_truth": ground_truths
})

# Evaluate using RAGAS
eval_result = evaluate(
    dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall
    ],
    llm=llm,
    embeddings=embedding_model
)

print("\n===== Evaluation Result =====")
print(eval_result.to_pandas())
eval_result.to_pandas().to_csv("evals/redo-eval.csv")
