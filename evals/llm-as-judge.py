import json
import os
import requests
import pandas as pd
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate

load_dotenv()

# === Config ===
EVAL_DATA_PATH = "evals/llm-as-judge-input.json"
CACHE_PATH = "evals/llm_cached_answers.json"
OUTPUT_CSV = "evals/llm_as_judge_eval.csv"
HF_SPACE_URL = "https://InsaneJSK-Code4Bharat-API.hf.space/chat-ncert"
MODEL_NAME = "gemma2-9b-it"

# === Setup LLM ===
llm = ChatGroq(
    model_name=MODEL_NAME,
    temperature=0
)

score_prompt = ChatPromptTemplate.from_messages([
    (
        "system", 
        """You are an evaluator. You will be given a question, a ground truth answer from the NCERT book, and a model-generated answer.
        Return a score between 0 and 100 based on how appropriate, correct, relevant, creative and close the model answer is with respect to the question and ground truth."""
    ),
    (
        "human", 
        """Question: {question}
        
        Model Answer: {answer}
        
        Ground Truth: {ground_truth}
        
        Give only a float score between 0 and 100 in JSON format like: {{\"score\": 95}}""")
])

# === Load evaluation data ===
with open(EVAL_DATA_PATH, "r", encoding="utf-8") as f:
    eval_data = json.load(f)

# === Load cache if exists ===
if os.path.exists(CACHE_PATH):
    with open(CACHE_PATH, "r", encoding="utf-8") as f:
        cached_answers = json.load(f)
else:
    cached_answers = {}

new_cache = {}
rows = []

for sample in eval_data:
    ques = sample["question"]
    cid = sample.get("cid", "")
    ground_truth = sample["ground_truth"]

    # Step 1: Get Answer (cached or fresh)
    if ques in cached_answers:
        print(f"[✓] Cached answer used for: {cid}")
        answer = cached_answers[ques]["answer"]
    else:
        print(f"[→] Getting answer for: {cid}")
        payload = {
            "messages": [],
            "user_input": ques,
            "cid": cid
        }

        try:
            response = requests.post(HF_SPACE_URL, json=payload).json()
            answer = response.get("response", "")
            new_cache[ques] = {
                "answer": answer
            }
        except Exception as e:
            print(f"[!] Error during RAG API call: {e}")
            continue

    # Step 2: Score with LLM-as-judge
    try:
        prompt = score_prompt.format_messages(
            question=ques,
            answer=answer,
            ground_truth=ground_truth
        )
        response = llm(prompt)
        score = float(json.loads(response.content)["score"])
    except Exception as e:
        print(f"[!] Error during Groq eval for cid={cid}: {e}")
        score = -1.0

    rows.append({
        "cid": cid,
        "question": ques,
        "ground_truth": ground_truth,
        "answer": answer,
        "llm_score": round(score, 3)
    })

# === Save cache ===
cached_answers.update(new_cache)
with open(CACHE_PATH, "w", encoding="utf-8") as f:
    json.dump(cached_answers, f, ensure_ascii=False, indent=2)

# === Save results to CSV ===
df = pd.DataFrame(rows)
df.to_csv(OUTPUT_CSV, index=False)
print("\n[✓] Evaluation complete. Saved to:", OUTPUT_CSV)
