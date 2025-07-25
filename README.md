# 🧠 Code4Bharat API (FastAPI-based)

This project is a modular FastAPI backend that powers a smart educational assistant. It includes:

- 🎬 **YouTube Search API** — find top YouTube videos using custom scoring.
- 📚 **Chapter Upserter** — upload and embed NCERT chapters to a vector store by finding the requested pdf for it on its own.
- 💬 **Chat-NCERT** — chat interface that answers questions strictly from a selected NCERT chapter.

---

## 📦 Modules Overview

### 🎬 YouTube Search Scoring API

Fetches the **top 3 most relevant YouTube videos** for a given query using a custom score:

```bash
score = views - (1.01 ^ days_old)
```

This favors high-view, recent videos.

- Endpoint: `/yt-search`
- Input: `query` (string)
- Output: Top 3 video titles, URLs, views, published date, and score

---

### 📚 Chapter Upserter

- Parses and splits NCERT chapter content (JSON format).
- Embeds content using HuggingFace Sentence Transformers.
- Stores the embeddings in **Qdrant Cloud** (with metadata per chapter).
- Endpoint: `/upsert-chapter`
- Input: `class_num` (int), `subject` (str), `chapter` (str)
- Output: Upsert status and number of chunks

Used for enabling chapter-specific retrieval in Chat-NCERT.

---

### 💬 Chat-NCERT

- Retrieves only chunks from the selected chapter (via Qdrant `filter`).
- Sends them to an LLM (like OpenAI or Groq-compatible) for answering.
- Adds system prompt to **restrict answers to the chapter only**.
- Ensures no hallucination from other chapters or prior knowledge.

---

## 🔧 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/InsaneJSK/Code4Bharat-API.git
cd Code4Bharat-API
```

### 2. Create a Virtual Environment (Optional)

```bash
python -m venv venv
venv\Scripts\activate  # Linux: source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up API Keys

Create a `.env` file in the root directory (refer to `.env.dist` for format) and provide the following:

### 5. Run the FastAPI Server

```bash
uvicorn main:app --reload
```

---

## 📊 LLM Evaluations

### RAGAS Evaluation (Factual QA)

Achieved an average of:

- Faithfulness: 0.91
- Answer Relevancy: 0.87
- Context Precision: 0.77
- Context Recall: 0.98

These indicate highly grounded, accurate answers with excellent use of retrieved context. However, the relatively lower context precision suggests some irrelevant content is also being retrieved.

To improve this, we could refine retrieval by incorporating tag-based filtering, chunk-level scoring, or better passage ranking logic to prioritize tightly matched content.

---

|    |   faithfulness |   answer_relevancy |   context_precision |   context_recall |
|---:|---------------:|-------------------:|--------------------:|-----------------:|
|  0 |           0.71 |               0.98 |                0.5  |              1   |
|  1 |           1    |               0.9  |                0.7  |              0.8 |
|  2 |           0.75 |               0.97 |                1    |              1   |
|  3 |           1    |               0.32 |                0.83 |              1   |
|  4 |           1    |               0.88 |                1    |              1   |
|  5 |           0.83 |               0.93 |                1    |              1   |
|  6 |           0.93 |               0.93 |                0.75 |              1   |
|  7 |           0.86 |               0.94 |                1    |              1   |
|  8 |           1    |               0.94 |                0.37 |              1   |
|  9 |           1    |               0.92 |                0.5  |              1   |

---

### LLM-as-a-Judge Evaluation (Open-ended QA)

- Used an LLM to score and evaluate descriptive and literary answers on a 0–100 scale based on alignment with excepted answers.
- Cached responses and strict scoring prompt ensure repeatable, fair grading.

|    |   llm_score |
|---:|------------:|
|  0 |          90 |
|  1 |          75 |
|  2 |          90 |
|  3 |          90 |
|  4 |          75 |

- Clearly, by fixating on factual accuracy, creative accuracy of the model suffers.

- Would be great if we separate the models for both the answers and perhaps use an agent to pick which model to go for depending upon the type of question.
