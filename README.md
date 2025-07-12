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
