# 🎬 YouTube Search Scoring API (FastAPI)

This is a lightweight API built with **FastAPI** that allows you to fetch the **top 3 most relevant YouTube videos** for a given search query — based on a custom relevance formula using views and age of video.

## 🧠 Scoring Formula

Each video is scored using:

```bash
score = views - (1.01 ^ days_old)
```

This favors high-view videos that are also more recent.

---

## 🔧 Setup Instructions

### 1. Clone this repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2. Create a virtual environment (optional)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the FastAPI server

uvicorn main:app --reload
