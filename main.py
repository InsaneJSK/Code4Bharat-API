from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from yt_search import get_top_videos
import uvicorn

app = FastAPI(title="YouTube Search Scoring API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/yt-search")
async def yt_search(query: str = Query(..., description="Search query")):
    return {"results": get_top_videos(query)}


# 🚀 Run with: python main.py
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
