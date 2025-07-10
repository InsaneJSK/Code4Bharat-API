from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from yt_search import get_top_videos
from chapter_upserter import upsert_chapter_text
import uvicorn

app = FastAPI(title="YouTube Search + NCERT Chapter API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/yt-search")
async def yt_search(query: str = Query(...)):
    return {"results": get_top_videos(query)}

app = FastAPI()

@app.get("/upsert-chapter")
def upsert_chapter(
    class_num: int = Query(..., ge=1, le=12),
    subject: str = Query(...),
    chapter: str = Query(...)
):
    return upsert_chapter_text(class_num, subject, chapter)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
