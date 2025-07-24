from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from yt_search import get_top_videos
from chapter_upserter import upsert_chapter_text
import uvicorn
from chat_ncert import run_chatbot
from pydantic import BaseModel
import os
from chat_title import chat_title

os.environ["TRANSFORMERS_CACHE"] = "./hf_cache"
os.environ["HF_HOME"] = "./hf_home"

class ChatRequest(BaseModel):
    messages: list[dict]
    user_input: str
    cid: str

app = FastAPI()

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

@app.get("/upsert-chapter")
def upsert_chapter(
    class_num: int = Query(..., ge=1, le=12),
    subject: str = Query(...),
    chapter: str = Query(...)
):
    result = upsert_chapter_text(class_num, subject, chapter)
    return result

@app.post("/chat-ncert")
def chat_ncert_endpoint(payload: ChatRequest):
    return {"response": run_chatbot(payload.messages, payload.user_input, payload.cid)}

@app.get("/chat-title")
def chat_title_endpoint(
    user_input: str = Query(...),
    llm_response: str = Query(...)
):
    return {"title": chat_title(user_input=user_input, llm_response=llm_response)}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
