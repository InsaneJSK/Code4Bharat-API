from ncert_parser import find_pdf_url, extract_text_from_pdf_url
from qdrant_utils import ensure_collection, chapter_exists, insert_vectors, chapter_id
from langchain.text_splitter import RecursiveCharacterTextSplitter
from embedder import LocalMiniLMEmbedder
from langdetect import detect

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
embedder = LocalMiniLMEmbedder()

def upsert_chapter_text(class_num, subject, chapter):
    ensure_collection()
    cid = chapter_id(class_num, subject, chapter)

    if chapter_exists(cid):
        return {
            "status": "Already upserted",
            "cid" : cid
        }

    full_text = extract_text_from_pdf_url(find_pdf_url(class_num, subject, chapter))
    if not full_text or "error" in full_text:
        return {"error": "Could not fetch chapter text"}

    chunks = splitter.split_text(full_text)
    vectors = embedder.embed_documents(chunks)
    insert_vectors(cid, vectors, chunks)

    return {
        "status": "upserted",
        "chunks": len(chunks),
        "cid" : cid
    }
