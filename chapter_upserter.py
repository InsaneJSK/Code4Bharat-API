from ncert_parser import find_pdf_url
from embedder import Embedder
from qdrant_utils import ensure_collection, chapter_exists, insert_vectors, chapter_id
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
embedder = Embedder()

def upsert_chapter_text(class_num, subject, chapter):
    ensure_collection()
    cid = chapter_id(class_num, subject, chapter)

    if chapter_exists(cid):
        return {"status": cid}

    full_text = find_pdf_url(class_num, subject, chapter)
    if not full_text or "error" in full_text:
        return {"error": "Could not fetch chapter text"}

    chunks = splitter.split_text(full_text)
    vectors = embedder.embed_texts(chunks)
    insert_vectors(cid, vectors, chunks)

    return {"status": "upserted", "chunks": len(chunks)}
