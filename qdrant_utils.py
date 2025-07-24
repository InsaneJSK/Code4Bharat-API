import os
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, Distance, VectorParams, ScoredPoint
from uuid import uuid4

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "ncert-chapters"

client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)

def ensure_collection():
    if COLLECTION_NAME not in [col.name for col in client.get_collections().collections]:
        client.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="cid",
        field_schema="keyword"
    )

def chapter_id(class_num, subject, chapter):
    return f"class{class_num}_{subject.lower().strip()}_{chapter.lower().strip()}"

def chapter_exists(id: str) -> bool:
    result = client.scroll(collection_name=COLLECTION_NAME, scroll_filter={"must": [{"key": "cid", "match": {"value": id}}]}, limit=1)
    return len(result[0]) > 0

def insert_vectors(id: str, vectors: list[list[float]], texts: list[str]):
    points = [
    PointStruct(
        id=str(uuid4()),
        vector=vec,
        payload={
            "text": text,
            "cid": id  # keep this as-is for filtering
        }
    )
    for vec, text in zip(vectors, texts)
]
    client.upsert(collection_name=COLLECTION_NAME, points=points)
