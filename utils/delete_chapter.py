import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

# Load env
load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "ncert-chapters"

client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

def delete_chapter(chapter_id: str):
    filter_obj = Filter(
        must=[FieldCondition(key="chapter_id", match=MatchValue(value=chapter_id))]
    )

    client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=filter_obj
    )
    print(f"Deleted all points with chapter_id: {chapter_id}")

if __name__ == "__main__":
    cid = input("Enter chapter_id to delete: ")
    delete_chapter(cid)
