import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient

# Load environment variables
load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "ncert-chapters"

# Initialize Qdrant client
client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

def get_texts_by_chapter_id(chapter_id: str):
    results = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter={
            "must": [{"key": "chapter_id", "match": {"value": chapter_id}}]
        },
        limit=1000  # Adjust as needed
    )
    points = results[0]
    return [point.payload["text"] for point in points]

if __name__ == "__main__":
    chapter_id = input("Enter chapter_id (e.g., class10_science_life processes): ")
    texts = get_texts_by_chapter_id(chapter_id)
    
    print(f"\n✅ Found {len(texts)} chunks for: {chapter_id}")
    input("press enter to continue")
    if len(texts) >= 100:
        for i, chunk in enumerate(texts[:100], 1):
            print(f"\n--- Chunk {i} ---\n{chunk}\n")
    else:
        for i, chunk in enumerate(texts, 1):
            print(f"\n--- Chunk {i} ---\n{chunk}\n")