import os
from qdrant_client import QdrantClient
from dotenv import load_dotenv

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "ncert-chapters"

client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)

def list_chapter_chunks():
    chapter_counts = {}

    scroll_offset = None
    while True:
        response = client.scroll(
            collection_name=COLLECTION_NAME,
            with_payload=True,
            limit=100,
            offset=scroll_offset
        )

        points, scroll_offset = response

        for point in points:
            cid = point.payload.get("chapter_id")
            if cid:
                chapter_counts[cid] = chapter_counts.get(cid, 0) + 1

        if not scroll_offset:
            break

    print(f"\nFound {len(chapter_counts)} chapters:\n")
    for chapter_id, count in sorted(chapter_counts.items()):
        print(f"{chapter_id} -> {count} chunk(s)")

if __name__ == "__main__":
    list_chapter_chunks()