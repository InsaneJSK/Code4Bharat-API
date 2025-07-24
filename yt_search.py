from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from googleapiclient.discovery import build
import os

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
YOUTUBE_API_SERVICE_NAME = "youtube"
YOUTUBE_API_VERSION = "v3"

def days_old(upload_date_iso):
    try:
        upload_date = datetime.strptime(upload_date_iso, "%Y-%m-%dT%H:%M:%SZ")
        return (datetime.utcnow() - upload_date).days
    except:
        return 0

def compute_score(views, days):
    return views - (1.01 ** days)

def get_video_details(youtube, video_ids):
    response = youtube.videos().list(
        part="snippet,statistics",
        id=",".join(video_ids)
    ).execute()
    return response.get("items", [])

def process_video(item):
    try:
        snippet = item["snippet"]
        stats = item.get("statistics", {})
        views = int(stats.get("viewCount", 0))
        upload_date = snippet.get("publishedAt")
        days = days_old(upload_date)
        score = compute_score(views, days)

        return {
            'title': snippet.get("title"),
            'url': f"https://www.youtube.com/watch?v={item['id']}",
            'views': views,
            'upload_date': upload_date,
            'days_old': days,
            'score': round(score, 2),
            'channel': snippet.get("channelTitle")
        }
    except:
        return None

def get_top_videos(query: str, limit: int = 6, top_n: int = 3):
    youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=YOUTUBE_API_KEY)

    # Step 1: Search for videos
    search_response = youtube.search().list(
        q=query,
        part="id",
        type="video",
        maxResults=limit
    ).execute()

    video_ids = [item["id"]["videoId"] for item in search_response.get("items", [])]

    # Step 2: Get video details
    if not video_ids:
        return []

    video_items = get_video_details(youtube, video_ids)

    # Step 3: Process and score
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(process_video, video_items))

    results = [r for r in results if r]
    return sorted(results, key=lambda v: v['score'], reverse=True)[:top_n]


# CLI usage
if __name__ == "__main__":
    query = input("Enter YouTube search query: ")
    results = get_top_videos(query)
    for r in results:
        print(r)
