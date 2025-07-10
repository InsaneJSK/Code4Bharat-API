from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import yt_dlp

def days_old(upload_date_str):
    try:
        upload_date = datetime.strptime(upload_date_str, "%Y%m%d")
        return (datetime.now() - upload_date).days
    except:
        return 0

def compute_score(views, days):
    return views - (1.01 ** days)

def process_video(entry):
    try:
        views = entry.get('view_count', 0)
        upload_date = entry.get('upload_date', '19700101')
        days = days_old(upload_date)
        score = compute_score(views, days)
        return {
            'title': entry.get('title'),
            'url': entry.get('webpage_url'),
            'views': views,
            'upload_date': upload_date,
            'days_old': days,
            'score': round(score, 2),
            'channel': entry.get('uploader')
        }
    except:
        return None

def get_top_videos(query: str, limit: int = 6, top_n: int = 3):
    ydl_opts = {
        'quiet': True,
        'noplaylist': True,
        'extract_flat': False,
        'forcejson': True,
        'simulate': True,
        'headers': {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'
        }
    }

    search_url = f"ytsearch{limit}:{query}"
    videos = []

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            search_results = ydl.extract_info(search_url, download=False)['entries']
        except Exception:
            return []

    with ThreadPoolExecutor() as executor:
        results = list(executor.map(process_video, search_results))

    results = [r for r in results if r]
    return sorted(results, key=lambda v: v['score'], reverse=True)[:top_n]


# 🧪 CLI usage
if __name__ == "__main__":
    query = input("Enter YouTube search query: ")
    results = get_top_videos(query)
    print(results)