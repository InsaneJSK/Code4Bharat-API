import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import time
import json

BASE_URL = "https://www.ncertpdf.com"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
}

def get_soup(url):
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        resp.raise_for_status()
        return BeautifulSoup(resp.content, 'lxml')
    except Exception as e:
        print(f"❌ Error fetching {url}: {e}")
        return None

def build_index():
    final_index = []

    for class_num in tqdm(range(1, 13), desc="Processing classes"):
        class_url = f"{BASE_URL}/class-{class_num}"
        soup = get_soup(class_url)
        time.sleep(0.5)
        if not soup:
            continue

        # Find all subject/book pages
        subject_links = soup.find_all('a', class_='group flex flex-col items-center')

        for subject_link in subject_links:
            subject_title = subject_link.text.strip().lower()
            subject_href = subject_link.get('href')
            subject_url = BASE_URL + subject_href

            subject_soup = get_soup(subject_url)
            time.sleep(0.5)
            if not subject_soup:
                continue

            # Find all chapters within subject
            chapter_links = subject_soup.find_all("a", class_="block hover:bg-gray-50 transition duration-150 ease-in-out")

            for chapter_link in chapter_links:
                chapter_title = chapter_link.text.strip()
                chapter_href = chapter_link.get('href')
                chapter_url = BASE_URL + chapter_href

                chapter_soup = get_soup(chapter_url)
                time.sleep(0.5)
                if not chapter_soup:
                    continue

                # Get final PDF link (usually inside a styled button)
                try:
                    pdf_button = chapter_soup.find_all('a', class_="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500")[0]
                    pdf_url = pdf_button.get('href')
                except Exception:
                    pdf_url = None

                if pdf_url and (".pdf" in pdf_url or "drive.google.com" in pdf_url):
                    final_index.append({
                        "class": class_num,
                        "subject": subject_title,
                        "chapter": chapter_title,
                        "pdf_url": pdf_url
                    })

        # Save intermediate result
        with open("ncert_index_partial.json", "w", encoding="utf-8") as f:
            json.dump(final_index, f, indent=2, ensure_ascii=False)

    # Save final index
    with open("ncert_index.json", "w", encoding="utf-8") as f:
        json.dump(final_index, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(final_index)} chapter entries to ncert_index.json")

if __name__ == "__main__":
    build_index()
