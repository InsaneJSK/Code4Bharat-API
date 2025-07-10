import json
import requests
import fitz  # PyMuPDF
import io

with open("ncert_index_final.json", "r", encoding="utf-8") as f:
    ncert_index = json.load(f)

def find_pdf_url(class_num: int, subject: str, chapter_name: str):
    subject = subject.lower().strip()
    chapter_name = chapter_name.strip()

    for entry in ncert_index:
        if (entry["class"] == class_num and
            subject in entry["subject"] and
            chapter_name.lower() in entry["chapter"].lower()):
            return entry["pdf_url"]

    return None

def extract_text_from_pdf_url(pdf_url: str) -> str:
    try:
        print(f"Downloading PDF: {pdf_url}")
        response = requests.get(pdf_url, timeout=15)
        response.raise_for_status()

        pdf_data = io.BytesIO(response.content)
        doc = fitz.open(stream=pdf_data, filetype="pdf")

        full_text = ""
        for page in doc:
            full_text += page.get_text()

        doc.close()
        return full_text.strip()

    except Exception as e:
        print(f"Failed to extract text: {e}")
        return ""

if __name__ == "__main__":
    class_num = int(input("Enter class num: "))
    subject = input("Enter subject: ")
    chapter_name = input("Enter chapter name: ")
    url = find_pdf_url(class_num, subject, chapter_name)
    if url:
        print("PDF found:", url)
        text = extract_text_from_pdf_url(url)
        print("Extracted text preview:\n", text[:500])
    else:
        print("No match found")
