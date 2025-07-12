from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer
from langdetect import detect

class LocalMiniLMEmbedder(Embeddings):
    def __init__(self):
        # Load the MiniLM model
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        clean_texts = []
        for text in texts:
            lang = detect(text)
            if lang == "en":
                clean_texts.append(text)
            else:
                print(f"[embed_documents] Skipping non-English text: {lang}")
        return self.model.encode(clean_texts, show_progress_bar=False).tolist() if clean_texts else []

    def embed_query(self, text: str) -> list[float]:
        lang = detect(text)
        if lang != "en":
            raise ValueError("Hindi support is in development.")
        return self.model.encode([text])[0].tolist()
