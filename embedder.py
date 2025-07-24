from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer
from langdetect import detect
import os
os.environ["TRANSFORMERS_CACHE"] = "./hf_cache"

class LocalMiniLMEmbedder(Embeddings):
    def __init__(self):
        # Load the MiniLM model
        self.model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        clean_texts = []
        for text in texts:
            lang = detect(text)
            clean_texts.append(text)
        return self.model.encode(clean_texts, show_progress_bar=False).tolist() if clean_texts else []

    def embed_query(self, text: str) -> list[float]:
        lang = detect(text)
        return self.model.encode([text])[0].tolist()
