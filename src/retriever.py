import faiss
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np

class Retriever:
    def __init__(self, index_path: str = "index/faiss.index", chunks_path: str = "index/chunks.pkl"):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.index = faiss.read_index(index_path)
        with open(chunks_path, "rb") as f:
            self.chunks = pickle.load(f)

    def search(self, question: str) -> list:
        question_vec = self.model.encode([question], show_progress_bar=False)
        D, I = self.index.search(np.array(question_vec, dtype="float32"), k=5)
        results = []
        for idx, distance in zip(I[0], D[0]):
            if idx < len(self.chunks) and distance < 2.0:
                results.append(self.chunks[idx])
        return results
