import faiss
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np


class Retriever:
    def __init__(self, index_path: str = "../index/faiss.index", chunks_path: str = "../index/chunks.pkl"):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.index = faiss.read_index(index_path)
        with open(chunks_path, "rb") as f:
            self.chunks = pickle.load(f)

    def search(self, question: str) -> list:
        question_vec = self.model.encode([question], show_progress_bar=False)
        D, I = self.index.search(np.array(question_vec, dtype="float32"), k=3)
        
        results = []
        for idx in I[0]:
            if idx < len(self.chunks):
                results.append(self.chunks[idx])
        
        return results


if __name__ == "__main__":
    r = Retriever()
    results = r.search("What is network segmentation?")
    for chunk in results:
        print(f"\n📄 Source: {chunk['source']}")
        print(chunk['text'][:200])
