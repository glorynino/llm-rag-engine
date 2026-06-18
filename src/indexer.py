from langchain_text_splitters import RecursiveCharacterTextSplitter
from loader import load_all_documents
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import pickle
import os

def split_documents(docs: list) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50
    )
    all_chunks = []
    for doc in docs:
        chunks = splitter.split_text(doc["text"])
        for chunk in chunks:
            all_chunks.append({
                "text": chunk,
                "source": doc["filename"]
            })
    return all_chunks

def build_embeddings(chunks: list):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [c["text"] for c in chunks]
    vectors = model.encode(texts, show_progress_bar=True)
    return np.array(vectors, dtype="float32")

def build_faiss_index(chunks: list, vectors):
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)
    
    os.makedirs("../index", exist_ok=True)
    faiss.write_index(index, "../index/faiss.index")
    
    with open("../index/chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)
    
    print(f"✅ Index sauvegardé : {len(chunks)} chunks")

if __name__ == "__main__":
    docs = load_all_documents("../docs/")
    chunks = split_documents(docs)
    print(f"Chunks : {len(chunks)}")
    vectors = build_embeddings(chunks)
    print(f"Vecteurs : {vectors.shape}")
    build_faiss_index(chunks, vectors)