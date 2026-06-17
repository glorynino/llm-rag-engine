from langchain_text_splitters import RecursiveCharacterTextSplitter
from loader import load_all_documents

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

if __name__ == "__main__":
    docs = load_all_documents("../docs/")
    chunks = split_documents(docs)
    print(f"Nombre de chunks : {len(chunks)}")
    print(f"Premier chunk : {chunks[0]['text'][:200]}")
