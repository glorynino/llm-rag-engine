from pathlib import Path
import fitz
from docx import Document

def load_document(filepath: str) -> str:
    path = Path(filepath)
    
    if path.suffix == ".pdf":
        doc = fitz.open(path)
        return "\n".join(page.get_text() for page in doc)
    
    elif path.suffix == ".docx":
        doc = Document(path)
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    
    elif path.suffix == ".txt":
        return path.read_text(encoding="utf-8")
    
    else:
        raise ValueError(f"Format non supporté : {path.suffix}")


def load_all_documents(folder: str) -> list:
    docs = []
    for path in Path(folder).rglob("*"):
        if path.suffix in [".pdf", ".docx", ".txt"]:
            text = load_document(str(path))
            docs.append({"filename": path.name, "text": text})
            print(f"Chargé : {path.name}")
    return docs



