from retriever import Retriever
from llm import LLMClient

class Pipeline:
    def __init__(self):
        self.retriever = Retriever()
        self.llm_client = LLMClient()

    def ask(self, question: str) -> dict:
        chunks = self.retriever.search(question)

        if not chunks:
            return {
                "answer": "Je ne dispose pas de cette information dans les documents de l'entreprise.",
                "sources": []
            }

        context = "\n\n".join(
            f"[Source: {chunk['source']}]\n{chunk['text']}"
            for chunk in chunks
        )

        answer = self.llm_client.generate(context, question)

        return {
            "answer": answer,
            "sources": list(set(chunk["source"] for chunk in chunks))
        }
