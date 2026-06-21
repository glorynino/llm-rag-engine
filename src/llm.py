from llama_cpp import Llama
from typing import Dict, Any

class LLMClient:
    def __init__(self, model_path: str = "models/qwen2.5-3b-instruct-q4_k_m.gguf"):
        self.model = Llama(
            model_path=model_path,
            n_ctx=4096,
            n_threads=2,
            verbose=False,
            chat_format="chatml"
        )

    def generate(self, context: str, question: str) -> str:
        response = self.model.create_chat_completion(  # type: ignore
            messages=[
                {
                    "role": "system",
                    "content": "Tu es un assistant d'entreprise. Réponds UNIQUEMENT en te basant sur le contexte fourni. Si la réponse n'est pas dans le contexte, réponds : 'Je ne dispose pas de cette information dans les documents de l'entreprise.'"
                },
                {
                    "role": "user",
                    "content": f"Contexte :\n{context}\n\nQuestion : {question}"
                }
            ],
            max_tokens=512,
            temperature=0.2,
            stream=False
        )
        content = response["choices"][0]["message"]["content"]  # type: ignore
        return content.strip() if content else ""
