from llama_cpp import Llama 

class model:
    def __init__(self,model_path: str = "../models/qwen2.5-1.5b-instruct-q4_k_m.gguf"):
        self.model = Llama(model_path=model_path, n_ctx=2048, n_threads=2, verbose = False)
    
    def generate(self, prompt: str, max_tokens: int = 200) -> str:
        response = self.model(
        prompt,
        max_tokens=512,
        temperature=0.2,
        stop=["<|im_end|>"]
    )
        return response["choices"][0]["text"].strip()