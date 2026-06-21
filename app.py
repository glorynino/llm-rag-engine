import gradio as gr
import sys
sys.path.insert(0, "src")
from pipeline import Pipeline

pipeline = Pipeline()

def chat(question, history):
    if not question.strip():
        return "", history
    
    result = pipeline.ask(question)
    sources = ", ".join(result["sources"]) if result["sources"] else "Aucune"
    response = f"{result['answer']}\n\n📎 Sources : {sources}"
    
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": response})
    return "", history

with gr.Blocks(title="Chatbot Entreprise") as app:
    gr.Markdown("## 🤖 Chatbot Entreprise — RAG Local")
    chatbot = gr.Chatbot(height=400)
    msg = gr.Textbox(placeholder="Posez votre question...", label="Question")
    clear = gr.Button("Effacer")
    
    msg.submit(chat, [msg, chatbot], [msg, chatbot])
    clear.click(lambda: ([], ""), outputs=[chatbot, msg])

app.launch()
