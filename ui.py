import gradio as gr
from rag.rag_query import (
    load_metadata_to_docs,
    init_chroma,
    retrieve_context,
    build_prompt,
    call_ollama
)
import os

# Initialize only once
script_dir = os.path.dirname(os.path.abspath(__file__))
metadata_path = os.path.join(script_dir, "rag", "metadata.json")
docs = load_metadata_to_docs(metadata_path)
collection, embed_model = init_chroma(docs)

def answer_question(question):
    if not question.strip():
        return "Please enter a question."

    context = retrieve_context(question, collection, embed_model)
    prompt = build_prompt(question, context)
    response = call_ollama(prompt)
    return response

iface = gr.Interface(
    fn=answer_question,
    inputs=gr.Textbox(lines=2, placeholder="e.g. What was at 4 S Glenwood Ave in 1929?"),
    outputs="text",
    title="🏡 Glen Ellyn History Bot",
    description="Ask about historical places and events in Glen Ellyn, IL. Powered by Sanborn maps and local archives."
)

if __name__ == "__main__":
    iface.launch()