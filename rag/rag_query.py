import json
import os
import chromadb
from sentence_transformers import SentenceTransformer
import requests

# --- Load and embed metadata ---
def load_metadata_to_docs(full_path: str):
    with open(full_path, 'r') as f:
        raw_data = json.load(f)

    docs = []
    for entry in raw_data:
        content = f"""
Address: {entry['address']}
Year: {entry['year']}
Building: {entry['building_description']}
Use: {entry['building_use']}
Stories: {entry['stories']}
Material: {entry['construction_material']}
Notes: {entry['notes']}
"""
        docs.append({
            "content": content.strip(),
            "metadata": {
                "address": entry["address"],
                "year": entry["year"],
                "map_sheet": entry["map_sheet"]
            }
        })

    return docs

# --- Setup Chroma + Embeddings ---
def init_chroma(docs):
    client = chromadb.Client()
    collection = client.get_or_create_collection("glen_ellyn_history")
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    for doc in docs:
        embedding = embed_model.encode(doc["content"]).tolist()
        doc_id = doc["metadata"]["address"] + "_" + str(doc["metadata"]["year"])
        collection.add(
            documents=[doc["content"]],
            embeddings=[embedding],
            metadatas=[doc["metadata"]],
            ids=[doc_id]
        )

    return collection, embed_model


# --- Query Vector DB ---
def retrieve_context(question, collection, embed_model, top_k=3):
    query_embed = embed_model.encode(question).tolist()
    results = collection.query(query_embeddings=[query_embed], n_results=top_k)
    return results["documents"][0]


# --- LLM via Ollama ---
def call_ollama(prompt, model="llama3"):
    url = "http://localhost:11434/api/generate"
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }

    response = requests.post(url, headers=headers, json=payload)
    return response.json()["response"]


# --- Wrap RAG Prompt ---
def build_prompt(question, context_chunks):
    context_str = "\n\n---\n\n".join(context_chunks)
    return f"""
You are a helpful local historian chatbot for the town of Glen Ellyn, IL.

Answer the following question using the historical context provided below.

Question: {question}

Context:
{context_str}

Be concise and friendly. If there's a specific year mentioned, prioritize information from that time.
"""


# --- Main entrypoint ---
if __name__ == "__main__":
    question = input("Ask a question about Glen Ellyn history:\n> ")

    import os

    script_dir = os.path.dirname(os.path.abspath(__file__))
    metadata_path = os.path.join(script_dir, "metadata.json")
    docs = load_metadata_to_docs(metadata_path) 
    collection, embed_model = init_chroma(docs)
    context = retrieve_context(question, collection, embed_model)
    prompt = build_prompt(question, context)
    response = call_ollama(prompt)

    print("\n🤖 Answer:\n")
    print(response)