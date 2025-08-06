import json
import os
import uuid
import chromadb
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- Config ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_PATH = os.path.join(SCRIPT_DIR, "narrative_sources.json")
COLLECTION_NAME = "glen_ellyn_history"  # Use same collection to unify context


# --- Load and Chunk Narratives ---
def load_narratives(json_path):
    with open(json_path, "r") as f:
        records = json.load(f)

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = []

    for record in records:
        doc_id_base = str(uuid.uuid4())  # Unique ID base for each doc
        raw_text = record["content"]
        split_docs = splitter.create_documents([raw_text])

        for i, chunk in enumerate(split_docs):
            chunks.append({
                "content": chunk.page_content,
                "metadata": {
                    "source": record.get("source", "unknown"),
                    "type": record.get("type", "narrative"),
                    "date": record.get("date", "unknown"),
                    "title": record.get("title", "Untitled")
                },
                "id": f"{doc_id_base}_{i}"
            })

    return chunks


# --- Embed and Store in Chroma ---
def embed_and_store(chunks):
    client = chromadb.Client()
    collection = client.get_or_create_collection(COLLECTION_NAME)
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    for chunk in chunks:
        embedding = embedder.encode(chunk["content"]).tolist()
        collection.add(
            documents=[chunk["content"]],
            embeddings=[embedding],
            metadatas=[chunk["metadata"]],
            ids=[chunk["id"]]
        )


# --- Run Ingestion ---
if __name__ == "__main__":
    chunks = load_narratives(JSON_PATH)
    print(f"Loaded and chunked {len(chunks)} narrative segments.")
    embed_and_store(chunks)
    print("✅ Narrative content indexed into ChromaDB.")
