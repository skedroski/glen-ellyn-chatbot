import os
import json
import re

RAW_ROOT = "narrative_raw"
OUTPUT_PATH = "rag/narrative_sources.json"

def clean_title(filename):
    base = os.path.splitext(filename)[0]
    return base.replace("_", " ").replace("-", " ").title()

def infer_type_from_path(path):
    if "bio" in path.lower():
        return "biography"
    elif "story" in path.lower():
        return "story"
    else:
        return "narrative"

def read_file(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception as e:
        print(f"⚠️ Skipped {path}: {e}")
        return None

def scan_files():
    records = []
    for root, _, files in os.walk(RAW_ROOT):
        for file in files:
            if not file.endswith((".txt", ".md")):
                continue

            full_path = os.path.join(root, file)
            rel_path = os.path.relpath(full_path, RAW_ROOT)
            content = read_file(full_path)
            if not content:
                continue

            title = clean_title(file)
            doc_type = infer_type_from_path(rel_path)
            record = {
                "title": title,
                "date": "unknown",  # No reliable dates in bios/stories
                "type": doc_type,
                "source": rel_path,
                "content": content
            }
            records.append(record)

    return records

def save_json(records):
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    print(f"✅ Saved {len(records)} entries to {OUTPUT_PATH}")

if __name__ == "__main__":
    records = scan_files()
    save_json(records)