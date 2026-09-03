"""
AI Learning Assistant
Final project for Modern Data Engineering for AI Systems

Pipeline:
Knowledge Base -> Data Quality -> Chunking -> Embeddings ->
ChromaDB -> Semantic Retrieval -> RAG-style Answer
"""

from pathlib import Path
import re
import chromadb
from sentence_transformers import SentenceTransformer

DATA_FILE = Path(__file__).parent / "course_knowledge.txt"
DB_DIR = str(Path(__file__).parent / "chroma_db")
COLLECTION_NAME = "course_knowledge"

def load_data():
    text = DATA_FILE.read_text(encoding="utf-8")
    return text

def data_quality_check(text):
    """Simple automated quality checks inspired by course Day 4."""
    report = {
        "not_empty": bool(text.strip()),
        "minimum_length": len(text.strip()) > 300,
        "has_sections": "SECTION:" in text,
        "duplicate_lines": 0,
    }

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    report["duplicate_lines"] = len(lines) - len(set(lines))
    report["passed"] = (
        report["not_empty"]
        and report["minimum_length"]
        and report["has_sections"]
        and report["duplicate_lines"] == 0
    )
    return report

def chunk_text(text, max_chars=650):
    """Split by sections, then make compact chunks."""
    sections = re.split(r"\n(?=SECTION:)", text)
    chunks = []

    for section in sections:
        section = section.strip()
        if not section:
            continue

        sentences = re.split(r"(?<=[.!?])\s+", section)
        current = ""

        for sentence in sentences:
            if len(current) + len(sentence) + 1 <= max_chars:
                current += (" " if current else "") + sentence
            else:
                if current:
                    chunks.append(current.strip())
                current = sentence

        if current:
            chunks.append(current.strip())

    return chunks

def build_vector_db(chunks):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks).tolist()

    client = chromadb.PersistentClient(path=DB_DIR)

    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "Modern Data Engineering course knowledge"}
    )

    collection.add(
        ids=[f"chunk_{i}" for i in range(len(chunks))],
        documents=chunks,
        embeddings=embeddings,
        metadatas=[{"source": "course_knowledge.txt", "chunk": i} for i in range(len(chunks))]
    )

    return model, collection

def retrieve(query, model, collection, top_k=3):
    query_embedding = model.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k
    )

    docs = results["documents"][0]
    distances = results["distances"][0]
    return list(zip(docs, distances))

def rag_answer(query, retrieved):
    """
    Lightweight RAG answer generator.
    It grounds the response only in retrieved context, so the demo
    works without requiring a paid external LLM API.
    """
    context = "\n".join([doc for doc, _ in retrieved])

    sentences = re.split(r"(?<=[.!?])\s+", context)
    query_terms = set(re.findall(r"\w+", query.lower()))

    ranked = []
    for sentence in sentences:
        terms = set(re.findall(r"\w+", sentence.lower()))
        score = len(query_terms & terms)
        ranked.append((score, sentence.strip()))

    ranked.sort(key=lambda x: x[0], reverse=True)
    selected = [s for score, s in ranked if s][:3]

    if not selected:
        return "I could not find enough grounded information in the knowledge base."

    return " ".join(selected)

def main():
    print("=" * 65)
    print("AI Learning Assistant — Smart Course Knowledge Platform")
    print("=" * 65)

    text = load_data()

    print("\n[1] DATA QUALITY")
    quality = data_quality_check(text)
    for key, value in quality.items():
        print(f"{key}: {value}")

    if not quality["passed"]:
        raise ValueError("Data quality checks failed.")

    print("\n[2] CHUNKING")
    chunks = chunk_text(text)
    print(f"Created {len(chunks)} chunks.")

    print("\n[3] EMBEDDINGS + VECTOR DATABASE")
    model, collection = build_vector_db(chunks)
    print(f"Indexed {collection.count()} chunks in ChromaDB.")

    print("\n[4] RAG READY")
    print("Ask questions about the course. Type 'exit' to stop.")

    while True:
        query = input("\nQuestion: ").strip()
        if query.lower() in {"exit", "quit"}:
            break

        retrieved = retrieve(query, model, collection, top_k=3)
        answer = rag_answer(query, retrieved)

        print("\nAnswer:")
        print(answer)

        print("\nRetrieved evidence:")
        for i, (doc, distance) in enumerate(retrieved, 1):
            print(f"\n{i}. distance={distance:.4f}")
            print(doc[:350] + ("..." if len(doc) > 350 else ""))

if __name__ == "__main__":
    main()
