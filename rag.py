"""
Simple RAG (Retrieval-Augmented Generation) pipeline for German PDFs.

Pipeline:
1. Download a PDF from a URL
2. Extract and chunk the text
3. Embed chunks with a multilingual sentence-transformer model
4. Index embeddings with FAISS
5. On a question: embed the question, retrieve top-k chunks, ask Claude to answer using them

Usage:
    python rag.py --build                 # downloads PDF + builds the index
    python rag.py --ask "Deine Frage?"     # asks a question against the existing index
"""

import argparse
import os
import pickle
import re
import sys

import faiss
import numpy as np
import requests
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PDF_URL = "https://www.goethe.de/pro/relaunch/prf/de/A1_SD1_Wortliste_02.pdf"
PDF_PATH = "document.pdf"
INDEX_PATH = "index.faiss"
CHUNKS_PATH = "chunks.pkl"

EMBED_MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"
CHUNK_SIZE = 800        # characters per chunk
CHUNK_OVERLAP = 150     # overlap between consecutive chunks
TOP_K = 4               # how many chunks to retrieve per question
NUM_QUIZ_QUESTIONS = 5  # how many questions to generate per --quiz run

# Free, fully local generation via Ollama (https://ollama.com)
# Install Ollama, then run: ollama pull llama3.1
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.1"


# ---------------------------------------------------------------------------
# Step 1: Download PDF
# ---------------------------------------------------------------------------

def download_pdf(url: str, out_path: str) -> None:
    print(f"Downloading PDF from: {url}")
    headers = {"User-Agent": "Mozilla/5.0 (RAG demo script)"}
    resp = requests.get(url, headers=headers, timeout=60)
    resp.raise_for_status()
    with open(out_path, "wb") as f:
        f.write(resp.content)
    print(f"Saved PDF to {out_path} ({len(resp.content) / 1024:.1f} KB)")


# ---------------------------------------------------------------------------
# Step 2: Extract + chunk text
# ---------------------------------------------------------------------------

def extract_text(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(text)
    full_text = "\n".join(pages)
    # Collapse excessive whitespace
    full_text = re.sub(r"\s+", " ", full_text).strip()
    return full_text


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end]
        chunks.append(chunk)
        if end == n:
            break
        start = end - overlap
    return chunks


# ---------------------------------------------------------------------------
# Step 3 + 4: Embed and index
# ---------------------------------------------------------------------------

def build_index(chunks: list[str], model: SentenceTransformer):
    print(f"Embedding {len(chunks)} chunks...")
    embeddings = model.encode(chunks, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine similarity via normalized + inner product
    index.add(embeddings.astype(np.float32))
    return index


def save_index(index, chunks: list[str]) -> None:
    faiss.write_index(index, INDEX_PATH)
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)
    print(f"Saved index to {INDEX_PATH} and chunks to {CHUNKS_PATH}")


def load_index():
    index = faiss.read_index(INDEX_PATH)
    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)
    return index, chunks


# ---------------------------------------------------------------------------
# Step 5: Retrieve + ask Claude
# ---------------------------------------------------------------------------

def retrieve(question: str, index, chunks: list[str], model: SentenceTransformer, top_k: int = TOP_K) -> list[str]:
    q_emb = model.encode([question], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
    scores, idxs = index.search(q_emb, top_k)
    return [chunks[i] for i in idxs[0] if i != -1]


def call_ollama(prompt: str) -> str:
    """Send a prompt to a free, fully local model served by Ollama.

    Requires Ollama installed and running (https://ollama.com) with a model
    pulled, e.g.:  ollama pull llama3.1
    """
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=120,
        )
        resp.raise_for_status()
    except requests.exceptions.ConnectionError:
        sys.exit(
            "Could not reach Ollama at http://localhost:11434.\n"
            "Install it from https://ollama.com, then run:\n"
            f"  ollama pull {OLLAMA_MODEL}\n"
            "  ollama serve   (if it's not already running)"
        )
    return resp.json()["response"].strip()


def ask_ollama(question: str, context_chunks: list[str]) -> str:
    """Tutor mode: answer a learner's question about the vocabulary, in German
    and English, using only the retrieved context."""
    context = "\n\n---\n\n".join(context_chunks)
    system_prompt = (
        "Du bist ein freundlicher Deutschlehrer fuer Anfaenger (Niveau A1). "
        "Beantworte die Frage des Lernenden ausschliesslich anhand des "
        "bereitgestellten Kontexts (einer Vokabelliste). "
        "Antworte zweisprachig: zuerst auf einfachem Deutsch, dann eine kurze "
        "englische Uebersetzung/Erklaerung in Klammern. Gib, wenn passend, ein "
        "Beispielsatz. Wenn die Antwort nicht im Kontext steht, sage das ehrlich."
    )
    prompt = f"{system_prompt}\n\nKontext:\n{context}\n\nFrage: {question}\n\nAntwort:"
    return call_ollama(prompt)


# ---------------------------------------------------------------------------
# Quiz mode: generate questions from random chunks, grade the user's answers
# ---------------------------------------------------------------------------

def generate_quiz_questions(chunk: str, n: int) -> str:
    prompt = (
        "Du bist ein Deutschlehrer fuer Anfaenger (Niveau A1). "
        f"Erstelle genau {n} kurze Vokabel-Quizfragen auf Basis des folgenden "
        "Textausschnitts (z.B. 'Was bedeutet das Wort X?' oder "
        "'Wie sagt man Y auf Deutsch?'). "
        "Nummeriere die Fragen 1. 2. 3. usw. Gib NUR die Fragen aus, keine "
        "Antworten.\n\n"
        f"Textausschnitt:\n{chunk}"
    )
    return call_ollama(prompt)


def grade_answer(question: str, user_answer: str, chunk: str) -> str:
    prompt = (
        "Du bist ein freundlicher Deutschlehrer fuer Anfaenger (Niveau A1). "
        "Bewerte die Antwort des Lernenden anhand des Kontexts. "
        "Sag kurz, ob sie richtig oder falsch ist, gib die korrekte Antwort "
        "an und erklaere kurz auf Deutsch und Englisch.\n\n"
        f"Kontext:\n{chunk}\n\n"
        f"Frage: {question}\n"
        f"Antwort des Lernenden: {user_answer}\n\n"
        "Bewertung:"
    )
    return call_ollama(prompt)


def quiz_pipeline(n: int = NUM_QUIZ_QUESTIONS):
    if not (os.path.exists(INDEX_PATH) and os.path.exists(CHUNKS_PATH)):
        sys.exit("No index found. Run with --build first.")
    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)

    import random
    chunk = random.choice(chunks)

    print("Generiere Quizfragen...\n")
    questions_text = generate_quiz_questions(chunk, n)
    questions = [q.strip() for q in re.split(r"\n+", questions_text) if q.strip()]

    print("=== Quiz ===\n")
    for q in questions:
        print(q)
        user_answer = input("Deine Antwort: ")
        feedback = grade_answer(q, user_answer, chunk)
        print(f"\n{feedback}\n{'-' * 40}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_pipeline(pdf_url: str = PDF_URL):
    download_pdf(pdf_url, PDF_PATH)
    text = extract_text(PDF_PATH)
    print(f"Extracted {len(text)} characters of text.")
    chunks = chunk_text(text)
    print(f"Created {len(chunks)} chunks.")
    model = SentenceTransformer(EMBED_MODEL_NAME)
    index = build_index(chunks, model)
    save_index(index, chunks)
    print("Build complete.")


def ask_pipeline(question: str):
    if not (os.path.exists(INDEX_PATH) and os.path.exists(CHUNKS_PATH)):
        sys.exit("No index found. Run with --build first.")
    model = SentenceTransformer(EMBED_MODEL_NAME)
    index, chunks = load_index()
    relevant = retrieve(question, index, chunks, model)
    print("\n--- Retrieved context chunks ---")
    for i, c in enumerate(relevant, 1):
        print(f"[{i}] {c[:200]}...\n")
    answer = ask_ollama(question, relevant)
    print("--- Answer ---")
    print(answer)


def main():
    parser = argparse.ArgumentParser(description="Simple German-language RAG pipeline")
    parser.add_argument("--build", action="store_true", help="Download PDF and build the index")
    parser.add_argument("--pdf-url", type=str, default=PDF_URL, help="URL of the PDF to download")
    parser.add_argument("--ask", type=str, help="Ask a question against the built index")
    parser.add_argument("--quiz", action="store_true", help="Generate a vocabulary quiz from the PDF")
    parser.add_argument("--num-questions", type=int, default=NUM_QUIZ_QUESTIONS, help="Number of quiz questions")
    args = parser.parse_args()

    if args.build:
        build_pipeline(args.pdf_url)
    elif args.ask:
        ask_pipeline(args.ask)
    elif args.quiz:
        quiz_pipeline(args.num_questions)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()