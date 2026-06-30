# Deutsch Lernen mit RAG (German Vocabulary Tutor) — 100% Free & Local

A local study tool: downloads a German A1 vocabulary PDF, then helps you
learn it two ways — ask it questions like a tutor, or have it quiz you.
Everything runs on your machine for free — no API keys, no cloud calls.

Default source: the official **Goethe-Institut A1 ("Start Deutsch 1")
vocabulary list** — ~650 essential beginner words with example sentences,
grouped by topic (greetings, numbers, family, food, etc).

## Setup

1. Install Python dependencies:

```bash
pip install -r requirements.txt
```

2. Install [Ollama](https://ollama.com) (free, local LLM runner) and pull a model:

```bash
ollama pull llama3.1
```

   `llama3.1` handles German well. Smaller/faster alternatives: `phi3`,
   `mistral`.

3. Make sure Ollama is running (it usually starts automatically after
   install; otherwise run `ollama serve` in a terminal).

## Usage

1. Build the index (downloads the vocabulary PDF, chunks it, embeds it):

```bash
python rag.py --build
```

   To use a different vocabulary/phrasebook PDF instead:

```bash
python rag.py --build --pdf-url "https://example.com/dein-wortschatz.pdf"
```

2. **Tutor mode** — ask it about words, phrases, or grammar from the list:

```bash
python rag.py --ask "Was bedeutet 'das Brötchen'?"
python rag.py --ask "Wie sagt man 'I am hungry' auf Deutsch?"
python rag.py --ask "Gib mir 3 Beispielsätze mit 'kommen'."
```

   It answers bilingually (simple German + English explanation), grounded
   only in the vocabulary list content.

3. **Quiz mode** — get tested on a random batch of words from the list:

```bash
python rag.py --quiz
python rag.py --quiz --num-questions 10
```

   It generates questions, asks you in the terminal, and gives feedback on
   each answer (correct/incorrect + explanation in German and English).

## How it works

1. **Download** — fetches the Goethe-Institut A1 vocabulary PDF (or your own).
2. **Extract & chunk** — `pypdf` pulls the text, split into ~800-character
   overlapping chunks so each piece covers a coherent group of words.
3. **Embed & index** — `sentence-transformers`
   (`paraphrase-multilingual-mpnet-base-v2`) embeds chunks, FAISS indexes
   them for fast retrieval. Fully local and free.
4. **Tutor mode** — your question is embedded, the most relevant chunks are
   retrieved, and a local LLM (via Ollama) answers using only that content.
5. **Quiz mode** — a random chunk is picked, the local LLM generates
   questions from it, then grades your typed answers against that same
   chunk.

## Customizing

- `CHUNK_SIZE` / `CHUNK_OVERLAP` — tune chunk granularity.
- `TOP_K` — how many chunks get retrieved per tutor question.
- `NUM_QUIZ_QUESTIONS` — default quiz length.
- `OLLAMA_MODEL` — swap for any model you've pulled (`mistral`, `phi3`,
  `gemma2`, etc).

## Other free vocabulary PDFs to try

- Goethe-Institut A2: search "Goethe-Zertifikat A2 Wortliste pdf" on goethe.de
- Deutsch Online glossaries (by chapter): lernen.goethe.de
- Any vocabulary list with German + English/translations and example
  sentences will work well with this pipeline.

## Notes

- First run downloads the embedding model (~1GB) and the Ollama model
  (~4-8GB) — only happens once.
- Quiz/tutor quality depends on the local model you choose; `llama3.1` or
  larger gives noticeably better German than `phi3`/small models.
- This assumes the PDF has a real text layer (not a scanned image).
