# DSA RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers questions grounded in a Data Structures and Algorithms textbook (PDF). Built with LangChain, LangGraph, and Streamlit.

## How it works

1. **Extraction** — the PDF is loaded page-by-page with PyMuPDF; front matter (cover, table of contents, preface) and the back-of-book index are excluded by page range, since they aren't useful retrieval content.
2. **Chunking** — content pages are split into ~1000-character chunks with 150-character overlap, using paragraph/line/sentence boundaries so code blocks and explanations stay together where possible.
3. **Embedding** — each chunk is converted into a vector using a local `sentence-transformers` model (`all-MiniLM-L6-v2`), so there's no API cost or rate limit for this step.
4. **Vector store** — vectors are persisted to disk with Chroma (`./chroma_db`), so the book only needs to be embedded once.
5. **Retrieval + generation** — a LangGraph app rewrites follow-up questions into standalone queries (using conversation history), retrieves the top matching chunks from Chroma, and asks Gemini to answer using only that retrieved context — citing which pages it came from.
6. **UI** — a Streamlit chat interface wraps the LangGraph app, with per-session conversation memory.

## Project structure

```
app.py                 # Streamlit chat UI (the app you actually run)
build_vectorstore.py   # One-time script: loads, chunks, embeds, and persists the book to ./chroma_db
chatbot.py              # Terminal version of the chatbot (same RAG logic as app.py, no UI)
pyproject.toml / uv.lock  # Dependencies, managed with uv
.env.example            # Template for required environment variables
```

## Setup

**1. Install dependencies**

```bash
uv sync
```

**2. Add your source PDF**

Place your own DSA textbook PDF in the project root, named `DataStructures.pdf` (or update the filename inside `build_vectorstore.py` and `app.py` if you'd rather use a different name).

> Note: the PDF itself is not included in this repo, since it's copyrighted material. You'll need to supply your own copy.

**3. Set up your API key**

Copy the example env file and fill in your own key:

```bash
cp .env.example .env
```

Then edit `.env` and add your Gemini API key (get one free at [aistudio.google.com](https://aistudio.google.com)):

```
GOOGLE_API_KEY=your-key-here
```

**4. Build the vector store (one-time)**

```bash
python build_vectorstore.py
```

This reads the PDF, chunks it, embeds every chunk locally, and saves the result to `./chroma_db`. Takes a few minutes depending on your machine; you only need to run this once (or again if you change the source PDF or chunking settings).

**5. Run the chatbot**

```bash
streamlit run app.py
```

This opens the chat UI in your browser (usually `http://localhost:8501`). Ask any DSA question — the bot will answer using only what's actually in the book, and cite the page numbers it drew from.

## Notes

- **Embeddings run locally** (no API key or rate limit needed for that step) — only the chat/generation step calls the Gemini API.
- **Gemini's free tier has daily request limits.** If you hit a `429 RESOURCE_EXHAUSTED` error, check [ai.google.dev/gemini-api/docs/rate-limits](https://ai.google.dev/gemini-api/docs/rate-limits) for current limits, or wait for the daily quota to reset.
- `chroma_db/` is excluded from git — it's regenerated locally by running `build_vectorstore.py`.
