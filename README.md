# AI Research Assistant (RAG)
Upload PDFs, ask questions, get answers grounded in the documents with page-level citations.

**Stack:** Python, FastAPI, Streamlit, ChromaDB, Sentence-Transformers (embeddings + cross-encoder reranker), any OpenAI-compatible LLM, SQLite, Pytest, Docker.

## Features
- PDF ingestion -> page-aware chunking (configurable size/overlap) -> embeddings -> Chroma
- Multi-document search with metadata filtering, optional cross-encoder reranking
- Refuses to answer when retrieval similarity is below `MIN_SCORE` or the LLM finds no support
- API-key auth, SQLite metadata, per-query latency / token / cost tracking (`/metrics`)
- Evaluation harness: Recall@k, answer correctness, groundedness, latency, cost across two configurations

## Run locally
```
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # add your LLM_API_KEY
uvicorn app.main:app --reload            # terminal 1 (API docs at /docs)
streamlit run streamlit_app.py           # terminal 2
pytest                                   # tests
```
Or: `docker compose up --build` (UI at :8501).

## Evaluate
Create `eval/questions.jsonl` (>=50 questions, format in `eval/questions.example.jsonl`, include a few unanswerable ones), put PDFs in a folder, then:
`python -m eval.run_eval eval/questions.jsonl ./pdfs` -> writes `eval/results.md`.

## Results
(Paste your real table from eval/results.md here.)

## Limitations
No OCR for scanned PDFs; API-key auth (not per-user accounts); LLM-as-judge metrics are approximate.
