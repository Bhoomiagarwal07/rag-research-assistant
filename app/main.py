import os, sqlite3, time, uuid
from fastapi import FastAPI, UploadFile, File, Depends, Header, HTTPException
from pydantic import BaseModel
from . import config as C, llm

os.makedirs(f"{C.DATA_DIR}/pdfs", exist_ok=True)
DB = f"{C.DATA_DIR}/meta.db"


def db():
    c = sqlite3.connect(DB)
    c.row_factory = sqlite3.Row
    return c


with db() as c:
    c.executescript("""
    CREATE TABLE IF NOT EXISTS documents(id TEXT PRIMARY KEY, filename TEXT, pages INT, chunks INT, uploaded_at REAL);
    CREATE TABLE IF NOT EXISTS queries(id INTEGER PRIMARY KEY AUTOINCREMENT, question TEXT, latency_s REAL,
      tokens_in INT, tokens_out INT, cost_usd REAL, answered INT, ts REAL);""")

app = FastAPI(title="RAG Research Assistant")
_rag = None


def rag():
    global _rag
    if _rag is None:
        from .rag import RAG
        _rag = RAG()
    return _rag


def auth(x_api_key: str = Header(None)):
    if x_api_key != C.API_KEY:
        raise HTTPException(401, "Invalid or missing X-API-Key")


class Ask(BaseModel):
    question: str
    doc_ids: list[str] | None = None
    rerank: bool = False


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/upload", dependencies=[Depends(auth)])
async def upload(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported")
    doc_id = uuid.uuid4().hex[:12]
    path = f"{C.DATA_DIR}/pdfs/{doc_id}.pdf"
    with open(path, "wb") as f:
        f.write(await file.read())
    try:
        pages, chunks = rag().ingest(doc_id, file.filename, path)
    except ValueError as e:
        raise HTTPException(422, str(e))
    with db() as c:
        c.execute("INSERT INTO documents VALUES(?,?,?,?,?)", (doc_id, file.filename, pages, chunks, time.time()))
    return {"doc_id": doc_id, "filename": file.filename, "pages": pages, "chunks": chunks}


@app.get("/documents", dependencies=[Depends(auth)])
def documents():
    with db() as c:
        return [dict(r) for r in c.execute("SELECT * FROM documents ORDER BY uploaded_at DESC")]


@app.post("/ask", dependencies=[Depends(auth)])
def ask(req: Ask):
    t = time.time()
    hits = rag().retrieve(req.question, req.doc_ids, req.rerank)
    if not hits or hits[0]["score"] < C.MIN_SCORE:
        res = {"answer": llm.NOT_FOUND, "tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0}
    else:
        res = llm.answer(req.question, hits)
    latency = round(time.time() - t, 3)
    answered = int(llm.NOT_FOUND not in res["answer"])
    with db() as c:
        c.execute("INSERT INTO queries(question,latency_s,tokens_in,tokens_out,cost_usd,answered,ts) VALUES(?,?,?,?,?,?,?)",
                  (req.question, latency, res["tokens_in"], res["tokens_out"], res["cost_usd"], answered, time.time()))
    return {**res, "latency_s": latency, "answered": bool(answered),
            "citations": [{k: h[k] for k in ("filename", "page", "score", "text")} for h in hits]}


@app.get("/metrics", dependencies=[Depends(auth)])
def metrics():
    with db() as c:
        r = c.execute("SELECT COUNT(*) n, AVG(latency_s) avg_latency_s, SUM(tokens_in) tokens_in, "
                      "SUM(tokens_out) tokens_out, SUM(cost_usd) total_cost_usd, "
                      "SUM(1-answered) unanswered FROM queries").fetchone()
    return dict(r)
