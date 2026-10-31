"""Ingestion + retrieval. Heavy libraries are imported lazily so unit tests stay fast."""
import hashlib
from . import config as C

_models = {}


def chunk_pages(pages, size, overlap):
    """pages: [(page_no, text)] -> [(page_no, chunk)] with character overlap, page numbers preserved."""
    assert 0 <= overlap < size, "overlap must be smaller than chunk size"
    out = []
    for pno, text in pages:
        text = " ".join(text.split())
        start = 0
        while start < len(text):
            piece = text[start:start + size]
            if piece.strip():
                out.append((pno, piece))
            if start + size >= len(text):
                break
            start += size - overlap
    return out


def _embedder(name):
    if name not in _models:
        from sentence_transformers import SentenceTransformer
        _models[name] = SentenceTransformer(name)
    return _models[name]


def _reranker():
    if "rr" not in _models:
        from sentence_transformers import CrossEncoder
        _models["rr"] = CrossEncoder(C.RERANK_MODEL)
    return _models["rr"]


class RAG:
    def __init__(self, embed_model=C.EMBED_MODEL, chunk_size=C.CHUNK_SIZE,
                 overlap=C.CHUNK_OVERLAP, rerank=False, top_k=C.TOP_K):
        import chromadb
        self.embed_model, self.size, self.overlap = embed_model, chunk_size, overlap
        self.rerank, self.top_k = rerank, top_k
        tag = hashlib.md5(f"{embed_model}|{chunk_size}|{overlap}".encode()).hexdigest()[:8]
        client = chromadb.PersistentClient(path=f"{C.DATA_DIR}/chroma")
        self.col = client.get_or_create_collection(f"c_{tag}", metadata={"hnsw:space": "cosine"})

    def ingest(self, doc_id, filename, pdf_path):
        from pypdf import PdfReader
        reader = PdfReader(pdf_path)
        pages = [(i + 1, p.extract_text() or "") for i, p in enumerate(reader.pages)]
        chunks = chunk_pages(pages, self.size, self.overlap)
        if not chunks:
            raise ValueError("No extractable text (scanned PDF? OCR is not supported).")
        texts = [c[1] for c in chunks]
        embs = _embedder(self.embed_model).encode(texts, normalize_embeddings=True).tolist()
        self.col.upsert(
            ids=[f"{doc_id}-{i}" for i in range(len(chunks))],
            embeddings=embs, documents=texts,
            metadatas=[{"doc_id": doc_id, "filename": filename, "page": p} for p, _ in chunks])
        return len(pages), len(chunks)

    def retrieve(self, question, doc_ids=None, rerank=None):
        rerank = self.rerank if rerank is None else rerank
        qe = _embedder(self.embed_model).encode([question], normalize_embeddings=True).tolist()
        where = {"doc_id": {"$in": doc_ids}} if doc_ids else None  # metadata filtering
        n = self.top_k * 4 if rerank else self.top_k
        r = self.col.query(query_embeddings=qe, n_results=n, where=where)
        hits = [{"text": d, "page": m["page"], "filename": m["filename"], "doc_id": m["doc_id"],
                 "score": round(1 - dist, 4)}
                for d, m, dist in zip(r["documents"][0], r["metadatas"][0], r["distances"][0])]
        if rerank and hits:
            scores = _reranker().predict([(question, h["text"]) for h in hits])
            for h, s in zip(hits, scores):
                h["rerank"] = float(s)
            hits.sort(key=lambda h: -h["rerank"])
        return hits[:self.top_k]
