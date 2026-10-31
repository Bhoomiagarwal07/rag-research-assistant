"""Usage: python -m eval.run_eval eval/questions.jsonl path/to/pdf_folder
Each line: {question, expected_answer, doc (pdf filename), pages [gold page numbers; [] if unanswerable]}"""
import json, os, sys, time
from app import config as C, llm
from app.rag import RAG

CONFIGS = {
    "A: chunk 500/50, no rerank": dict(chunk_size=500, overlap=50, rerank=False),
    "B: chunk 800/100, rerank": dict(chunk_size=800, overlap=100, rerank=True),
}


def judge(prompt):
    r = llm.client().chat.completions.create(model=C.LLM_MODEL, temperature=0,
        messages=[{"role": "user", "content": prompt + "\nReply with only 1 (yes) or 0 (no)."}])
    return 1 if r.choices[0].message.content.strip().startswith("1") else 0


def run(qs, pdf_dir, cfg):
    rag = RAG(**cfg)
    for f in {q["doc"] for q in qs}:
        rag.ingest(f, f, os.path.join(pdf_dir, f))
    rec, cor, grd, lat, cost = [], [], [], [], []
    for q in qs:
        t = time.time()
        hits = rag.retrieve(q["question"], [q["doc"]])
        res = llm.answer(q["question"], hits)
        lat.append(time.time() - t)
        cost.append(res["cost_usd"])
        if q["pages"]:
            rec.append(int(any(h["page"] in q["pages"] for h in hits)))
        ctx = "\n".join(h["text"] for h in hits)
        cor.append(judge(f"Question: {q['question']}\nExpected: {q['expected_answer']}\nAnswer: {res['answer']}\nDoes the answer match the expected answer in meaning?"))
        grd.append(judge(f"Context:\n{ctx}\n\nAnswer: {res['answer']}\nIs every claim in the answer supported by the context?"))
    m = lambda x: round(sum(x) / len(x), 3) if x else float("nan")
    return {f"Recall@{rag.top_k} (page-level)": m(rec), "Answer correctness": m(cor), "Groundedness": m(grd),
            "Avg latency (s)": m(lat), "Avg cost/query ($)": round(sum(cost) / len(cost), 6)}


if __name__ == "__main__":
    qs = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
    rows = {name: run(qs, sys.argv[2], cfg) for name, cfg in CONFIGS.items()}
    cols = list(next(iter(rows.values())))
    md = f"Questions: {len(qs)}\n\n| Config | " + " | ".join(cols) + " |\n|" + "---|" * (len(cols) + 1) + "\n"
    for n, r in rows.items():
        md += f"| {n} | " + " | ".join(str(r[c]) for c in cols) + " |\n"
    open("eval/results.md", "w").write(md)
    print(md)
