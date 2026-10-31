import time
from . import config as C

NOT_FOUND = "I could not find this in the uploaded documents."
SYSTEM = ("You are a research assistant. Answer ONLY from the numbered context passages. "
          "Cite sources inline like [1], [2] and mention page numbers. "
          f"If the context does not contain the answer, reply exactly: {NOT_FOUND}")
_client = None


def client():
    global _client
    if _client is None:
        from openai import OpenAI
        _client = OpenAI(base_url=C.LLM_BASE_URL, api_key=C.LLM_API_KEY)
    return _client


def answer(question, hits):
    ctx = "\n\n".join(f"[{i+1}] ({h['filename']}, p.{h['page']}) {h['text']}" for i, h in enumerate(hits))
    t = time.time()
    r = client().chat.completions.create(
        model=C.LLM_MODEL, temperature=0,
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": f"Context:\n{ctx}\n\nQuestion: {question}"}])
    u = r.usage
    return {"answer": r.choices[0].message.content.strip(),
            "tokens_in": u.prompt_tokens, "tokens_out": u.completion_tokens,
            "cost_usd": (u.prompt_tokens * C.PRICE_IN + u.completion_tokens * C.PRICE_OUT) / 1e6,
            "llm_latency_s": round(time.time() - t, 3)}
