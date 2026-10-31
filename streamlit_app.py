import os, requests, streamlit as st

st.set_page_config(page_title="AI Research Assistant", layout="wide")
st.title("AI Research Assistant (RAG)")
api = st.sidebar.text_input("API URL", os.getenv("API_URL", "http://localhost:8000"))
key = st.sidebar.text_input("API key", os.getenv("APP_API_KEY", "dev-key"), type="password")
H = {"X-API-Key": key}

up = st.sidebar.file_uploader("Upload PDF", type="pdf")
if up and st.sidebar.button("Ingest"):
    with st.spinner("Extracting, chunking, embedding..."):
        r = requests.post(f"{api}/upload", headers=H, files={"file": (up.name, up.getvalue())})
    if r.ok: 
        st.sidebar.success(f"Ingested {r.json()['filename']}: {r.json()['pages']} pages, {r.json()['chunks']} chunks")
    else:
        st.sidebar.error(r.text)

docs = requests.get(f"{api}/documents", headers=H).json() if key else []
names = {d["filename"] + f" ({d['id']})": d["id"] for d in docs} if isinstance(docs, list) else {}
chosen = st.sidebar.multiselect("Search in (empty = all)", list(names))
rerank = st.sidebar.checkbox("Rerank results")

q = st.chat_input("Ask a question about your documents")
if q:
    st.chat_message("user").write(q)
    r = requests.post(f"{api}/ask", headers=H,
                      json={"question": q, "doc_ids": [names[n] for n in chosen] or None, "rerank": rerank})
    if not r.ok:
        st.error(r.text)
    else:
        d = r.json()
        with st.chat_message("assistant"):
            st.write(d["answer"])
            st.caption(f"{d['latency_s']}s | {d['tokens_in']}+{d['tokens_out']} tokens | ${d['cost_usd']:.6f}")
            for i, c in enumerate(d["citations"], 1):
                with st.expander(f"[{i}] {c['filename']} - page {c['page']} (sim {c['score']})"):
                    st.write(c["text"])
