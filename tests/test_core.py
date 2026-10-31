from fastapi.testclient import TestClient
from app.rag import chunk_pages
from app.main import app


def test_chunk_overlap_and_pages():
    ch = chunk_pages([(1, "a" * 250), (2, "b" * 50)], size=100, overlap=20)
    assert [p for p, _ in ch] == [1, 1, 1, 2]
    assert all(len(t) <= 100 for _, t in ch)


def test_chunk_rejects_bad_overlap():
    import pytest
    with pytest.raises(AssertionError):
        chunk_pages([(1, "x")], size=10, overlap=10)


def test_health_and_auth():
    c = TestClient(app)
    assert c.get("/health").json() == {"status": "ok"}
    assert c.get("/documents").status_code == 401
    assert c.get("/documents", headers={"X-API-Key": "dev-key"}).status_code == 200
