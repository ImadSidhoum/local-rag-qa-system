from langchain_core.documents import Document

from app.chunking import split_documents


def test_split_documents_is_deterministic() -> None:
    text = " ".join(["token"] * 400)
    docs = [Document(page_content=text, metadata={"source": "test.pdf", "page": 1})]

    chunks_one = split_documents(docs, chunk_size=120, overlap=20)
    chunks_two = split_documents(docs, chunk_size=120, overlap=20)

    chunk_ids_one = [chunk.metadata["chunk_id"] for chunk in chunks_one]
    chunk_ids_two = [chunk.metadata["chunk_id"] for chunk in chunks_two]
    assert chunk_ids_one == chunk_ids_two
    assert len(chunks_one) > 1
    assert all(chunk.page_content.strip() for chunk in chunks_one)


def test_split_documents_validates_params() -> None:
    try:
        split_documents([], chunk_size=100, overlap=100)
    except ValueError as exc:
        assert "overlap" in str(exc)
    else:
        raise AssertionError("Expected ValueError")
