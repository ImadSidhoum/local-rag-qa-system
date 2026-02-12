from app.chunking import _split_text_with_overlap


def test_split_text_with_overlap_is_deterministic() -> None:
    text = " ".join(["token"] * 400)

    chunks_one = _split_text_with_overlap(text, chunk_size=120, overlap=20)
    chunks_two = _split_text_with_overlap(text, chunk_size=120, overlap=20)

    assert chunks_one == chunks_two
    assert len(chunks_one) > 1
    assert all(chunk.strip() for chunk in chunks_one)


def test_split_text_with_overlap_validates_params() -> None:
    try:
        _split_text_with_overlap("abc", chunk_size=100, overlap=100)
    except ValueError as exc:
        assert "overlap" in str(exc)
    else:
        raise AssertionError("Expected ValueError")
