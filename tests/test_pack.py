from skex.data.pack import pack_document


def test_priority_and_cap():
    text = pack_document({"title": "T", "abstract": "A", "other": "X"}, max_chars=100)
    assert text.index("title") < text.index("abstract")
    assert "other" in text
