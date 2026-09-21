from embedfallback.chunking import FixedSizeChunker, RecursiveCharacterChunker, SemanticChunker, get_chunker


def test_fixed_size_chunker_covers_whole_text():
    text = "a" * 2500
    chunker = FixedSizeChunker(chunk_size=1000, overlap=100)
    chunks = chunker.split(text, "doc1")
    assert chunks[0].start_char == 0
    assert chunks[-1].end_char == len(text)
    # every char should appear in at least one chunk
    reconstructed = set()
    for c in chunks:
        reconstructed.update(range(c.start_char, c.end_char))
    assert reconstructed == set(range(len(text)))


def test_recursive_chunker_respects_paragraph_boundaries_when_possible():
    text = "Para one is short.\n\n" + ("word " * 300) + "\n\nPara three is short."
    chunker = RecursiveCharacterChunker(chunk_size=500, overlap=0)
    chunks = chunker.split(text, "doc2")
    assert len(chunks) > 1
    assert all(len(c.text) <= 550 for c in chunks)  # some slack for split-word edge cases


def test_semantic_chunker_without_similarity_fn_still_produces_valid_chunks():
    text = " ".join([f"Sentence number {i}." for i in range(50)])
    chunker = SemanticChunker(target_chunk_size=100)
    chunks = chunker.split(text, "doc3")
    assert len(chunks) > 1
    assert all(c.text.strip() for c in chunks)


def test_semantic_chunker_breaks_on_low_similarity():
    def fake_similarity(a, b):
        return 0.0 if "TOPIC_SHIFT" in b else 1.0

    text = "Sentence one. Sentence two. TOPIC_SHIFT sentence three. Sentence four."
    chunker = SemanticChunker(target_chunk_size=10000, similarity_fn=fake_similarity, similarity_threshold=0.5)
    chunks = chunker.split(text, "doc4")
    assert len(chunks) >= 2


def test_get_chunker_factory():
    assert isinstance(get_chunker("fixed"), FixedSizeChunker)
    assert isinstance(get_chunker("recursive"), RecursiveCharacterChunker)
    assert isinstance(get_chunker("semantic"), SemanticChunker)
