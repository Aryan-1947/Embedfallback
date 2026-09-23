# Embedfallback

A rate-limit-resilient embedding ingestion pipeline that lets you chain
multiple embedding API providers as fallbacks -- **without silently
corrupting retrieval quality** -- by mathematically aligning each
provider's vector space to a shared canonical space.

Add your API keys, point it at a document, and it handles rate limits for
you: when one provider hits its limit mid-document, the next provider picks
up automatically, and the vectors stay consistent across the switch. No
manual retries, no waiting for a cooldown, no silently broken retrieval
quality from mixing incompatible embedding spaces.

## The problem

Free-tier embedding APIs (Google AI Studio, Cohere, Voyage, Hugging Face)
impose rate limits -- often just 100 requests/minute or a low daily cap.
Fine-grained chunking strategies, semantic chunking especially, can produce
hundreds of chunks per document, reliably blowing through those limits
mid-ingestion.

Simple retry/routing tools solve *which provider to call next*. They don't
solve what happens after: if half your document's chunks get embedded by
Google and the other half by Cohere, **those vectors live in different,
mathematically incompatible spaces**. Mixing them in the same retrieval
index silently degrades search quality, with no error and no warning.

`embedfallback` solves the alignment problem, not just the routing problem.

## How it works

1. Each document gets embedded by one **canonical provider** -- whichever
   succeeds first.
2. If that provider hits a rate limit mid-document, the router classifies
   *why* (short per-minute cooldown vs. a longer quota exhaustion vs.
   unknown) and picks the next available provider.
3. Vectors from any non-canonical provider are passed through the
   **Alignment Engine**, which learns a scaled-rotation (Procrustes)
   transform between that provider's space and the canonical provider's
   space, using a shared anchor corpus embedded by both.
4. The result: every vector in a document's output lives in one consistent
   space, with a per-chunk `alignment_confidence` score so you can see
   when a chunk went through a lower-confidence fallback path.

```
Document -> Chunker -> Orchestrator -> Router -> Provider Adapter
                                          |            |
                                          |     (rate limited?)
                                          |            v
                                          `----> Alignment Engine -> aligned vector
```


## Supported providers

| Provider | Model (default) | Dimensions | Get a key |
|---|---|---|---|
| Google | `gemini-embedding-001` | 3072 | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) |
| Cohere | `embed-english-v3.0` | 1024 | [dashboard.cohere.com/api-keys](https://dashboard.cohere.com/api-keys) |
| Voyage AI | `voyage-3.5` | 1024 | [dashboard.voyageai.com](https://dashboard.voyageai.com) |
| Hugging Face | `sentence-transformers/all-MiniLM-L6-v2` (Inference API) | 384 | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) (Read scope) |

All four are pure API calls -- no local model downloads, no GPU required.
Use any subset of them; more providers configured means more resilience
against any single one running out of quota.


> **Want a provider not listed here?** You'll need to add support for it
> yourself by writing a new adapter file (see `embedfallback/providers/`,
> e.g. `google.py` or `voyage.py`, as a template -- each one just wraps that
> provider's API call and classifies its rate-limit errors into the shared
> `RateLimitExceeded` format). Once a new provider's adapter exists and is
> added to your `providers=[...]` list, alignment between it and any other
> provider builds automatically the first time you use that combination --
> no manual anchor setup required, just a one-time wait while it embeds the
> anchor corpus for that new provider. You don't need deep familiarity with
> this codebase to write one -- any AI coding assistant can generate the
> adapter for you by following the pattern of the existing provider files.

## Getting started

1. Get API keys for whichever providers you want to use (you don't need
   all four -- even two gives you real fallback protection): Google
   ([aistudio.google.com/apikey](https://aistudio.google.com/apikey)),
   Cohere ([dashboard.cohere.com/api-keys](https://dashboard.cohere.com/api-keys)),
   Voyage AI ([dashboard.voyageai.com](https://dashboard.voyageai.com)),
   Hugging Face ([huggingface.co/settings/tokens](https://huggingface.co/settings/tokens), Read scope is enough).

2. Install the library:

```bash
pip install "embedfallback[all] @ git+https://github.com/Aryan-1947/embedfallback.git"
```

3. Set your keys as environment variables (or use a `.env` file with
   `python-dotenv` -- either works):

```bash
export GOOGLE_API_KEY=your-key-here
export COHERE_API_KEY=your-key-here
export VOYAGE_API_KEY=your-key-here
export HF_API_KEY=your-key-here
```

4. Write a few lines of Python (see Quickstart below) and run it against a
   real document. That's it -- no separate setup step, no anchor-corpus
   wait, since the default provider pairing ships pre-built with the
   package.

## Quickstart

```python
from embedfallback import IngestionPipeline, providers

pipeline = IngestionPipeline(
    providers=[
        providers.Google(api_key="..."),
        providers.Cohere(api_key="..."),
        providers.Voyage(api_key="..."),
        providers.HuggingFace(api_key="..."),
    ],
    chunk_strategy="semantic",  # or "fixed", "recursive"
)

result = pipeline.ingest("my_document.pdf")

for chunk in result.chunks:
    my_own_vector_db.add(chunk.vector, chunk.text, chunk.metadata)

print(f"Canonical provider: {result.canonical_provider}")
print(f"Provider switches during ingestion: {result.provider_switches}")
```

Storage is explicitly **not** this library's job -- it hands you back
`(vector, text, metadata)` per chunk and you decide whether that goes into
Chroma, Pinecone, pgvector, FAISS, or a plain file.

Documents can be `.pdf`, `.html`, `.md`, `.txt`, or raw text passed
directly as a string.

## Command-line usage (no Python required)

If you don't want to write any code, use `ingest.py` directly from the
terminal. It automatically picks up whichever provider keys you have set
as environment variables (or in a `.env` file):

```bash
python ingest.py --doc my_document.pdf --output vectors.json
```

Optional flags:

```bash
python ingest.py --doc my_document.pdf --output vectors.json --strategy semantic --providers google,cohere
```

- `--strategy`: `fixed`, `recursive` (default), or `semantic`
- `--providers`: comma-separated list to restrict which providers get used
  (default: every provider whose API key is set)

The output JSON contains each chunk's text, vector, which provider embedded
it, and its `alignment_confidence`, along with the run's canonical provider
and how many times it had to fall back to another provider.

## No setup wait on first use

The alignment math needs a shared "anchor corpus" -- a fixed set of
sentences embedded by every provider once, so the library can learn how to
convert between their vector spaces. Building that from scratch the first
time can take a while and eats into your API quota.

This library ships with that anchor corpus **already pre-computed** for the
four default models above (900 real English sentences, each embedded by
all four providers, cached as part of the package). If you're using the
default models, your very first `pipeline.ingest()` call gets instant,
working cross-provider alignment -- no setup wait, no extra API calls spent
on anchor embedding.

If you use a different model for any provider (e.g. a different Google
embedding model), or supply `anchor_corpus_extra` for a specialized domain,
the library falls back to building that specific anchor cache fresh on
first use, then caches it locally (`~/.embedfallback/cache/`) for every
call after that.

## Resumability

Progress is checkpointed after every embedded chunk. If ingestion crashes
or is interrupted, calling `pipeline.ingest()` again with the same document
resumes rather than re-embedding completed chunks -- but only after
validating that the checkpoint's canonical provider and model still match
your current configuration:

- Canonical provider no longer configured at all -> raises `ResumeValidationError`.
- Canonical provider present but its model string changed -> proceeds, but
  logs a warning and re-derives alignment for the remaining chunks.

## Concurrent ingestion

Chunks are embedded in parallel batches (5 at a time by default) rather
than strictly one-at-a-time, which meaningfully speeds up ingestion for
large documents -- a 78-chunk real-world PDF went from 84s to 28s in
testing, roughly a 3x improvement. Adjust batch size via `concurrency=`:

```python
pipeline = IngestionPipeline(providers=[...], concurrency=10)
```

The batch size is currently fixed rather than automatically sized to each
provider's specific rate limit -- a higher-limit provider could safely
handle a larger batch than a stricter one. Tune `concurrency` down if you
hit rate limits more often than expected, or up if a provider has generous
limits and you want faster ingestion.  

## Evaluation harness

Run real, measured comparisons between a single-provider baseline and a
forced multi-provider + alignment run:

```bash
export GOOGLE_API_KEY=...
export COHERE_API_KEY=...
export HF_API_KEY=...
export VOYAGE_API_KEY=...
python evaluate.py --doc sample.pdf --strategy semantic --providers google,cohere,huggingface,voyage
```

Reports: top-5 retrieval overlap, MRR retention, PCA explained-variance and
corpus-size-ratio diagnostics, scale factor, and mean alignment confidence.

## Real measured results

From testing on this project's own sample documents, using the shipped
900-sentence anchor corpus:

| Provider pair | Alignment confidence | Retrieval retention |
|---|---|---|
| Voyage -> Google | 0.83 | 100% |
| Cohere -> Google | 0.68 | 100% |
| HuggingFace -> Google | 0.53 | 100% |

Alignment confidence varies by how different two providers' vector spaces
are (bigger dimension gaps generally align less precisely), but retrieval
quality held up in testing even for the lower-confidence pairs. Your own
results will vary by document domain and provider pair -- use the
evaluation harness on your own data if retrieval quality is critical.

## Honest known limitations

- **Alignment is approximate, not exact.** Expect measurable (if usually
  small) retrieval quality degradation for fallback-embedded chunks.
- **Scaled Procrustes assumes a roughly linear relationship** between
  embedding spaces. This may not hold for structurally very different
  models with large dimension gaps.
- **The shipped 900-sentence anchor corpus is a practical middle ground,
  not maximal.** A larger corpus would generally improve alignment
  precision, but was traded off against real free-tier daily quota limits
  (Google's free tier caps at ~1000 requests/day). If you have paid API
  access and want higher alignment precision, you can regenerate a larger
  anchor corpus and let the library build a fresh cache for it.
- **Anchor corpus domain-match matters.** The shipped corpus is generic
  English sentences. For specialized domains (legal, medical, code), supply
  `anchor_corpus_extra` with representative text from your domain.
- **Rate-limit classification depends on providers returning informative
  error payloads.** Providers that don't will fall into a conservative
  cooldown that may still be wrong in either direction.

## Project layout

```
embedfallback/
  providers/            # base interface + Google/Cohere/HuggingFace/Voyage adapters
  router.py             # provider selection + kind-aware cooldown tracking
  chunking.py            # fixed / recursive / semantic chunking strategies
  alignment.py            # anchor corpus, anchor cache (incl. shipped pre-built cache), Procrustes, diagnostics
  orchestrator.py          # ties it all together, resumable checkpointing
  pipeline.py               # the IngestionPipeline user-facing API
  assets/
    anchor_corpus_base.json          # the 900 anchor sentences
    prebuilt_anchor_cache/            # pre-computed anchor embeddings, shipped with the package
evaluate.py                            # evaluation harness CLI
tests/                                  # unit + integration tests (no live API calls required)
samples/                                 # sample docs used for testing
```


## License

MIT
