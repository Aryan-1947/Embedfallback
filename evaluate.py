#!/usr/bin/env python3
"""Evaluation harness: proves, with measured numbers, whether alignment
preserves retrieval quality.

Usage:
    python evaluate.py --doc sample.pdf --strategy semantic --providers google,openai,local
    python evaluate.py --doc legal_contract.pdf --strategy semantic --providers google,openai \
        --domain-mismatch-check

Metrics: top-k retrieval overlap between a single-provider baseline and a
forced multi-provider+alignment run, Mean Reciprocal Rank (MRR) for both,
PCA explained-variance / corpus-size-ratio diagnostics, and per-chunk
alignment_confidence.

Requires provider API keys as environment variables:
    GOOGLE_API_KEY, OPENAI_API_KEY, COHERE_API_KEY
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from embedfallback import providers as provider_module
from embedfallback.alignment import AlignmentEngine, get_anchor_corpus
from embedfallback.chunking import get_chunker
from embedfallback.pipeline import _load_document_text

try:
    from rich.console import Console
    from rich.table import Table
    _HAS_RICH = True
except ImportError:
    _HAS_RICH = False


def build_provider(name: str) -> "provider_module.EmbeddingProvider":
    name = name.lower()
    if name == "google":
        key = os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise SystemExit("GOOGLE_API_KEY environment variable is required for --providers google")
        return provider_module.Google(api_key=key)
    if name == "cohere":
        key = os.environ.get("COHERE_API_KEY")
        if not key:
            raise SystemExit("COHERE_API_KEY environment variable is required for --providers cohere")
        return provider_module.Cohere(api_key=key)
    if name == "huggingface":
        key = os.environ.get("HF_API_KEY")
        if not key:
            raise SystemExit("HF_API_KEY environment variable is required for --providers huggingface")
        return provider_module.HuggingFace(api_key=key)
    if name == "voyage":
        key = os.environ.get("VOYAGE_API_KEY")
        if not key:
            raise SystemExit("VOYAGE_API_KEY environment variable is required for --providers voyage")
        return provider_module.Voyage(api_key=key)
    raise SystemExit(f"Unknown provider: {name}")


def cosine_sim_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-12, None)
    b_norm = b / np.clip(np.linalg.norm(b, axis=1, keepdims=True), 1e-12, None)
    return a_norm @ b_norm.T


def top_k_indices(sims: np.ndarray, k: int) -> np.ndarray:
    return np.argsort(-sims, axis=1)[:, :k]


def mean_reciprocal_rank(sims: np.ndarray, relevant_idx: np.ndarray) -> float:
    """Treats each row's own index (in a self-retrieval sense against a
    query set) as ground truth relevant doc for MRR purposes."""
    ranks = np.argsort(-sims, axis=1)
    rr = []
    for i, rel in enumerate(relevant_idx):
        pos = np.where(ranks[i] == rel)[0]
        rr.append(1.0 / (pos[0] + 1) if len(pos) else 0.0)
    return float(np.mean(rr))


def run_evaluation(doc_path: str, strategy: str, provider_names: list[str], domain_mismatch_check: bool):
    console = Console() if _HAS_RICH else None

    def log(msg: str):
        if console:
            console.print(msg)
        else:
            print(msg)

    doc_id, text = _load_document_text(doc_path)
    chunker = get_chunker(strategy)
    chunks = chunker.split(text, doc_id)
    texts = [c.text for c in chunks]
    log(f"Loaded {len(texts)} chunks from {doc_path!r} using '{strategy}' strategy.")

    provs = {name: build_provider(name) for name in provider_names}
    baseline_provider = provs[provider_names[0]]

    log("Running baseline (single-provider, no fallback)...")
    t0 = time.time()
    baseline_vectors = []
    batch = baseline_provider.max_batch_size()
    for i in range(0, len(texts), batch):
        baseline_vectors.extend(baseline_provider.embed(texts[i:i + batch]).vectors)
    baseline_arr = np.array(baseline_vectors)
    baseline_time = time.time() - t0

    corpus_texts, corpus_version = get_anchor_corpus(None)
    aligner = AlignmentEngine()

    log("Running multi-provider (forced fallback + alignment)...")
    t0 = time.time()
    aligned_vectors = []
    confidences = []
    provider_switches = 0
    diagnostics_by_pair = {}

    other_providers = [provs[n] for n in provider_names[1:]] or [baseline_provider]
    for i, chunk_text in enumerate(texts):
        # Force alternation across providers to simulate fallback churn.
        forced_provider = other_providers[i % len(other_providers)]
        result = None
        attempts = 0
        while result is None:
            attempts += 1
            try:
                result = forced_provider.embed([chunk_text])
            except Exception as e:
                from embedfallback.providers.base import RateLimitExceeded
                if not isinstance(e, RateLimitExceeded) or attempts > 5:
                    raise
                wait = e.retry_after_seconds if e.retry_after_seconds else 60.0
                print(f"  Rate-limited on {forced_provider.name} (chunk {i}, attempt {attempts}/5). Waiting {wait:.0f}s...")
                time.sleep(wait + 1.0)
        vec = result.vectors[0]
        if forced_provider.name != baseline_provider.name:
            provider_switches += 1
            aligned, confidence = aligner.align(
                [vec], forced_provider, baseline_provider, corpus_version, corpus_texts
            )
            vec = aligned[0]
            pair_key = (forced_provider.name, baseline_provider.name)
            if pair_key not in diagnostics_by_pair:
                _, diag = aligner.get_transform(forced_provider, baseline_provider, corpus_version, corpus_texts)
                diagnostics_by_pair[pair_key] = diag
        else:
            confidence = 1.0
        aligned_vectors.append(vec)
        confidences.append(confidence)
    aligned_arr = np.array(aligned_vectors)
    aligned_time = time.time() - t0

    # Self-retrieval-style metrics: each vector should best match its own
    # chunk's baseline embedding.
    sims_baseline = cosine_sim_matrix(baseline_arr, baseline_arr)
    sims_aligned = cosine_sim_matrix(aligned_arr, baseline_arr)
    n = len(texts)
    relevant = np.arange(n)

    top5_baseline = top_k_indices(sims_baseline, min(5, n))
    top5_aligned = top_k_indices(sims_aligned, min(5, n))
    overlap = np.mean([
        len(set(top5_baseline[i]) & set(top5_aligned[i])) / len(top5_baseline[i])
        for i in range(n)
    ])

    mrr_baseline = mean_reciprocal_rank(sims_baseline, relevant)
    mrr_aligned = mean_reciprocal_rank(sims_aligned, relevant)
    retention = (mrr_aligned / mrr_baseline) if mrr_baseline > 0 else 0.0

    log("")
    if console:
        table = Table(title="Alignment Quality Report")
        table.add_column("Metric")
        table.add_column("Value")
        for pair, diag in diagnostics_by_pair.items():
            table.add_row(f"Provider pair", f"{pair[0]} → {pair[1]}")
            ev = f"{diag.pca_explained_variance:.1%}" if diag.pca_explained_variance is not None else "n/a (same dim)"
            table.add_row("PCA explained variance at reduced dim", ev)
            table.add_row("Scale factor applied", f"{diag.scale_factor:.2f}")
            table.add_row("Anchor corpus alignment error (held-out, mean cosine)", f"{diag.mean_cosine_alignment_error:.2f}")
            if diag.pca_fallback_used:
                table.add_row("PCA guardrail fallback", diag.pca_fallback_used)
        table.add_row("Top-5 retrieval overlap vs baseline", f"{overlap:.0%}")
        table.add_row("MRR (baseline)", f"{mrr_baseline:.2f}")
        table.add_row("MRR (aligned)", f"{mrr_aligned:.2f}")
        table.add_row("Retrieval quality retention", f"~{retention:.0%}")
        table.add_row("Mean alignment_confidence across aligned chunks", f"{np.mean(confidences):.2f}")
        table.add_row("Provider switches during ingestion", str(provider_switches))
        table.add_row("Total chunks", str(n))
        table.add_row("Total time", f"{aligned_time:.1f}s")
        console.print(table)
    else:
        print("=== Alignment Quality Report ===")
        for pair, diag in diagnostics_by_pair.items():
            print(f"Provider pair: {pair[0]} -> {pair[1]}")
            ev = f"{diag.pca_explained_variance:.1%}" if diag.pca_explained_variance is not None else "n/a (same dim)"
            print(f"  PCA explained variance at reduced dim: {ev}")
            print(f"  Scale factor applied: {diag.scale_factor:.2f}")
            print(f"  Anchor corpus alignment error (held-out, mean cosine): {diag.mean_cosine_alignment_error:.2f}")
            if diag.pca_fallback_used:
                print(f"  PCA guardrail fallback: {diag.pca_fallback_used}")
        print(f"Top-5 retrieval overlap vs baseline: {overlap:.0%}")
        print(f"MRR (baseline): {mrr_baseline:.2f} | MRR (aligned): {mrr_aligned:.2f}")
        print(f"Retrieval quality retention: ~{retention:.0%}")
        print(f"Mean alignment_confidence: {np.mean(confidences):.2f}")
        print(f"Provider switches: {provider_switches} | Total chunks: {n} | Total time: {aligned_time:.1f}s")

    if domain_mismatch_check:
        log(
            "\n[domain-mismatch-check] This flag signals CI to treat this document as an "
            "intentionally domain-mismatched sample (e.g. legal/medical text far from the "
            "base anchor corpus). Compare 'retrieval quality retention' above against your "
            "CI threshold for mismatched domains -- expect it to be lower than well-matched "
            "documents, and fail the build if it drops below your chosen floor."
        )

    return {
        "overlap": overlap,
        "mrr_baseline": mrr_baseline,
        "mrr_aligned": mrr_aligned,
        "retention": retention,
        "provider_switches": provider_switches,
        "n_chunks": n,
    }


def main():
    parser = argparse.ArgumentParser(description="embedfallback evaluation harness")
    parser.add_argument("--doc", required=True, help="Path to a document (.txt, .md, .pdf, .html)")
    parser.add_argument("--strategy", default="semantic", choices=["fixed", "recursive", "semantic"])
    parser.add_argument("--providers", required=True, help="Comma-separated provider names, first is baseline (e.g. google,openai,local)")
    parser.add_argument("--domain-mismatch-check", action="store_true", help="Mark this run as an intentional domain-mismatch test case")
    args = parser.parse_args()

    provider_names = [p.strip() for p in args.providers.split(",")]
    run_evaluation(args.doc, args.strategy, provider_names, args.domain_mismatch_check)


if __name__ == "__main__":
    main()
