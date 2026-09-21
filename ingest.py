#!/usr/bin/env python3
"""Command-line interface for embedfallback -- no Python coding required.

Usage:
    python ingest.py --doc my_document.pdf --output vectors.json
    python ingest.py --doc my_document.pdf --output vectors.json --strategy fixed
    python ingest.py --doc my_document.pdf --output vectors.json --providers google,cohere

Reads API keys from environment variables (or a .env file in the current
directory): GOOGLE_API_KEY, COHERE_API_KEY, VOYAGE_API_KEY, HF_API_KEY.
Only providers whose key is actually set get included automatically,
unless --providers explicitly names a smaller subset.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from dotenv import load_dotenv
load_dotenv()

from embedfallback import IngestionPipeline, providers as provider_module

_PROVIDER_BUILDERS = {
    "google": lambda key: provider_module.Google(api_key=key),
    "cohere": lambda key: provider_module.Cohere(api_key=key),
    "voyage": lambda key: provider_module.Voyage(api_key=key),
    "huggingface": lambda key: provider_module.HuggingFace(api_key=key),
}

_PROVIDER_ENV_VARS = {
    "google": "GOOGLE_API_KEY",
    "cohere": "COHERE_API_KEY",
    "voyage": "VOYAGE_API_KEY",
    "huggingface": "HF_API_KEY",
}


def build_providers(requested: list[str] | None) -> list:
    names = requested or list(_PROVIDER_BUILDERS.keys())
    built = []
    missing = []

    for name in names:
        name = name.strip().lower()
        if name not in _PROVIDER_BUILDERS:
            print(f"Unknown provider: {name!r}. Valid options: {list(_PROVIDER_BUILDERS.keys())}", file=sys.stderr)
            sys.exit(1)
        env_var = _PROVIDER_ENV_VARS[name]
        key = os.environ.get(env_var)
        if key:
            built.append(_PROVIDER_BUILDERS[name](key))
        elif requested is not None:
            # User explicitly asked for this provider -- missing key is an error.
            missing.append((name, env_var))

    if missing:
        for name, env_var in missing:
            print(f"Requested provider {name!r} but {env_var} is not set.", file=sys.stderr)
        sys.exit(1)

    if not built:
        print(
            "No API keys found. Set at least one of: "
            + ", ".join(_PROVIDER_ENV_VARS.values())
            + " (as environment variables or in a .env file).",
            file=sys.stderr,
        )
        sys.exit(1)

    return built


def main():
    parser = argparse.ArgumentParser(description="embedfallback CLI -- ingest a document, get vectors out, no Python required.")
    parser.add_argument("--doc", required=True, help="Path to the document (.pdf, .html, .md, .txt)")
    parser.add_argument("--output", required=True, help="Path to write the output JSON file")
    parser.add_argument("--strategy", default="recursive", choices=["fixed", "recursive", "semantic"], help="Chunking strategy (default: recursive)")
    parser.add_argument("--providers", default=None, help="Comma-separated provider names to use (default: all providers with a key set). Options: google, cohere, voyage, huggingface")
    args = parser.parse_args()

    requested = args.providers.split(",") if args.providers else None
    provs = build_providers(requested)

    provider_names = [p.name for p in provs]
    print(f"Using providers: {provider_names}")
    print(f"Ingesting {args.doc!r} with '{args.strategy}' chunking...")

    pipeline = IngestionPipeline(providers=provs, chunk_strategy=args.strategy)
    result = pipeline.ingest(args.doc)

    output_data = {
        "doc_id": result.doc_id,
        "canonical_provider": result.canonical_provider,
        "canonical_model": result.canonical_model,
        "provider_switches": result.provider_switches,
        "elapsed_seconds": result.elapsed_seconds,
        "chunks": [
            {
                "text": chunk.text,
                "vector": chunk.vector,
                "provider": chunk.provider,
                "alignment_confidence": chunk.alignment_confidence,
                "metadata": chunk.metadata,
            }
            for chunk in result.chunks
        ],
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nDone. {len(result.chunks)} chunks written to {args.output!r}")
    print(f"Canonical provider: {result.canonical_provider}")
    print(f"Provider switches during ingestion: {result.provider_switches}")
    print(f"Total time: {result.elapsed_seconds:.1f}s")


if __name__ == "__main__":
    main()