"""The Alignment Engine: makes vectors from different providers comparable
within one document's vector space.

Pipeline: anchor corpus -> per-provider anchor embeddings (cached) ->
scaled/affine Procrustes transform (with PCA guardrails when dimensions
differ) -> per-chunk application with an attached confidence score.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import requests as _requests

logger = logging.getLogger("embedfallback.alignment")

DEFAULT_CACHE_DIR = Path.home() / ".embedfallback" / "cache"

_MIN_EXPLAINED_VARIANCE = 0.90
_MIN_CORPUS_TO_DIM_RATIO = 5.0


# ---------------------------------------------------------------------------
# Anchor corpus
# ---------------------------------------------------------------------------

def load_base_anchor_corpus() -> list[str]:
    path = Path(__file__).parent / "assets" / "anchor_corpus_base.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_corpus_version(base_texts: list[str], extra_texts: list[str] | None) -> str:
    base_hash = hashlib.sha256("\u241f".join(base_texts).encode("utf-8")).hexdigest()[:12]
    if not extra_texts:
        return f"base-{base_hash}"
    extra_hash = hashlib.sha256("\u241f".join(extra_texts).encode("utf-8")).hexdigest()[:12]
    return f"base-{base_hash}_ext-{extra_hash}"


def get_anchor_corpus(extra_texts: list[str] | None = None) -> tuple[list[str], str]:
    base = load_base_anchor_corpus()
    combined = base + list(extra_texts) if extra_texts else base
    version = build_corpus_version(base, extra_texts)
    return combined, version


# ---------------------------------------------------------------------------
# Anchor embedding cache
# ---------------------------------------------------------------------------

class AnchorCache:
    # Anchor embeddings shipped with the package itself, pre-computed once
    # and committed to the repo. A fresh install checks here first, before
    # ~/.embedfallback/cache and before ever calling a provider's API --
    # this is what lets a brand-new user get working alignment on their
    # very first document, for the default models/corpus, with zero wait.
    _SHIPPED_CACHE_DIR = Path(__file__).parent / "assets" / "prebuilt_anchor_cache"

    def __init__(self, cache_dir: Path | str = DEFAULT_CACHE_DIR):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path_for(self, provider_name: str, model: str, corpus_version: str) -> Path:
        safe_model = model.replace("/", "_")
        return self.cache_dir / f"{provider_name}_{safe_model}_{corpus_version}.npy"

    def _shipped_path_for(self, provider_name: str, model: str, corpus_version: str) -> Path:
        safe_model = model.replace("/", "_")
        return self._SHIPPED_CACHE_DIR / f"{provider_name}_{safe_model}_{corpus_version}.npy"

    def get_or_compute(self, provider, corpus_version: str, corpus_texts: list[str]) -> np.ndarray:
        """Returns cached anchor embeddings for this provider+model+
        corpus_version. Checks, in order: (1) the package's shipped
        pre-built cache, (2) the user's local ~/.embedfallback/cache,
        (3) computes fresh via the provider's API if neither has it.

        Progress is also saved incrementally to a .partial.npy file after
        every successful batch, so a crash partway through doesn't lose
        everything already embedded -- a retry resumes from the last
        completed batch instead of starting over from zero.
        """
        path = self._path_for(provider.name, provider.model, corpus_version)
        if path.exists():
            logger.debug("Anchor cache hit (local): %s", path.name)
            return np.load(path)

        shipped_path = self._shipped_path_for(provider.name, provider.model, corpus_version)
        if shipped_path.exists():
            logger.debug("Anchor cache hit (shipped with package): %s", shipped_path.name)
            arr = np.load(shipped_path)
            # Copy into the user's local cache too, so future lookups hit
            # the fast path without touching the read-only package assets.
            np.save(path, arr)
            return arr

        partial_path = path.with_suffix(".partial.npy")
        vectors: list[list[float]] = []
        start_i = 0
        if partial_path.exists():
            partial = np.load(partial_path)
            vectors = partial.tolist()
            start_i = len(vectors)
            logger.info(
                "Resuming anchor embedding for %s/%s @ %s from batch %d "
                "(found partial progress on disk).",
                provider.name, provider.model, corpus_version, start_i,
            )
        else:
            logger.info(
                "Anchor cache miss for %s/%s @ %s -- embedding %d anchor texts.",
                provider.name, provider.model, corpus_version, len(corpus_texts),
            )

        batch_size = provider.max_batch_size()
        i = start_i
        retry_counts: dict[int, int] = {}

        while i < len(corpus_texts):
            batch = corpus_texts[i:i + batch_size]
            try:
                result = provider.embed(batch)
            except Exception as e:
                from .providers.base import RateLimitExceeded, RateLimitKind

                if isinstance(e, _requests.exceptions.RequestException) and not isinstance(e, RateLimitExceeded):
                    logger.warning(
                        "Anchor embedding hit a network error on %s (batch %d-%d/%d): %s. "
                        "Retrying in 10s...",
                        provider.name, i, i + len(batch), len(corpus_texts), e,
                    )
                    time.sleep(10)
                    continue

                if not isinstance(e, RateLimitExceeded):
                    raise

                if e.kind == RateLimitKind.QUOTA_EXHAUSTED:
                    raise RuntimeError(
                        f"Anchor embedding stopped: '{provider.name}' has hit its "
                        f"daily/quota limit (batch {i}-{i + len(batch)}/{len(corpus_texts)} "
                        f"of anchor corpus). This won't clear soon -- try again "
                        f"tomorrow, or switch to a provider with quota remaining. "
                        f"Progress so far is saved -- rerunning later will resume "
                        f"from batch {i}. ({e.message})"
                    ) from e

                wait = e.retry_after_seconds if e.retry_after_seconds else 60.0
                retry_counts[i] = retry_counts.get(i, 0) + 1
                if retry_counts[i] > 5:
                    raise RuntimeError(
                        f"Anchor embedding stopped: '{provider.name}' failed "
                        f"5 times in a row on batch {i}-{i + len(batch)}/{len(corpus_texts)}. "
                        f"Stopping instead of retrying forever. Progress so far is "
                        f"saved -- rerunning later will resume from batch {i}. "
                        f"({e.message})"
                    ) from e

                logger.warning(
                    "Anchor embedding rate-limited on %s (batch %d-%d/%d, attempt %d/5). "
                    "Waiting %.0fs before retrying this batch...",
                    provider.name, i, i + len(batch), len(corpus_texts),
                    retry_counts[i], wait,
                )
                time.sleep(wait + 1.0)
                continue

            vectors.extend(result.vectors)
            i += batch_size
            np.save(partial_path, np.array(vectors, dtype=np.float64))

        arr = np.array(vectors, dtype=np.float64)
        np.save(path, arr)
        partial_path.unlink(missing_ok=True)
        return arr


# ---------------------------------------------------------------------------
# Transform computation
# ---------------------------------------------------------------------------

@dataclass
class AlignmentDiagnostics:
    pca_explained_variance: float | None
    anchor_corpus_size: int
    reduced_dimension: int | None
    mean_cosine_alignment_error: float
    scale_factor: float
    pca_fallback_used: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AlignmentTransform:
    rotation: np.ndarray
    scale: float
    pca_source_components: np.ndarray | None = None
    pca_target_components: np.ndarray | None = None
    pca_source_mean: np.ndarray | None = None
    pca_target_mean: np.ndarray | None = None
    pad_to_dim: int | None = None
    target_dim: int | None = None

    def apply(self, vectors: np.ndarray) -> np.ndarray:
        v = vectors
        if self.pca_source_components is not None:
            v = (v - self.pca_source_mean) @ self.pca_source_components
        elif self.pad_to_dim is not None and v.shape[1] < self.pad_to_dim:
            pad_width = self.pad_to_dim - v.shape[1]
            v = np.pad(v, ((0, 0), (0, pad_width)), mode="constant")

        v = self.scale * (v @ self.rotation)

        if self.pca_target_components is not None:
            v = v @ self.pca_target_components.T + self.pca_target_mean
        elif self.pad_to_dim is not None and self.target_dim is not None and v.shape[1] > self.target_dim:
            v = v[:, : self.target_dim]
        return v


def _fit_pca(x: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, float]:
    mean = x.mean(axis=0)
    centered = x - mean
    u, s, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:k].T
    total_var = (s ** 2).sum()
    explained = (s[:k] ** 2).sum() / total_var if total_var > 0 else 0.0
    return components, mean, float(explained)


def compute_alignment(
    source_anchors: np.ndarray,
    target_anchors: np.ndarray,
    min_explained_variance: float = _MIN_EXPLAINED_VARIANCE,
    min_corpus_to_dim_ratio: float = _MIN_CORPUS_TO_DIM_RATIO,
    held_out_fraction: float = 0.2,
    random_seed: int = 0,
) -> tuple[AlignmentTransform, AlignmentDiagnostics]:
    n = source_anchors.shape[0]
    assert target_anchors.shape[0] == n, "Anchor sets must be paired 1:1"

    rng = np.random.default_rng(random_seed)
    perm = rng.permutation(n)
    n_holdout = max(1, int(n * held_out_fraction))
    holdout_idx, fit_idx = perm[:n_holdout], perm[n_holdout:]

    src_fit, tgt_fit = source_anchors[fit_idx], target_anchors[fit_idx]
    src_hold, tgt_hold = source_anchors[holdout_idx], target_anchors[holdout_idx]

    d_src, d_tgt = source_anchors.shape[1], target_anchors.shape[1]
    pca_src_comp = pca_tgt_comp = pca_src_mean = pca_tgt_mean = None
    explained_variance = None
    reduced_dim = None
    pad_to_dim = None
    fallback_used = None

    src_work, tgt_work = src_fit, tgt_fit

    if d_src != d_tgt:
        reduced_dim = min(d_src, d_tgt)
        n_fit = len(fit_idx)
        ratio = n_fit / reduced_dim if reduced_dim > 0 else 0.0

        if d_src > reduced_dim:
            pca_src_comp, pca_src_mean, ev_src = _fit_pca(src_fit, reduced_dim)
        else:
            ev_src = 1.0
        if d_tgt > reduced_dim:
            pca_tgt_comp, pca_tgt_mean, ev_tgt = _fit_pca(tgt_fit, reduced_dim)
        else:
            ev_tgt = 1.0

        explained_variance = min(ev_src, ev_tgt)
        guardrail_failed = (
            explained_variance < min_explained_variance or ratio < min_corpus_to_dim_ratio
        )

        if guardrail_failed:
            logger.warning(
                "PCA guardrail triggered (explained_variance=%.3f, "
                "corpus/dim ratio=%.1fx, thresholds=%.2f/%.1fx). "
                "Falling back to zero-padding the smaller space instead of "
                "trusting an under-supported PCA basis.",
                explained_variance, ratio, min_explained_variance, min_corpus_to_dim_ratio,
            )
            pca_src_comp = pca_tgt_comp = pca_src_mean = pca_tgt_mean = None
            pad_to_dim = max(d_src, d_tgt)
            fallback_used = "padded_zeros"
            reduced_dim = pad_to_dim

            def pad(x, target_dim):
                if x.shape[1] >= target_dim:
                    return x
                return np.pad(x, ((0, 0), (0, target_dim - x.shape[1])), mode="constant")

            src_work = pad(src_fit, pad_to_dim)
            tgt_work = pad(tgt_fit, pad_to_dim)
        else:
            src_work = (src_fit - pca_src_mean) @ pca_src_comp if pca_src_comp is not None else src_fit
            tgt_work = (tgt_fit - pca_tgt_mean) @ pca_tgt_comp if pca_tgt_comp is not None else tgt_fit

    m = src_work.T @ tgt_work
    u, s, vt = np.linalg.svd(m)
    rotation = u @ vt
    norm_src_sq = float((src_work ** 2).sum())
    scale = float(s.sum() / norm_src_sq) if norm_src_sq > 0 else 1.0

    transform = AlignmentTransform(
        rotation=rotation,
        scale=scale,
        pca_source_components=pca_src_comp,
        pca_target_components=pca_tgt_comp,
        pca_source_mean=pca_src_mean,
        pca_target_mean=pca_tgt_mean,
        pad_to_dim=pad_to_dim,
        target_dim=d_tgt,
    )

    predicted_hold = transform.apply(src_hold)
    predicted_norms = np.linalg.norm(predicted_hold, axis=1, keepdims=True)
    target_norms = np.linalg.norm(tgt_hold, axis=1, keepdims=True)
    predicted_norms[predicted_norms == 0] = 1e-12
    target_norms[target_norms == 0] = 1e-12
    cos_sim = np.sum((predicted_hold / predicted_norms) * (tgt_hold / target_norms), axis=1)
    mean_cosine_error = float(1.0 - cos_sim.mean())

    diagnostics = AlignmentDiagnostics(
        pca_explained_variance=explained_variance,
        anchor_corpus_size=n,
        reduced_dimension=reduced_dim,
        mean_cosine_alignment_error=mean_cosine_error,
        scale_factor=scale,
        pca_fallback_used=fallback_used,
    )
    return transform, diagnostics


# ---------------------------------------------------------------------------
# Runtime application
# ---------------------------------------------------------------------------

class AlignmentEngine:
    def __init__(self, anchor_cache: AnchorCache | None = None):
        self.anchor_cache = anchor_cache or AnchorCache()
        self._transforms: dict[tuple[str, str, str], tuple[AlignmentTransform, AlignmentDiagnostics]] = {}

    def get_transform(
        self,
        source_provider,
        target_provider,
        corpus_version: str,
        corpus_texts: list[str],
    ) -> tuple[AlignmentTransform, AlignmentDiagnostics]:
        key = (source_provider.name, target_provider.name, corpus_version)
        if key in self._transforms:
            return self._transforms[key]

        source_anchors = self.anchor_cache.get_or_compute(source_provider, corpus_version, corpus_texts)
        target_anchors = self.anchor_cache.get_or_compute(target_provider, corpus_version, corpus_texts)

        transform, diagnostics = compute_alignment(source_anchors, target_anchors)
        self._transforms[key] = (transform, diagnostics)
        return transform, diagnostics

    def align(
        self,
        vectors: list[list[float]],
        source_provider,
        target_provider,
        corpus_version: str,
        corpus_texts: list[str],
    ) -> tuple[list[list[float]], float]:
        if source_provider.name == target_provider.name:
            return vectors, 1.0

        transform, diagnostics = self.get_transform(
            source_provider, target_provider, corpus_version, corpus_texts
        )
        arr = np.array(vectors, dtype=np.float64)
        aligned = transform.apply(arr)
        confidence = max(0.0, min(1.0, 1.0 - diagnostics.mean_cosine_alignment_error))
        return aligned.tolist(), confidence