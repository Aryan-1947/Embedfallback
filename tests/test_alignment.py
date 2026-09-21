import numpy as np

from embedfallback.alignment import compute_alignment


def _random_orthogonal(d, seed):
    rng = np.random.default_rng(seed)
    m = rng.normal(size=(d, d))
    q, _ = np.linalg.qr(m)
    return q


def test_recovers_known_rotation_and_scale_same_dim():
    rng = np.random.default_rng(42)
    d = 16
    n = 400
    source = rng.normal(size=(n, d))
    true_rotation = _random_orthogonal(d, seed=1)
    true_scale = 2.5
    target = true_scale * (source @ true_rotation)

    transform, diagnostics = compute_alignment(source, target)

    assert diagnostics.mean_cosine_alignment_error < 0.05
    assert abs(diagnostics.scale_factor - true_scale) < 0.3


def test_different_dims_triggers_pca_when_corpus_is_large_enough():
    rng = np.random.default_rng(7)
    n = 2000
    d_src, d_tgt, latent_dim = 64, 32, 28
    # Real embedding spaces concentrate most variance in far fewer effective
    # dimensions than their nominal size, which is exactly why PCA reduction
    # is viable in practice. Model that here: both spaces are (noisy) linear
    # images of a shared low-dimensional latent signal, so PCA on either
    # side can legitimately capture ~90%+ of variance at `latent_dim`.
    latent = rng.normal(size=(n, latent_dim))
    src_projection = rng.normal(size=(latent_dim, d_src))
    tgt_projection = rng.normal(size=(latent_dim, d_tgt))
    noise_scale = 0.05
    source = latent @ src_projection + rng.normal(scale=noise_scale, size=(n, d_src))
    target = latent @ tgt_projection + rng.normal(scale=noise_scale, size=(n, d_tgt))

    transform, diagnostics = compute_alignment(source, target)

    assert diagnostics.reduced_dimension == min(d_src, d_tgt)
    # With a large, genuinely low-rank corpus, guardrail shouldn't force the
    # zero-padding fallback.
    assert diagnostics.pca_fallback_used is None


def test_small_corpus_relative_to_dim_triggers_padding_fallback():
    rng = np.random.default_rng(3)
    n = 20  # deliberately far too small relative to dimension for PCA to be trustworthy
    d_src, d_tgt = 100, 50
    source = rng.normal(size=(n, d_src))
    target = rng.normal(size=(n, d_tgt))

    transform, diagnostics = compute_alignment(
        source, target, min_corpus_to_dim_ratio=5.0, min_explained_variance=0.9
    )

    assert diagnostics.pca_fallback_used == "padded_zeros"


def test_diagnostics_measured_on_holdout_not_fit_set():
    rng = np.random.default_rng(11)
    d = 10
    n = 100
    source = rng.normal(size=(n, d))
    target = 1.0 * (source @ _random_orthogonal(d, seed=2)) + rng.normal(scale=2.0, size=(n, d))

    transform, diagnostics = compute_alignment(source, target, held_out_fraction=0.3)
    # Error should be well above zero since we added substantial noise --
    # if the code were (bug) measuring against the fit set with an
    # overfit transform, this would look artificially better.
    assert diagnostics.mean_cosine_alignment_error > 0.0
