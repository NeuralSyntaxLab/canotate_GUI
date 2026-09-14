"""Correctness tests for the ZNCC core.

The fast path uses FFT convolution and running sums; the point of these tests
is that it agrees exactly with a naive direct implementation, so that any later
disagreement in the experiment is about the data and not about the arithmetic.

Run: python -m pytest sandbox/xcorr/test_scoring.py -q
"""
import numpy as np
import pytest

from scoring import (zncc_surface, zncc_surface_reference, peak_psr,
                     band_limit, modulation_band, decide, TemplateBank)


def test_matches_direct_implementation_on_random_input():
    rng = np.random.default_rng(0)
    for _ in range(20):
        n_freq = int(rng.integers(3, 25))
        m = int(rng.integers(2, 15))
        n = m + int(rng.integers(0, 60))
        phrase = rng.normal(size=(n_freq, n))
        template = rng.normal(size=(n_freq, m))
        np.testing.assert_allclose(
            zncc_surface(phrase, template),
            zncc_surface_reference(phrase, template),
            atol=1e-10)


def test_matches_direct_implementation_on_nonnegative_input():
    """Spectrograms are non-negative, which is where catastrophic cancellation
    in the running-sum variance would show up if it were going to."""
    rng = np.random.default_rng(1)
    phrase = rng.gamma(2.0, 1.0, size=(40, 300)) + 100.0
    template = rng.gamma(2.0, 1.0, size=(40, 25)) + 100.0
    np.testing.assert_allclose(zncc_surface(phrase, template),
                               zncc_surface_reference(phrase, template),
                               atol=1e-8)


def test_exact_match_scores_one():
    rng = np.random.default_rng(2)
    phrase = rng.normal(size=(10, 100))
    template = phrase[:, 30:45]
    surface = zncc_surface(phrase, template)
    assert surface.argmax() == 30
    assert surface[30] == pytest.approx(1.0)


def test_invariant_to_affine_rescaling():
    """ZNCC removes gain and offset — the reason absolute calibration is not
    needed for a within-patch comparison."""
    rng = np.random.default_rng(3)
    phrase = rng.normal(size=(8, 80))
    template = rng.normal(size=(8, 12))
    base = zncc_surface(phrase, template)
    np.testing.assert_allclose(zncc_surface(3.5 * phrase + 7.0, template),
                               base, atol=1e-10)
    np.testing.assert_allclose(zncc_surface(phrase, 2.0 * template - 4.0),
                               base, atol=1e-10)


def test_constant_input_does_not_divide_by_zero():
    phrase = np.ones((5, 50))
    template = np.ones((5, 10))
    assert np.all(zncc_surface(phrase, template) == 0.0)


def test_rejects_shorter_phrase_and_mismatched_bands():
    with pytest.raises(ValueError):
        zncc_surface(np.ones((5, 3)), np.ones((5, 10)))
    with pytest.raises(ValueError):
        zncc_surface(np.ones((5, 30)), np.ones((6, 10)))


def test_psr_is_high_for_a_sharp_peak_and_low_for_noise():
    rng = np.random.default_rng(4)
    noisy = rng.normal(scale=0.1, size=200)
    _, _, psr_noise = peak_psr(noisy, template_cols=5)
    sharp = rng.normal(scale=0.1, size=200)
    sharp[100] = 5.0
    _, _, psr_sharp = peak_psr(sharp, template_cols=5)
    assert psr_sharp > 10 * psr_noise


def test_psr_is_nan_when_the_sidelobe_is_too_small():
    """A surface with too few lags yields no sidelobe estimate; that must be
    reported as nan rather than silently returning a number."""
    _, _, psr = peak_psr(np.array([0.1, 0.9, 0.2]), template_cols=5)
    assert np.isnan(psr)


def test_psr_excludes_the_region_around_the_peak():
    surface = np.zeros(100)
    surface[50] = 1.0
    surface[48] = surface[52] = 0.9   # shoulders, inside the exclusion zone
    _, lag, psr = peak_psr(surface, template_cols=5)
    assert lag == 50
    # Shoulders excluded, so the sidelobe is all zeros -> zero spread -> nan.
    assert np.isnan(psr)


def test_band_limit_selects_rows_and_rejects_empty_bands():
    S = np.arange(50).reshape(5, 10).astype(float)
    F = np.array([0.0, 100.0, 200.0, 300.0, 400.0])
    S_band, F_band = band_limit(S, F, 100.0, 300.0)
    assert S_band.shape == (3, 10)
    np.testing.assert_array_equal(F_band, [100.0, 200.0, 300.0])
    with pytest.raises(ValueError):
        band_limit(S, F, 1000.0, 2000.0)


def test_modulation_band_follows_the_modulating_rows_not_the_loudest():
    """Row 0 is loud but constant; rows 5-7 are quiet but modulating.  The band
    must follow the modulation, which is where the syllable structure is."""
    F = np.linspace(0, 10000, 21)
    patch = np.zeros((21, 60))
    patch[0, :] = 1000.0
    rng = np.random.default_rng(5)
    patch[5:8, :] = rng.normal(loc=1.0, scale=5.0, size=(3, 60))
    fmin, fmax = modulation_band([patch], F)
    assert fmin >= F[4] and fmax <= F[9]


def test_decide_returns_argmax_and_signed_margin():
    scores = {4: {"peak": 0.40}, 5: {"peak": 0.81}, 6: {"peak": 0.62}}
    assert decide(scores, assigned_label=5) == (5, pytest.approx(0.19))
    predicted, margin = decide(scores, assigned_label=6)
    assert predicted == 5 and margin == pytest.approx(-0.19)


def test_template_bank_matches_the_single_surface_path():
    """The production path must produce the same numbers as the simple one,
    or the sandbox's measured accuracy does not transfer to it."""
    rng = np.random.default_rng(10)
    phrase = rng.gamma(2.0, 1.0, size=(30, 400))
    bank_input = {
        4: [rng.gamma(2.0, 1.0, size=(30, 50))],
        5: [rng.gamma(2.0, 1.0, size=(30, 50)),
            rng.gamma(2.0, 1.0, size=(30, 37))],
    }
    bank = TemplateBank(bank_input)
    surfaces = bank.surfaces(phrase)
    for label, stack in bank_input.items():
        for k, template in enumerate(stack):
            np.testing.assert_allclose(surfaces[label][k],
                                       zncc_surface(phrase, template),
                                       atol=1e-8)


def test_template_bank_reuses_its_transform_across_phrases():
    """Templates are transformed once per FFT length, not once per phrase."""
    rng = np.random.default_rng(11)
    bank = TemplateBank({1: [rng.normal(size=(12, 20))]})
    phrase = rng.normal(size=(12, 200))
    bank.surfaces(phrase)
    first = bank._spectra[1][0]
    bank.surfaces(rng.normal(size=(12, 200)))
    assert bank._spectra[1][0] is first


def test_template_bank_marks_templates_longer_than_the_phrase():
    rng = np.random.default_rng(12)
    bank = TemplateBank({1: [rng.normal(size=(8, 60))]})
    assert bank.surfaces(rng.normal(size=(8, 30)))[1] == [None]


def test_template_bank_rejects_mixed_frequency_bands():
    with pytest.raises(ValueError):
        TemplateBank({1: [np.ones((8, 10))], 2: [np.ones((9, 10))]})
