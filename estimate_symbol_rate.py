"""
Blind symbol-rate / timing estimation.

Works on either a raw complex baseband signal or a pre-built discriminator
signal (e.g. the FSK hard-decision discriminator built in analysis_pipeline.py).
Approach: a nonlinearity (squared magnitude of the first difference) turns
symbol transitions into a periodic pulse train, whose period shows up as a
spectral line at the symbol rate. This is the standard "delay-and-multiply /
squaring" timing-recovery trick, done in one shot on the whole capture
instead of a PLL, which is enough for offline blind analysis.
"""

import numpy as np


def estimate_symbol_rate(sig, sample_rate, min_sps=2, max_sps=2000):
    """
    Estimate the symbol rate (Hz) and samples-per-symbol (int) of `sig`.

    Raises ValueError if no reliable periodicity can be found (e.g. signal
    too short, no transitions, or the estimate falls outside a sane
    min/max samples-per-symbol range).
    """
    sig = np.asarray(sig)
    n = len(sig)
    if n < 16:
        raise ValueError("signal too short to estimate symbol timing")

    # Nonlinearity: magnitude of the first difference, squared. Flat
    # sections between symbol transitions contribute ~0; transitions
    # produce a pulse. Squaring sharpens the pulses relative to noise.
    diffs = np.abs(np.diff(sig))
    nl = diffs ** 2

    if np.max(nl) <= 0 or np.allclose(nl, nl[0]):
        raise ValueError("no symbol transitions detected in signal")

    nl = nl - np.mean(nl)
    spectrum = np.abs(np.fft.rfft(nl))
    freqs = np.fft.rfftfreq(len(nl), d=1.0 / sample_rate)

    # Ignore DC and any bin whose implied samples-per-symbol falls outside
    # the sane range, so we don't lock onto a slow amplitude drift or
    # single-sample noise spikes.
    with np.errstate(divide="ignore"):
        implied_sps = sample_rate / np.where(freqs > 0, freqs, np.inf)
    valid = (freqs > 0) & (implied_sps >= min_sps) & (implied_sps <= max_sps)

    if not np.any(valid):
        raise ValueError("no spectral line found in the valid samples/symbol range")

    spectrum_valid = np.where(valid, spectrum, -1)
    peak_idx = int(np.argmax(spectrum_valid))
    peak_idx = _prefer_fundamental(spectrum_valid, freqs, peak_idx)
    symbol_rate = float(freqs[peak_idx])

    if symbol_rate <= 0:
        raise ValueError("estimated symbol rate is non-positive")

    sps = int(round(sample_rate / symbol_rate))
    if sps < min_sps or sps > max_sps:
        raise ValueError("estimated samples/symbol (" + str(sps) + ") outside sane range")

    return symbol_rate, sps


def _prefer_fundamental(spectrum, freqs, peak_idx, ratio=0.5, tol=0.03, max_harm=4):
    """The transition pulse train has a line at the symbol rate AND at every
    harmonic of it, all of nearly equal height (the pulses are ~1 sample wide), so
    plain argmax can land on ANY harmonic - up to the Nyquist limit, which gives a
    samples/symbol 2-10x too small. Pick instead the LOWEST strong line whose own
    harmonics (2f, 3f, 4f, up to Nyquist) are all comparably strong. A true symbol
    rate passes; its sub-harmonics do not (no line at f/2 for ordinary data).
    Falls back to the plain peak when no line qualifies (e.g. a discriminator signal
    whose harmonics are weak)."""
    peak_mag = float(spectrum[peak_idx])
    if peak_mag <= 0:
        return peak_idx
    thresh = ratio * peak_mag
    f_nyq = freqs[-1]
    is_peak = np.zeros(len(spectrum), dtype=bool)
    is_peak[1:-1] = (spectrum[1:-1] >= spectrum[:-2]) & (spectrum[1:-1] >= spectrum[2:])
    cands = np.where(is_peak & (spectrum >= thresh))[0]

    def line_mag(f):
        lo = np.searchsorted(freqs, f * (1 - tol))
        hi = np.searchsorted(freqs, f * (1 + tol)) + 1
        return float(spectrum[lo:hi].max()) if hi > lo else 0.0

    for idx in cands:                       # ascending frequency = lowest first
        f = freqs[idx]
        if f <= 0:
            continue
        k_max = min(max_harm, int(f_nyq / f))
        if k_max < 2:
            break                           # only the top line(s) left: nothing to verify
        if all(line_mag(k * f) >= thresh for k in range(2, k_max + 1)):
            return int(idx)
    return peak_idx


if __name__ == "__main__":
    # Quick self-test with a synthetic BPSK-like signal: rectangular
    # symbols of known sps, no channel noise.
    rng = np.random.default_rng(0)
    true_sps = 20
    sample_rate = 200000.0
    n_symbols = 500
    bits = rng.integers(0, 2, n_symbols)
    symbols = 2 * bits - 1
    baseband = np.repeat(symbols, true_sps).astype(complex)

    rate, sps = estimate_symbol_rate(baseband, sample_rate)
    print("Estimated symbol rate:", round(rate, 1), "Hz -> sps =", sps, "(true sps =", true_sps, ")")
    assert sps == true_sps, "self-test failed: sps mismatch"
    print("estimate_symbol_rate self-test OK")
