"""
General M-PSK / M-QAM demodulation and blind constellation-order
estimation, for modulations beyond the hand-written BPSK/FSK paths.
"""

import numpy as np

QAM_ORDERS = (4, 16, 64, 256)
PSK_ORDERS = (2, 4, 8, 16)


def _symbol_centers(baseband, sps):
    """Downsample a baseband signal to one sample per symbol, taking the
    mid-symbol sample (away from transition edges)."""
    offset = sps // 2
    n_symbols = (len(baseband) - offset) // sps
    idx = offset + np.arange(n_symbols) * sps
    return baseband[idx]


def _qam_constellation(order):
    """Square (or near-square for non-power-of-4 orders) Gray-ish QAM
    constellation, unit average power."""
    side = int(round(np.sqrt(order)))
    levels = np.arange(side) - (side - 1) / 2.0
    re, im = np.meshgrid(levels, levels)
    points = (re + 1j * im).flatten()
    points = points / np.sqrt(np.mean(np.abs(points) ** 2))
    return points


def _psk_constellation(order):
    angles = 2 * np.pi * np.arange(order) / order
    return np.exp(1j * angles)


def _constellation(order, mod_type):
    if mod_type.upper() == "QAM":
        return _qam_constellation(order)
    return _psk_constellation(order)


def estimate_constellation_order(baseband, sps, mod_type="QAM"):
    """
    Blindly pick the best-fitting constellation order for `mod_type` by
    normalizing the received symbols to unit average power, then scoring
    each candidate order by mean squared distance from the nearest ideal
    constellation point (lower = better fit).

    Returns (order, score).
    """
    symbols = _symbol_centers(baseband, sps)
    if len(symbols) == 0:
        raise ValueError("no symbols recovered - sps/signal length mismatch")

    power = np.mean(np.abs(symbols) ** 2)
    if power <= 0:
        raise ValueError("received symbols have zero power")
    symbols_norm = symbols / np.sqrt(power)

    candidates = QAM_ORDERS if mod_type.upper() == "QAM" else PSK_ORDERS

    # Raw mean-squared-distance alone is not comparable across orders: a
    # denser constellation (more points) is closer to ANY cloud of symbols
    # almost by construction, regardless of which order actually generated
    # them - verified this concretely on a 16-QAM test signal, where raw MSE
    # preferred 256-QAM (score 0.0041) over the true 16-QAM (0.0151) simply
    # because 256-QAM's points are packed 4x closer together. M-PSK has a
    # related but different problem: 4/8/16-PSK's constellations are strict
    # supersets of 2-PSK's, so plain BPSK data scores identically for
    # order 2/4/8 and only noise-sized differences separate them.
    #
    # Fix for both: normalize each order's raw score by that order's own
    # minimum inter-symbol distance squared (d_min^2) before comparing.
    # This is the standard fix for exactly this bias - it turns "mean
    # distance to nearest point" into something comparable to "distance as a
    # fraction of that order's own decision-region size", so a tighter grid
    # no longer wins purely by being tighter.
    scores = {}
    for order in candidates:
        const = _constellation(order, mod_type)
        # distance of every symbol to every constellation point
        dists = np.abs(symbols_norm[:, None] - const[None, :]) ** 2
        min_dists = np.min(dists, axis=1)
        raw_score = float(np.mean(min_dists))
        d_min_sq = float(np.min(np.abs(const[:, None] - const[None, :])[~np.eye(len(const), dtype=bool)]) ** 2)
        scores[order] = raw_score / d_min_sq

    best_order = min(candidates, key=lambda o: scores[o])
    best_score = scores[best_order]

    return best_order, best_score


def demodulate_mpsk_or_qam(baseband, sps, mod_type, order):
    """
    Slice `baseband` into symbols at `sps` samples/symbol, map each to the
    nearest point of an ideal (unit-power) M-PSK/M-QAM constellation, and
    return the recovered bitstream (log2(order) bits per symbol, MSB
    first, in constellation-index order - not Gray coded).
    """
    # Symbol points are the mean of the middle half of each symbol (noise averaging);
    # the caller is expected to have aligned `baseband` to a symbol boundary
    # (estimate_timing_offset) - for an already-aligned signal this is identical
    # to the old single mid-symbol sample on clean rectangular pulses.
    symbols = symbol_points_avg(baseband, sps)
    power = np.mean(np.abs(symbols) ** 2)
    if power <= 0:
        raise ValueError("received symbols have zero power")
    return hard_bits_from_points(symbols / np.sqrt(power), mod_type, order)


# ---------------------------------------------------------------------------
# Synchronisation that the original test signals never needed (they all start on
# a symbol boundary with carrier phase exactly 0) but real captures always do:
#   * symbol TIMING OFFSET  - where inside the sample stream a symbol starts
#   * carrier PHASE         - constant rotation left after frequency removal
#   * rotation AMBIGUITY    - blind phase recovery is only known modulo the
#                             constellation's symmetry (pi for BPSK, pi/2 for
#                             QPSK/QAM, 2*pi/M for M-PSK); the FEC stage resolves it.
# ---------------------------------------------------------------------------
def estimate_timing_offset(baseband, sps):
    """Index (0..sps-1) of the first sample of a symbol, from where the
    signal-transition energy |x[n+1]-x[n]|^2 concentrates when folded modulo sps."""
    sps = int(sps)
    nl = np.abs(np.diff(np.asarray(baseband))) ** 2
    n = (len(nl) // sps) * sps
    if n < sps * 4:
        return 0
    folded = nl[:n].reshape(-1, sps).mean(axis=0)
    return int((np.argmax(folded) + 1) % sps)


def symbol_points_avg(baseband, sps):
    """One complex value per symbol: mean of the middle half of each symbol
    (better noise averaging than the single mid-symbol sample). Assumes the
    baseband already starts on a symbol boundary (see estimate_timing_offset)."""
    sps = int(sps)
    n = len(baseband) // sps
    if n == 0:
        return np.zeros(0, dtype=complex)
    seg = np.asarray(baseband)[: n * sps].reshape(n, sps)
    lo = sps // 4
    hi = max(lo + 1, sps - sps // 4)
    return seg[:, lo:hi].mean(axis=1)


def estimate_phase(points, mod_type, order):
    """Constant carrier phase (radians) of symbol points, modulo the constellation symmetry.
    M-PSK: angle(sum y^M)/M.  Square QAM: the 4th power has a NEGATIVE real mean
    (E[c^4] < 0 for 4/16/64/256-QAM), so angle(sum y^4) = pi + 4*theta."""
    y = np.asarray(points)
    if len(y) == 0:
        return 0.0
    if mod_type.upper() == "QAM":
        return float((np.angle(np.sum(y ** 4)) - np.pi) / 4)
    return float(np.angle(np.sum(y ** order)) / order)


def ambiguity_angles(mod_type, order):
    """Rotations (radians) that map the constellation onto itself."""
    if mod_type.upper() == "QAM":
        return [k * np.pi / 2 for k in range(4)]
    return [k * 2 * np.pi / order for k in range(order)]


def _fit_score(points_norm, order, mod_type):
    const = _constellation(order, mod_type)
    d = np.min(np.abs(points_norm[:, None] - const[None, :]) ** 2, axis=1)
    dmin_sq = float(np.min(np.abs(const[:, None] - const[None, :])[~np.eye(len(const), dtype=bool)]) ** 2)
    return float(np.mean(d)) / dmin_sq


def estimate_order_and_phase(baseband, sps, mod_type="QAM"):
    """Blind constellation order + carrier phase, jointly. For every candidate order the
    phase is estimated with that order's own M-th-power statistic, the symbols are
    de-rotated, and the fit is scored (normalised by d_min^2, as in
    estimate_constellation_order). Returns (order, score, theta)."""
    pts = symbol_points_avg(baseband, sps)
    if len(pts) == 0:
        raise ValueError("no symbols recovered - sps/signal length mismatch")
    power = np.mean(np.abs(pts) ** 2)
    if power <= 0:
        raise ValueError("received symbols have zero power")
    pts = pts / np.sqrt(power)
    candidates = QAM_ORDERS if mod_type.upper() == "QAM" else PSK_ORDERS
    best = None
    for order in candidates:
        theta = estimate_phase(pts, mod_type, order)
        score = _fit_score(pts * np.exp(-1j * theta), order, mod_type)
        if best is None or score < best[1]:
            best = (order, score, theta)
    return best


def soft_demap_mpsk_or_qam(baseband, sps, mod_type, order, noise_var=None):
    """Max-log soft demapper: one LLR per bit, POSITIVE = "bit 1 more likely" (same
    convention as demodulate_bpsk_soft). Same bit labelling as demodulate_mpsk_or_qam
    (constellation index, MSB first).

        LLR_b = ( min_{c: bit_b(c)=0} |y-c|^2  -  min_{c: bit_b(c)=1} |y-c|^2 ) / noise_var

    noise_var is the complex noise variance of the symbol point relative to unit average
    symbol power (i.e. 1/SNR). If None the LLRs are unscaled distance differences:
    identical decisions, and fine for the Viterbi decoder (scale-invariant), but pass a
    real noise_var (e.g. from snr_m2m4) when a decoder needs true LLRs (LDPC)."""
    pts = symbol_points_avg(baseband, sps)
    power = np.mean(np.abs(pts) ** 2)
    if power <= 0:
        raise ValueError("received symbols have zero power")
    pts = pts / np.sqrt(power)
    const = _constellation(order, mod_type)
    d2 = np.abs(pts[:, None] - const[None, :]) ** 2            # (n_sym, M)
    bps = int(np.log2(order))
    idx = np.arange(order)
    llr = np.zeros((len(pts), bps))
    for b in range(bps):
        bit = (idx >> (bps - 1 - b)) & 1
        llr[:, b] = d2[:, bit == 0].min(axis=1) - d2[:, bit == 1].min(axis=1)
    if noise_var:
        llr = llr / noise_var
    return llr.reshape(-1)


def hard_bits_from_points(points_norm, mod_type, order):
    const = _constellation(order, mod_type)
    indices = np.argmin(np.abs(points_norm[:, None] - const[None, :]) ** 2, axis=1)
    bps = int(np.log2(order))
    shifts = bps - 1 - np.arange(bps)
    return ((indices[:, None] >> shifts[None, :]) & 1).reshape(-1).astype(int)


if __name__ == "__main__":
    # Quick self-test: 16-QAM, no noise, should recover order and bits
    # exactly.
    rng = np.random.default_rng(0)
    order = 16
    sps = 10
    n_symbols = 300
    const = _qam_constellation(order)
    tx_indices = rng.integers(0, order, n_symbols)
    tx_symbols = const[tx_indices]
    baseband = np.repeat(tx_symbols, sps)

    est_order, score = estimate_constellation_order(baseband, sps, "QAM")
    print("Estimated order:", est_order, "(true:", order, "), score:", round(score, 6))
    assert est_order == order, "self-test failed: order mismatch"

    bits = demodulate_mpsk_or_qam(baseband, sps, "QAM", order)
    bits_per_symbol = int(np.log2(order))
    n_recovered = len(bits) // bits_per_symbol
    rx_indices = np.array([
        int("".join(str(b) for b in bits[i * bits_per_symbol:(i + 1) * bits_per_symbol]), 2)
        for i in range(n_recovered)
    ])
    assert np.array_equal(rx_indices, tx_indices[:n_recovered]), "self-test failed: bit mismatch"
    print("demodulate_general self-test OK")
