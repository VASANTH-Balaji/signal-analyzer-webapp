"""
LDPC encode/decode - pure NumPy, no external LDPC library.

Originally this wrapped `pyldpc`, but pyldpc depends on `numba` for its
decoder, and numba needs to JIT-compile and load native code at runtime.
On a machine with an Application Control / WDAC policy (common on
locked-down college lab PCs), loading that unsigned compiled code is
blocked outright - no pip flag or version pin fixes that, since the
block happens at the OS level, not the Python level. So this module
builds the LDPC code and decodes it with plain NumPy: nothing here is
compiled or JIT'd, so there's nothing for an Application Control policy
to object to.

What's implemented:
  - build_ldpc_code(n, d_v, d_c, seed): builds a regular (d_v, d_c) LDPC
    parity-check matrix H and a systematic generator matrix G, purely
    from the four given numbers (deterministic given the seed - the
    same call on the encode side and the decode side reproduces the
    identical H/G, so nothing needs to be shipped in the file metadata
    except these four numbers).
  - ldpc_encode(message_bits, G): systematic GF(2) encode.
  - ldpc_decode(soft_symbols, H, G, perm, snr_db=None, maxiter=50):
    min-sum belief-propagation decode of soft (un-thresholded) BPSK
    symbols, returning the recovered message bits.

Bit convention: a codeword bit d in {0, 1} is transmitted as the BPSK
symbol (-1)**d, i.e. d=0 -> +1, d=1 -> -1. Both the test generator and
the decoder here need to agree on this, since it's baked into the
channel log-likelihood-ratio formula below.
"""

import numpy as np


# ---------- code construction ----------

def build_regular_h(n, d_v, d_c, seed):
    """Random regular LDPC parity-check matrix: every column has exactly
    d_v ones, every row exactly d_c ones. Built by pairing up n*d_v
    "variable sockets" with m*d_c "check sockets" via a random
    permutation, retrying if that happens to produce a repeated
    (variable, check) pair (a multi-edge, not allowed in a simple
    Tanner graph)."""
    rng = np.random.default_rng(seed)
    if (n * d_v) % d_c != 0:
        raise ValueError("n * d_v must be divisible by d_c")
    m = n * d_v // d_c

    for _ in range(500):
        var_sockets = np.repeat(np.arange(n), d_v)
        chk_sockets = np.repeat(np.arange(m), d_c)
        rng.shuffle(chk_sockets)
        pairs = set(zip(var_sockets.tolist(), chk_sockets.tolist()))
        if len(pairs) == len(var_sockets):
            H = np.zeros((m, n), dtype=np.uint8)
            for v, c in pairs:
                H[c, v] = 1
            return H
    raise RuntimeError("could not build a simple regular H - try a different seed")


def _h_to_systematic(H):
    """GF(2) Gaussian elimination (row XOR + column swaps) reducing H to
    [I_m | P] form. Returns the reduced matrix and the column
    permutation applied (perm[j] = which original column ended up at
    position j). This necessarily destroys H's sparsity/regularity -
    it's only used to derive G, never used for decoding."""
    H = H.copy().astype(np.uint8)
    m, n = H.shape
    perm = np.arange(n)
    row = 0
    for col in range(n):
        if row >= m:
            break
        pivot = None
        for r in range(row, m):
            if H[r, col] == 1:
                pivot = r
                break
        if pivot is None:
            swap_col = None
            for c2 in range(col + 1, n):
                if np.any(H[row:, c2] == 1):
                    swap_col = c2
                    break
            if swap_col is None:
                continue
            H[:, [col, swap_col]] = H[:, [swap_col, col]]
            perm[[col, swap_col]] = perm[[swap_col, col]]
            for r in range(row, m):
                if H[r, col] == 1:
                    pivot = r
                    break
        H[[row, pivot]] = H[[pivot, row]]
        for r in range(m):
            if r != row and H[r, col] == 1:
                H[r] = H[r] ^ H[row]
        row += 1
    if row < m:
        raise RuntimeError("H does not have full row rank - try a different seed")
    return H, perm


def build_ldpc_code(n, d_v, d_c, seed):
    """
    Build a regular LDPC code from just (n, d_v, d_c, seed). Returns
    (H, G, perm):
      H    - (n*d_v/d_c) x n regular parity-check matrix (sparse structure,
             used for decoding)
      G    - n x k systematic generator matrix (k = n - m)
      perm - column permutation needed to recover the k message bits
             from a decoded codeword: message = codeword[perm[m:]]
    """
    H = build_regular_h(n, d_v, d_c, seed)
    m = H.shape[0]
    k = n - m
    H_sys, perm = _h_to_systematic(H)
    P = H_sys[:, m:]                                     # m x k
    G_perm = np.vstack([P, np.eye(k, dtype=np.uint8)])    # n x k
    G = np.zeros_like(G_perm)
    G[perm, :] = G_perm
    return H, G, perm


# ---------- encode ----------

def ldpc_encode(message_bits, G):
    message_bits = np.asarray(message_bits, dtype=np.uint8)
    return (G @ message_bits) % 2


def message_from_codeword(codeword_bits, perm, k):
    """Inverse of the systematic encode: pull the k message bits back
    out of a (correctly decoded) codeword."""
    m = len(perm) - k
    return np.asarray(codeword_bits)[perm[m:]]


# ---------- blind SNR estimate ----------

def estimate_snr_db(soft_symbols):
    """Blind SNR estimate: assumes the ideal symbol magnitude is the
    median magnitude of what was actually received (no assumption about
    transmit power/AGC), and treats deviation from that as noise."""
    soft_symbols = np.asarray(soft_symbols, dtype=float)
    amp = np.median(np.abs(soft_symbols))
    if amp <= 0:
        return 0.0
    normalized = soft_symbols / amp
    noise_var = float(np.mean((np.abs(normalized) - 1.0) ** 2))
    noise_var = max(noise_var, 1e-6)
    snr_linear = 1.0 / (2 * noise_var)
    return float(10 * np.log10(snr_linear))


# ---------- belief-propagation decode (min-sum, pure NumPy) ----------

def _build_edge_maps(H):
    """For a regular H, return check_edges (m x d_c) and var_edges
    (n x d_v): each holds the global edge ids incident to that
    check/variable, so the min-sum update can be done as dense NumPy
    ops instead of a per-node Python loop."""
    m, n = H.shape
    d_c = int(H.sum(axis=1)[0])
    d_v = int(H.sum(axis=0)[0])
    check_fill = np.zeros(m, dtype=int)
    var_fill = np.zeros(n, dtype=int)
    check_edges = np.zeros((m, d_c), dtype=int)
    var_edges = np.zeros((n, d_v), dtype=int)
    edge_var_of = np.zeros(m * d_c, dtype=int)
    e = 0
    for c in range(m):
        for v in np.nonzero(H[c])[0]:
            check_edges[c, check_fill[c]] = e
            var_edges[v, var_fill[v]] = e
            edge_var_of[e] = v
            check_fill[c] += 1
            var_fill[v] += 1
            e += 1
    return check_edges, var_edges, edge_var_of, d_v, d_c


def ldpc_decode_codeword(H, soft_y, snr_db, maxiter=50):
    """Min-sum belief-propagation decode of soft BPSK symbols. Returns
    the decoded codeword bits (length n) and the number of iterations
    used (stops early once a valid codeword - all parity checks
    satisfied - is found)."""
    m, n = H.shape
    check_edges, var_edges, edge_var_of, d_v, d_c = _build_edge_maps(H)

    snr_linear = 10 ** (snr_db / 10.0)
    sigma2 = 1.0 / (2 * snr_linear)
    channel_llr = 2 * np.asarray(soft_y, dtype=float) / sigma2

    msg_v2c = channel_llr[edge_var_of].copy()
    msg_c2v = np.zeros(m * d_c, dtype=float)
    d_hat = (channel_llr < 0).astype(int)

    for iteration in range(maxiter):
        vals = msg_v2c[check_edges]                 # (m, d_c)
        signs = np.sign(vals)
        signs[signs == 0] = 1
        abs_vals = np.abs(vals)

        total_sign = np.prod(signs, axis=1, keepdims=True)
        leave_out_sign = total_sign * signs

        sorted_idx = np.argsort(abs_vals, axis=1)
        min1 = abs_vals[np.arange(m), sorted_idx[:, 0]][:, None]
        min2 = abs_vals[np.arange(m), sorted_idx[:, 1]][:, None]
        is_min = (np.arange(d_c)[None, :] == sorted_idx[:, 0][:, None])
        leave_out_min = np.where(is_min, min2, min1)

        msg_c2v[check_edges] = leave_out_sign * leave_out_min

        vals_v = msg_c2v[var_edges]                  # (n, d_v)
        row_sum = vals_v.sum(axis=1) + channel_llr
        msg_v2c[var_edges] = row_sum[:, None] - vals_v

        total_llr = channel_llr + vals_v.sum(axis=1)
        d_hat = (total_llr < 0).astype(int)
        if np.all((H @ d_hat) % 2 == 0):
            return d_hat, iteration + 1

    return d_hat, maxiter


def ldpc_decode(soft_symbols, H, G, perm, snr_db=None, maxiter=50):
    """Full decode: soft BPSK symbols -> message bits. If snr_db isn't
    given, estimates it blindly from the received symbols."""
    soft_symbols = np.asarray(soft_symbols, dtype=float)
    n = H.shape[1]
    if len(soft_symbols) != n:
        soft_symbols = soft_symbols[:n]
    if snr_db is None:
        snr_db = estimate_snr_db(soft_symbols)
    d_hat, iters = ldpc_decode_codeword(H, soft_symbols, snr_db, maxiter=maxiter)
    k = G.shape[1]
    message = message_from_codeword(d_hat, perm, k)
    return message.astype(int), snr_db


if __name__ == "__main__":
    n, d_v, d_c, seed = 400, 3, 4, 11
    H, G, perm = build_ldpc_code(n, d_v, d_c, seed)
    k = G.shape[1]
    print("n =", n, "k =", k, "d_v =", d_v, "d_c =", d_c)
    assert np.all((H @ G) % 2 == 0), "self-test failed: G is not in H's null space"

    rng = np.random.default_rng(3)
    message = rng.integers(0, 2, k)
    codeword = ldpc_encode(message, G)
    assert np.array_equal(message_from_codeword(codeword, perm, k), message), \
        "self-test failed: message_from_codeword did not invert ldpc_encode"

    for noise_std in (0.3, 0.4, 0.5):
        symbols = (-1.0) ** codeword
        received = symbols + noise_std * rng.standard_normal(n)
        decoded, snr_db = ldpc_decode(received, H, G, perm)
        errors = int(np.sum(decoded != message))
        print(f"noise_std={noise_std}  est_snr_db={snr_db:.2f}  message_errors={errors}/{k}")
        assert errors == 0, "self-test failed: LDPC decode had bit errors at noise_std=" + str(noise_std)
    print("fec_ldpc self-test OK (pure NumPy, no pyldpc/numba)")
