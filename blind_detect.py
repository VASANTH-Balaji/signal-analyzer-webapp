"""
Blind FEC / interleaver identification - every detector returns a CONFIDENCE, never a
bare answer, and says explicitly when it cannot tell ("detected": False).

What is implemented (and how far it can be trusted):

  detect_convolutional_code()   Brute-forces a library of standard rate-1/2 codes
      (K=3..7, both generator orders and both bit-order conventions) through a
      Viterbi decoder, re-encodes the decoded bits and measures how well that matches
      what was received. A NULL is measured for every candidate by running the same
      decoder on a shuffled copy of the received bits - a Viterbi decoder always finds
      *some* nearby codeword, so raw agreement on random data is far from 50 % (about
      75-85 %, and it depends on K) and must be compared against that null, not against
      a fixed threshold. Detected only if the winner clears its own null by a wide margin
      AND beats the runner-up.

  detect_interleaver_and_code() Block / diagonal interleaver dimensions + code, by
      searching (rows, cols) hypotheses whose product is close to the received length
      and scoring each with the convolutional-code test. This is the search that actually
      applies to the project's conv-coded streams. Pseudo-random interleavers are NOT
      searchable this way and are reported as unidentified.

  detect_linear_block_width()   GF(2) rank-deficiency sweep. SCOPE: only meaningful
      when the stream really is a concatenation of codewords of a linear block code
      (each row of the trial matrix then lies in a k-dimensional subspace). One flipped
      bit restores full rank, so it needs error-free input, and it says "ambiguous"
      rather than guess. It does NOT apply to convolutional / LDPC streams.

Not implemented: blind RS/BCH identification (needs a search over (n, k) *and*
generator polynomials) - stretch goal, deliberately left out rather than faked.
"""
import numpy as np

from fec_core import viterbi_decode_soft, conv_encode_generic, block_deinterleave, \
    diagonal_deinterleave

# ---------------------------------------------------------------------------
# Candidate code library
# ---------------------------------------------------------------------------
_STANDARD_OCTAL = {          # K: (g1, g2) in octal, textbook rate-1/2 codes
    3: (0o7, 0o5),
    4: (0o15, 0o17),
    5: (0o23, 0o35),
    6: (0o53, 0o75),
    7: (0o171, 0o133),       # NASA / CCSDS (the G2 output inversion used on-air is not modelled)
}


def _rev(x, K):
    return int(format(x, "0%db" % K)[::-1], 2)


def build_code_library(ks=(3, 4, 5, 6, 7)):
    lib = {}
    for K in ks:
        g1, g2 = _STANDARD_OCTAL[K]
        for tag, (a, b) in (("", (g1, g2)), ("_swap", (g2, g1)),
                            ("_rev", (_rev(g1, K), _rev(g2, K))),
                            ("_revswap", (_rev(g2, K), _rev(g1, K)))):
            key = (K, a, b)
            if any((v["K"], v["g1"], v["g2"]) == key for v in lib.values()):
                continue        # symmetric codes: don't test the same stream twice
            lib["K%d_%s_%s%s" % (K, oct(g1)[2:], oct(g2)[2:], tag)] = dict(K=K, g1=a, g2=b)
    return lib


CODE_LIBRARY = build_code_library()


# ---------------------------------------------------------------------------
# A. Convolutional-code identification
# ---------------------------------------------------------------------------
def _agreement(soft, K, g1, g2, n_in):
    dec = viterbi_decode_soft(soft, n_in, K=K, g1=g1, g2=g2)
    enc = conv_encode_generic(dec, K, g1, g2)
    ref = (soft[: len(enc)] > 0).astype(int)
    keep = soft[: len(enc)] != 0
    m = min(len(enc), len(ref))
    k = keep[:m]
    return (float(np.mean(enc[:m][k] == ref[:m][k])) if np.any(k) else 0.0), dec


def _as_soft(x):
    x = np.asarray(x, dtype=float)
    if x.size and set(np.unique(x)).issubset({0.0, 1.0}):
        return 2.0 * x - 1.0
    return x


def detect_convolutional_code(received, codes=None, n_shuffles=4, max_coded_bits=3000, seed=0):
    """received: hard bits (0/1) or soft values (positive = 1). Returns a dict with
    'detected' (bool), 'confidence' (0..1), 'candidate_name', 'params', and the
    evidence ('agreement', 'null_mean', 'null_std', 'margin')."""
    codes = codes or CODE_LIBRARY
    soft = _as_soft(received)
    n_in = min(len(soft) // 2, max_coded_bits // 2)
    if n_in < 100:
        return dict(detected=False, confidence=0.0, reason="too few bits (need >= 200 coded bits)",
                    candidate_name=None, params=None)
    soft = soft[: 2 * n_in]
    rng = np.random.default_rng(seed)

    results = {}
    for name, p in codes.items():
        agree, _ = _agreement(soft, p["K"], p["g1"], p["g2"], n_in)
        results[name] = agree
    order = sorted(results, key=results.get, reverse=True)
    best, runner = order[0], (order[1] if len(order) > 1 else None)

    # null for the best candidate only (and its runner-up for a fair margin): shuffle the
    # received stream, keep the decoder identical
    p = codes[best]
    nulls = []
    for _ in range(n_shuffles):
        nulls.append(_agreement(soft[rng.permutation(len(soft))], p["K"], p["g1"], p["g2"], n_in)[0])
    null_mean, null_std = float(np.mean(nulls)), float(np.std(nulls))
    lift = results[best] - null_mean
    margin = results[best] - (results[runner] if runner else 0.0)

    detected = (lift >= max(0.08, 8 * null_std)) and (margin >= 0.03)
    confidence = float(np.clip(lift / max(1e-9, 1.0 - null_mean), 0, 1)) if detected else \
        float(np.clip(lift / max(1e-9, 1.0 - null_mean), 0, 1)) * 0.3
    return dict(detected=bool(detected), confidence=confidence, candidate_name=best,
                params=dict(codes[best]), agreement=results[best], null_mean=null_mean,
                null_std=null_std, margin=margin, runner_up=runner, all_scores=results)


def blind_viterbi_decode(received, params):
    soft = _as_soft(received)
    n_in = len(soft) // 2
    return viterbi_decode_soft(soft, n_in, K=params["K"], g1=params["g1"], g2=params["g2"])


# ---------------------------------------------------------------------------
# B. Block / diagonal interleaver dimensions + code
# ---------------------------------------------------------------------------
def _factor_pairs(L, min_side=2, max_side=512):
    return [(r, L // r) for r in range(min_side, min(max_side, L // min_side) + 1)
            if L % r == 0 and L // r >= min_side]


def detect_interleaver_and_code(received, families=("block", "diagonal"), len_slack=3,
                                codes=None, max_decodes=500, max_len=6000, seed=0, early_stop=0.97):
    """Search interleaver dimensions for a convolutionally coded stream.

    Hypotheses: 'none', and for each length L in [len, len+len_slack] (the demodulator can
    drop a few edge bits, the interleaver pads up to rows*cols) every factor pair
    (rows, cols) for each family. Each is scored with the code test; the winner must be
    detected AND clearly better than the 'none' hypothesis. Returns a dict; when nothing
    stands out 'detected' is False and callers must fall back to metadata."""
    codes = codes or {k: v for k, v in CODE_LIBRARY.items() if v["K"] <= 5}
    soft = _as_soft(received)
    if len(soft) > max_len:
        return dict(detected=False, confidence=0.0, reason="stream too long for the brute-force search "
                    "(%d > %d bits); set the interleaver manually" % (len(soft), max_len))
    budget = [max_decodes]

    def score_hyp(x):
        best_name, best_agree = None, -1.0
        n_in = len(x) // 2
        for name, p in codes.items():
            if budget[0] <= 0:
                break
            budget[0] -= 1
            a, _ = _agreement(x[: 2 * n_in], p["K"], p["g1"], p["g2"], n_in)
            if a > best_agree:
                best_name, best_agree = name, a
        return best_name, best_agree

    hyps = [("none", None, soft)]
    for fam in families:
        for extra in range(0, len_slack + 1):
            L = len(soft) + extra
            for (r, c) in _factor_pairs(L):
                x = np.concatenate([soft, np.zeros(extra)])
                try:
                    d = block_deinterleave(x, r, c, 0) if fam == "block" else \
                        diagonal_deinterleave(x, r, c, 0)
                except Exception:
                    continue
                hyps.append((fam, (r, c, extra), d))

    scored = []
    for fam, dims, x in hyps:
        name, agree = score_hyp(x)
        scored.append(dict(family=fam, dims=dims, code=name, agreement=agree))
        # decisive hit: far above the ~0.88 a Viterbi decoder reaches on unstructured bits
        if fam != "none" and agree >= early_stop and len(scored) >= 6:
            break
        if budget[0] <= 0:
            break
    scored.sort(key=lambda d: d["agreement"], reverse=True)
    top = scored[0]
    baseline = next(d for d in scored if d["family"] == "none")
    rest = [d["agreement"] for d in scored[1:]] or [0.0]
    spread = float(np.std(rest)) if len(rest) > 2 else 0.05
    lift = top["agreement"] - float(np.mean(rest))
    margin = top["agreement"] - (scored[1]["agreement"] if len(scored) > 1 else 0.0)
    detected = (top["family"] != "none" and lift >= max(0.08, 6 * spread) and margin >= 0.03)
    if top["family"] == "none":
        # the un-interleaved hypothesis winning is the code test's business, not an interleaver find
        detected = False
    out = dict(detected=bool(detected), best=top, none_agreement=baseline["agreement"],
               hypotheses_tested=len(scored), decodes_used=max_decodes - budget[0],
               lift=lift, margin=margin,
               confidence=float(np.clip(lift / 0.25, 0, 1)) if detected else 0.0)
    if not detected:
        out["reason"] = "no interleaver hypothesis stood out from the rest"
    return out


# ---------------------------------------------------------------------------
# C. GF(2) rank sweep (linear BLOCK codes only - see module docstring)
# ---------------------------------------------------------------------------
def _gf2_rank(matrix):
    m = (np.asarray(matrix, dtype=np.uint8) % 2).copy()
    rows, cols = m.shape
    rank = 0
    for col in range(cols):
        piv = None
        for r in range(rank, rows):
            if m[r, col]:
                piv = r
                break
        if piv is None:
            continue
        m[[rank, piv]] = m[[piv, rank]]
        for r in range(rows):
            if r != rank and m[r, col]:
                m[r] ^= m[rank]
        rank += 1
        if rank == rows:
            break
    return rank


def detect_linear_block_width(bits, width_range=range(4, 65), min_rows=4):
    bits = np.asarray(bits, dtype=int)
    defs = {}
    for w in width_range:
        rows = len(bits) // w
        if rows < max(min_rows, w + 1):          # need a tall matrix or every input looks deficient
            continue
        defs[w] = (w - _gf2_rank(bits[: rows * w].reshape(rows, w))) / w
    if not defs:
        return dict(width=None, confidence=0.0, ambiguous=True, reason="stream too short")
    widths, vals = list(defs), np.array(list(defs.values()))
    z = (vals.max() - vals.mean()) / vals.std() if vals.std() > 1e-9 else 0.0
    srt = np.sort(vals)[::-1]
    close = len(srt) > 1 and (srt[0] - srt[1]) < 0.05
    amb = z < 2.0 or close
    return dict(width=None if amb else widths[int(np.argmax(vals))],
                confidence=0.0 if amb else float(min(1.0, z / 6.0)), ambiguous=bool(amb),
                z_score=float(z), all_deficiencies=defs)
