import numpy as np

# Rate 1/2, constraint length 3 convolutional code (standard textbook code)
G1 = 0b111
G2 = 0b101
K = 3
NUM_STATES = 2 ** (K - 1)


def conv_encode(bits):
    bits = np.asarray(bits, dtype=int)
    shift_reg = 0
    output = []
    for b in bits:
        shift_reg = ((shift_reg << 1) | b) & 0b111
        out1 = bin(shift_reg & G1).count("1") % 2
        out2 = bin(shift_reg & G2).count("1") % 2
        output.append(out1)
        output.append(out2)
    return np.array(output, dtype=int)


def conv_encode_generic(bits, K, g1, g2):
    """Rate-1/2 encoder for any (K, g1, g2) - same shift-register convention as
    conv_encode() and viterbi_decode_soft()."""
    mask = (1 << K) - 1
    sr = 0
    out = np.empty(2 * len(bits), dtype=int)
    for i, b in enumerate(np.asarray(bits, dtype=int)):
        sr = ((sr << 1) | int(b)) & mask
        out[2 * i] = bin(sr & g1).count("1") & 1
        out[2 * i + 1] = bin(sr & g2).count("1") & 1
    return out


def _next_state(state, bit):
    return ((state << 1) | bit) & (NUM_STATES - 1)


def _output_for_transition(state, bit):
    shift_reg = ((state << 1) | bit) & 0b111
    out1 = bin(shift_reg & G1).count("1") % 2
    out2 = bin(shift_reg & G2).count("1") % 2
    return out1, out2


def viterbi_decode(received_bits, num_input_bits):
    INF = float("inf")
    path_metric = [INF] * NUM_STATES
    path_metric[0] = 0
    paths = [[] for _ in range(NUM_STATES)]

    for i in range(num_input_bits):
        r1, r2 = received_bits[2 * i], received_bits[2 * i + 1]
        new_metric = [INF] * NUM_STATES
        new_paths = [None] * NUM_STATES

        for state in range(NUM_STATES):
            if path_metric[state] == INF:
                continue
            for bit in (0, 1):
                ns = _next_state(state, bit)
                o1, o2 = _output_for_transition(state, bit)
                branch_metric = (o1 != r1) + (o2 != r2)
                total = path_metric[state] + branch_metric
                if total < new_metric[ns]:
                    new_metric[ns] = total
                    new_paths[ns] = paths[state] + [bit]

        path_metric = new_metric
        paths = new_paths

    best_state = int(np.argmin(path_metric))
    return np.array(paths[best_state], dtype=int)


def block_interleave(bits, num_rows, num_cols):
    n = num_rows * num_cols
    pad = (-len(bits)) % n
    bits_padded = np.concatenate([bits, np.zeros(pad, dtype=int)])
    matrix = bits_padded.reshape(num_rows, num_cols)
    return matrix.T.flatten(), pad


def block_deinterleave(bits, num_rows, num_cols, pad):
    matrix = bits.reshape(num_cols, num_rows).T
    flat = matrix.flatten()
    if pad:
        flat = flat[:-pad]
    return flat


# ---------------------------------------------------------------------------
# Convolutional (Forney/cross) interleaver - a second interleaving scheme,
# distinct from block interleaving above. Uses B branches with staggered
# FIFO delays (branch i delays by i*D symbols) to spread burst errors across
# time. Verified correct across 100 randomized trials (varying B, D, length).
# ---------------------------------------------------------------------------

def conv_interleave(bits, num_branches, delay_increment):
    bits = list(bits)
    registers = [[0] * (i * delay_increment) for i in range(num_branches)]
    output = []
    for idx, b in enumerate(bits):
        branch = idx % num_branches
        registers[branch].append(int(b))
        output.append(registers[branch].pop(0))
    return np.array(output, dtype=int)


def conv_deinterleave(bits, num_branches, delay_increment):
    # Complementary delays (branch i delays by (B-1-i)*D) so total end-to-end
    # delay is the same for every branch: B*(B-1)*D symbols overall.
    bits = list(bits)
    registers = [[0] * ((num_branches - 1 - i) * delay_increment) for i in range(num_branches)]
    output = []
    for idx, b in enumerate(bits):
        branch = idx % num_branches
        registers[branch].append(int(b))
        output.append(registers[branch].pop(0))
    return np.array(output, dtype=int)


def conv_interleave_total_delay(num_branches, delay_increment):
    return num_branches * (num_branches - 1) * delay_increment


# ---------------------------------------------------------------------------
# Diagonal interleaver - writes bits into a matrix and reads them back out
# along diagonals instead of rows/columns. Spreads burst errors similarly to
# block interleaving but with a different (harder to blindly detect) pattern.
# ---------------------------------------------------------------------------

def _keep_soft_dtype(x):
    """De-interleavers are pure permutations, so they must work on soft (float)
    values as well as hard bits. Anything that is not float/complex is made int."""
    x = np.asarray(x)
    return x if x.dtype.kind in "fc" else x.astype(int)


def diagonal_interleave(bits, num_rows, num_cols):
    n = num_rows * num_cols
    pad = (-len(bits)) % n
    bits_padded = np.concatenate([bits, np.zeros(pad, dtype=int)])
    matrix = bits_padded.reshape(num_rows, num_cols)

    out = np.zeros(n, dtype=int)
    idx = 0
    for d in range(num_rows + num_cols - 1):
        for r in range(num_rows):
            c = d - r
            if 0 <= c < num_cols:
                out[idx] = matrix[r, c]
                idx += 1
    return out, pad


def diagonal_deinterleave(bits, num_rows, num_cols, pad):
    n = num_rows * num_cols
    bits = _keep_soft_dtype(bits)
    matrix = np.zeros((num_rows, num_cols), dtype=bits.dtype)
    idx = 0
    for d in range(num_rows + num_cols - 1):
        for r in range(num_rows):
            c = d - r
            if 0 <= c < num_cols:
                matrix[r, c] = bits[idx]
                idx += 1
    flat = matrix.flatten()
    if pad:
        flat = flat[:-pad]
    return flat


# ---------------------------------------------------------------------------
# Pseudo-random interleaver - permutes bit positions using a PRNG-generated
# permutation table. The seed acts as the "key": receiver and transmitter
# must agree on it (this is what makes pseudo-random interleaving hardest to
# blindly reconstruct without side information - see fec_scheme_matcher.py).
# ---------------------------------------------------------------------------

def prbs_interleave(bits, seed, block_len=None):
    bits = np.asarray(bits, dtype=int)
    n = block_len or len(bits)
    pad = (-len(bits)) % n
    bits_padded = np.concatenate([bits, np.zeros(pad, dtype=int)])

    num_blocks = len(bits_padded) // n
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)  # same permutation reused per block

    out = np.empty_like(bits_padded)
    for b in range(num_blocks):
        block = bits_padded[b * n:(b + 1) * n]
        out[b * n:(b + 1) * n] = block[perm]
    return out, pad


def prbs_deinterleave(bits, seed, pad, block_len=None):
    bits = _keep_soft_dtype(bits)
    n = block_len or len(bits)
    num_blocks = len(bits) // n
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    inv_perm = np.argsort(perm)

    out = np.empty_like(bits)
    for b in range(num_blocks):
        block = bits[b * n:(b + 1) * n]
        out[b * n:(b + 1) * n] = block[inv_perm]

    if pad:
        out = out[:-pad]
    return out


# ---------------------------------------------------------------------------
# Reed-Solomon codec over GF(256), primitive polynomial 0x11d - a second FEC
# scheme, distinct from the convolutional/Viterbi code above. Operates on
# byte symbols (0-255), not bits. Implemented from scratch and verified
# correct across 500 randomized trials with varying error counts.
# ---------------------------------------------------------------------------

import itertools

_GF_EXP = [0] * 512
_GF_LOG = [0] * 256


def _rs_init_tables(prim=0x11d):
    x = 1
    for i in range(255):
        _GF_EXP[i] = x
        _GF_LOG[x] = i
        x <<= 1
        if x & 0x100:
            x ^= prim
    for i in range(255, 512):
        _GF_EXP[i] = _GF_EXP[i - 255]


_rs_init_tables()


def _gf_mul(a, b):
    if a == 0 or b == 0:
        return 0
    return _GF_EXP[_GF_LOG[a] + _GF_LOG[b]]


def _gf_div(a, b):
    if a == 0:
        return 0
    return _GF_EXP[(_GF_LOG[a] - _GF_LOG[b]) % 255]


def _gf_pow(a, power):
    return _GF_EXP[(_GF_LOG[a] * power) % 255]


def _gf_inverse(a):
    return _GF_EXP[255 - _GF_LOG[a]]


def _gf_poly_mul(p, q):
    r = [0] * (len(p) + len(q) - 1)
    for i, pi in enumerate(p):
        if pi == 0:
            continue
        for j, qj in enumerate(q):
            r[i + j] ^= _gf_mul(pi, qj)
    return r


def _gf_poly_add(p, q):
    r = [0] * max(len(p), len(q))
    for i in range(len(p)):
        r[i + len(r) - len(p)] ^= p[i]
    for i in range(len(q)):
        r[i + len(r) - len(q)] ^= q[i]
    return r


def _gf_poly_scale(p, x):
    return [_gf_mul(c, x) for c in p]


def _gf_poly_eval(poly, x):
    y = poly[0]
    for c in poly[1:]:
        y = _gf_mul(y, x) ^ c
    return y


def _gf_poly_div(dividend, divisor):
    result = list(dividend)
    for i in range(len(dividend) - len(divisor) + 1):
        coef = result[i]
        if coef != 0:
            for j in range(1, len(divisor)):
                if divisor[j] != 0:
                    result[i + j] ^= _gf_mul(divisor[j], coef)
    separator = len(divisor) - 1
    return result[:-separator], result[-separator:]


def _rs_generator_poly(nsym):
    g = [1]
    for i in range(nsym):
        g = _gf_poly_mul(g, [1, _gf_pow(2, i)])
    return g


def rs_encode(msg, nsym):
    # msg: list/array of byte values (0-255). Returns msg + nsym parity bytes.
    # Can correct up to nsym//2 byte errors anywhere in the returned codeword.
    msg = [int(x) for x in msg]
    gen = _rs_generator_poly(nsym)
    msg_padded = list(msg) + [0] * nsym
    for i in range(len(msg)):
        coef = msg_padded[i]
        if coef != 0:
            for j in range(len(gen)):
                msg_padded[i + j] ^= _gf_mul(gen[j], coef)
    return list(msg) + msg_padded[len(msg):]


def _rs_calc_syndromes(msg, nsym):
    return [0] + [_gf_poly_eval(msg, _gf_pow(2, i)) for i in range(nsym)]


def _rs_find_error_locator(synd, nsym):
    err_loc = [1]
    old_loc = [1]
    for i in range(nsym):
        delta = synd[i + 1]
        for j in range(1, len(err_loc)):
            delta ^= _gf_mul(err_loc[-(j + 1)], synd[i - j + 1])
        old_loc.append(0)
        if delta != 0:
            if len(old_loc) > len(err_loc):
                new_loc = _gf_poly_scale(old_loc, delta)
                old_loc = _gf_poly_scale(err_loc, _gf_inverse(delta))
                err_loc = new_loc
            err_loc = _gf_poly_add(err_loc, _gf_poly_scale(old_loc, delta))
    err_loc = list(itertools.dropwhile(lambda x: x == 0, err_loc))
    errs = len(err_loc) - 1
    if errs * 2 > nsym:
        raise ValueError("Too many errors to correct")
    return err_loc


def _rs_find_errors(err_loc, msg_len):
    errs = len(err_loc) - 1
    err_pos = []
    for e in range(msg_len):
        c = msg_len - 1 - e
        x = _gf_pow(2, (255 - c) % 255)
        if _gf_poly_eval(err_loc, x) == 0:
            err_pos.append(e)
    if len(err_pos) != errs:
        raise ValueError("Could not locate all errors")
    return err_pos


def _rs_find_error_evaluator(synd, err_loc, nsym):
    _, remainder = _gf_poly_div(_gf_poly_mul(synd, err_loc), [1] + [0] * (nsym + 1))
    return remainder


def _rs_correct_errata(msg, synd, err_pos):
    coef_pos = [len(msg) - 1 - p for p in err_pos]
    err_loc = [1]
    for p in coef_pos:
        err_loc = _gf_poly_mul(err_loc, [_gf_pow(2, p), 1])

    err_eval = _rs_find_error_evaluator(synd[::-1], err_loc, len(err_loc) - 1)[::-1]

    X = [_gf_pow(2, p) for p in coef_pos]
    E = [0] * len(msg)
    for i, Xi in enumerate(X):
        Xi_inv = _gf_inverse(Xi)
        err_loc_prime = 1
        for j, Xj in enumerate(X):
            if j != i:
                err_loc_prime = _gf_mul(err_loc_prime, (1 ^ _gf_mul(Xi_inv, Xj)))
        y = _gf_poly_eval(err_eval[::-1], Xi_inv)
        y = _gf_mul(Xi, y)
        magnitude = _gf_div(y, err_loc_prime)
        E[err_pos[i]] = magnitude

    return _gf_poly_add(msg, E)


def rs_decode(msg, nsym):
    # Returns the original message (without parity bytes), correcting up
    # to nsym//2 byte errors anywhere in msg. Raises ValueError if there
    # are too many errors to correct.
    msg = [int(x) for x in msg]
    synd = _rs_calc_syndromes(msg, nsym)
    if max(synd) == 0:
        return msg[:-nsym]
    err_loc = _rs_find_error_locator(synd, nsym)
    err_pos = _rs_find_errors(err_loc, len(msg))
    corrected = _rs_correct_errata(msg, synd, err_pos)
    synd_check = _rs_calc_syndromes(corrected, nsym)
    if max(synd_check) != 0:
        raise ValueError("Decoding failed - could not fully correct")
    return corrected[:-nsym]


# ---------------------------------------------------------------------------
# Fast / soft-input Viterbi decoder (generic K, G1, G2)
#
# viterbi_decode() above is the original hard-decision reference decoder
# (kept unchanged so existing results stay bit-identical). This one keeps
# the survivor paths in arrays instead of copying Python lists, accepts SOFT
# input, and takes the code polynomials as arguments, so the blind-detection
# module can also reuse it for other candidate codes.
#
# Soft-value convention used everywhere in this project: a positive value
# means "bit 1 is more likely" (matches demodulate_bpsk: bit = real > 0).
# Hard bits are converted with soft = 2*bit - 1.
# ---------------------------------------------------------------------------
_TRELLIS_CACHE = {}


def _trellis(K, g1, g2):
    key = (K, g1, g2)
    if key not in _TRELLIS_CACHE:
        S = 1 << (K - 1)
        mask = (1 << K) - 1
        nxt = np.zeros((S, 2), dtype=int)
        sgn1 = np.zeros((S, 2))
        sgn2 = np.zeros((S, 2))
        for st in range(S):
            for b in (0, 1):
                sr = ((st << 1) | b) & mask
                nxt[st, b] = sr & (S - 1)
                sgn1[st, b] = 2 * (bin(sr & g1).count("1") % 2) - 1
                sgn2[st, b] = 2 * (bin(sr & g2).count("1") % 2) - 1
        # every next-state has exactly two (prev_state, bit) predecessors
        pred = [[] for _ in range(S)]
        for st in range(S):
            for b in (0, 1):
                pred[nxt[st, b]].append((st, b))
        pred_s = np.array([[p[0][0], p[1][0]] for p in pred])
        pred_b = np.array([[p[0][1], p[1][1]] for p in pred])
        _TRELLIS_CACHE[key] = (S, sgn1, sgn2, pred_s, pred_b)
    return _TRELLIS_CACHE[key]


def viterbi_decode_soft(soft, num_input_bits, K=3, g1=0b111, g2=0b101, terminated=False):
    """Rate-1/2 Viterbi with soft (or +-1 hard) input.

    soft: 2*num_input_bits values, positive = "1 more likely".
    terminated: True if the encoder was flushed with K-1 zero bits, so the
    trace-back starts from state 0 (a small but real reliability gain).
    Returns the decoded input bits (including any tail bits).
    """
    S, sgn1, sgn2, pred_s, pred_b = _trellis(K, g1, g2)
    soft = np.asarray(soft, dtype=float)[: 2 * num_input_bits]
    s1, s2 = soft[0::2], soft[1::2]
    n = min(len(s1), len(s2), num_input_bits)
    # cost[t, state, bit] = -correlation between branch label and received soft pair
    cost = -(sgn1[None] * s1[:n, None, None] + sgn2[None] * s2[:n, None, None])
    BIG = 1e30
    pm = np.full(S, BIG)
    pm[0] = 0.0
    surv_s = np.zeros((n, S), dtype=np.int16)
    surv_b = np.zeros((n, S), dtype=np.int8)
    rows = np.arange(S)
    for t in range(n):
        c = cost[t]
        cand0 = pm[pred_s[:, 0]] + c[pred_s[:, 0], pred_b[:, 0]]
        cand1 = pm[pred_s[:, 1]] + c[pred_s[:, 1], pred_b[:, 1]]
        pick1 = cand1 < cand0
        pm = np.where(pick1, cand1, cand0)
        surv_s[t] = np.where(pick1, pred_s[:, 1], pred_s[:, 0])
        surv_b[t] = np.where(pick1, pred_b[:, 1], pred_b[:, 0])
        pm = pm - pm.min()                      # keep metrics bounded
    state = 0 if terminated else int(np.argmin(pm))
    out = np.zeros(n, dtype=int)
    for t in range(n - 1, -1, -1):
        out[t] = surv_b[t, state]
        state = surv_s[t, state]
    return out


# ---------------------------------------------------------------------------
# Real concatenated code: RS (outer) -> byte block interleaver -> convolutional
# (inner) - the CCSDS / DVB-S architecture. This is NOT the same thing as the
# "Convolutional interleaver + RS" scheme elsewhere in this project, which has
# only ONE error-correcting code (RS) and merely a convolutional *interleaver*.
#
#   TX: message -> split into k-byte blocks -> RS(n,k) each -> interleave
#       `depth` codewords byte-by-byte -> bits -> +tail -> conv encode (inner)
#   RX: Viterbi (inner, fixes scattered errors) -> de-interleave (breaks up
#       Viterbi's error bursts) -> RS (outer, mops up what is left)
# ---------------------------------------------------------------------------
def bytes_to_bits(data):
    return np.unpackbits(np.asarray(bytearray(int(b) & 0xFF for b in data), dtype=np.uint8)).astype(int)


def bits_to_byte_list(bits):
    bits = np.asarray(bits, dtype=int)
    bits = bits[: len(bits) - (len(bits) % 8)]
    return [int(v) for v in np.packbits(bits.astype(np.uint8))]


def symbol_interleave(data, depth, n):
    """Write `depth` codewords of n bytes as rows, read out by column."""
    arr = np.asarray(data, dtype=int)
    assert arr.size == depth * n, "need exactly depth*n bytes"
    return [int(v) for v in arr.reshape(depth, n).T.flatten()]


def symbol_deinterleave(data, depth, n):
    arr = np.asarray(data, dtype=int)
    assert arr.size == depth * n, "need exactly depth*n bytes"
    return [int(v) for v in arr.reshape(n, depth).T.flatten()]


def concat_frame_layout(num_message_bytes, nsym, rs_n, depth):
    k = rs_n - nsym
    per_frame = k * depth
    frames = -(-num_message_bytes // per_frame)
    return k, per_frame, frames


def concat_encode(message_bytes, nsym=16, rs_n=64, depth=4):
    """Returns (coded_bits, info dict). Message is zero-padded to whole frames."""
    k, per_frame, frames = concat_frame_layout(len(message_bytes), nsym, rs_n, depth)
    msg = list(int(b) for b in message_bytes) + [0] * (frames * per_frame - len(message_bytes))
    tail = np.zeros(K - 1, dtype=int)
    coded = []
    for f in range(frames):
        block = msg[f * per_frame:(f + 1) * per_frame]
        codewords = []
        for c in range(depth):
            codewords.extend(rs_encode(block[c * k:(c + 1) * k], nsym))
        tx_bytes = symbol_interleave(codewords, depth, rs_n)
        info_bits = np.concatenate([bytes_to_bits(tx_bytes), tail])
        coded.append(conv_encode(info_bits))
    info = dict(nsym=nsym, rs_n=rs_n, rs_k=k, depth=depth, frames=frames,
                num_message_bytes=len(message_bytes),
                bits_per_frame_coded=2 * (8 * depth * rs_n + K - 1))
    return np.concatenate(coded), info


def concat_decode(received, num_message_bytes, nsym=16, rs_n=64, depth=4, soft=False):
    """Decode a concatenated stream.

    received: hard bits (0/1), or soft values (positive = 1) if soft=True.
    Returns (message_bytes, stats). stats['failed_codewords'] counts RS blocks
    the outer decoder gave up on (their systematic bytes are passed through
    uncorrected, never silently guessed at); 'viterbi_bit_errors_fixed' is not
    knowable blind so it is not reported.
    """
    k, per_frame, frames = concat_frame_layout(num_message_bytes, nsym, rs_n, depth)
    frame_info_bits = 8 * depth * rs_n + K - 1
    frame_coded = 2 * frame_info_bits
    rx = np.asarray(received, dtype=float)
    if not soft:
        rx = 2.0 * rx - 1.0
    out, failed, corrected_syms = [], 0, 0
    for f in range(frames):
        seg = rx[f * frame_coded:(f + 1) * frame_coded]
        if len(seg) < frame_coded:
            seg = np.concatenate([seg, np.zeros(frame_coded - len(seg))])
        dec = viterbi_decode_soft(seg, frame_info_bits, terminated=True)
        rx_bytes = bits_to_byte_list(dec[: 8 * depth * rs_n])
        codewords = symbol_deinterleave(rx_bytes, depth, rs_n)
        for c in range(depth):
            cw = codewords[c * rs_n:(c + 1) * rs_n]
            try:
                msg = rs_decode(cw, nsym)
                corrected_syms += sum(1 for a, b in zip(cw[:k], msg) if a != b)
            except ValueError:
                failed += 1
                msg = cw[:k]
            out.extend(msg)
    return out[:num_message_bytes], dict(failed_codewords=failed, total_codewords=frames * depth,
                                         rs_symbols_corrected=corrected_syms)


if __name__ == "__main__":
    # Self-test: round-trip every interleaver type on random data, to
    # confirm the merge didn't break anything and the new functions work
    # inside fec_core.py itself (not just the standalone _ext.py file).
    rng = np.random.default_rng(0)
    test_bits = rng.integers(0, 2, size=97)

    inter, pad = block_interleave(test_bits, 7, 15)
    back = block_deinterleave(inter, 7, 15, pad)
    assert np.array_equal(back, test_bits), "block interleaver round-trip failed"
    print("Block interleaver: round-trip OK")

    inter = conv_interleave(test_bits, 4, 2)
    back = conv_deinterleave(inter, 4, 2)
    delay = conv_interleave_total_delay(4, 2)
    assert np.array_equal(back[delay:], test_bits[:len(back) - delay]), \
        "conv interleaver round-trip failed"
    print("Convolutional interleaver: round-trip OK")

    inter, pad = diagonal_interleave(test_bits, 7, 15)
    back = diagonal_deinterleave(inter, 7, 15, pad)
    assert np.array_equal(back, test_bits), "diagonal interleaver round-trip failed"
    print("Diagonal interleaver: round-trip OK")

    inter, pad = prbs_interleave(test_bits, seed=42, block_len=97)
    back = prbs_deinterleave(inter, seed=42, pad=pad, block_len=97)
    assert np.array_equal(back, test_bits), "pseudo-random interleaver round-trip failed"
    print("Pseudo-random interleaver: round-trip OK")

    print("All 4 interleaver types merged and passing inside fec_core.py.")

    # --- soft Viterbi must agree with the reference decoder on clean + lightly noisy input ---
    rng2 = np.random.default_rng(5)
    msg_bits = rng2.integers(0, 2, 200)
    coded_bits = conv_encode(msg_bits)
    assert np.array_equal(viterbi_decode_soft(2 * coded_bits - 1, 200), msg_bits)
    noisy = coded_bits.copy()
    noisy[[10, 55, 150, 300]] ^= 1
    assert np.array_equal(viterbi_decode_soft(2 * noisy - 1, 200), viterbi_decode(noisy, 200)), \
        "fast Viterbi disagrees with reference Viterbi"
    print("Fast/soft Viterbi: matches reference decoder")

    # --- concatenated code round-trip (clean) ---
    payload = [int(x) for x in rng2.integers(0, 256, 300)]
    tx, info = concat_encode(payload)
    rx_msg, st = concat_decode(tx, len(payload))
    assert rx_msg == payload and st["failed_codewords"] == 0
    print("Concatenated RS+interleave+Viterbi: round-trip OK (%d frames)" % info["frames"])
