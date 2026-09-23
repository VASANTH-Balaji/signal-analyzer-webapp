import json
import numpy as np
from signal_io import load_iq
from analysis_pipeline import (estimate_carrier_squaring as estimate_carrier_precise,
                               downconvert, demodulate_bpsk)
from fec_core import viterbi_decode, block_deinterleave

IQ_FILE = "test_signal_framed.iq"
META_FILE = "test_signal_framed_meta.json"


def find_sync_word(bitstream, sync_word):
    # Slide the sync word across the bitstream and score each position by
    # how many bits match. The position with the fewest mismatches is the
    # most likely start of the frame - this is bitstream correlation.
    bits_pm1 = 2 * bitstream.astype(int) - 1        # convert 0/1 to -1/+1
    sync_pm1 = 2 * np.array(sync_word) - 1

    n = len(sync_word)
    best_pos = 0
    best_score = -1e9
    scores = []
    for pos in range(len(bitstream) - n + 1):
        window = bits_pm1[pos:pos + n]
        score = np.dot(window, sync_pm1)  # correlation score
        scores.append(score)
        if score > best_score:
            best_score = score
            best_pos = pos
    return best_pos, best_score, n, scores


def main():
    with open(META_FILE) as f:
        meta = json.load(f)

    sample_rate = meta["sample_rate"]
    samples_per_symbol = meta["samples_per_symbol"]
    rows = meta["interleaver_rows"]
    cols = meta["interleaver_cols"]
    pad = meta["interleaver_pad"]
    num_total_bits = meta["num_total_bits"]
    sync_word = meta["sync_word"]
    payload_len = meta["payload_len"]
    truth_payload = meta["ground_truth_payload_bits"]

    iq = load_iq(IQ_FILE)
    carrier = estimate_carrier_precise(iq, sample_rate)
    baseband = downconvert(iq, sample_rate, carrier)
    raw_bits = demodulate_bpsk(baseband, samples_per_symbol)

    # Resolve phase ambiguity by trying both polarities and keeping the one
    # whose sync-word correlation is strongest
    best = None
    for label, bits_variant in [("normal", raw_bits), ("inverted", 1 - raw_bits)]:
        try:
            deinterleaved = block_deinterleave(bits_variant, rows, cols, pad)
            decoded = viterbi_decode(deinterleaved, num_total_bits)
            pos, score, sync_len, _ = find_sync_word(decoded, sync_word)
            if best is None or score > best[4]:
                best = (label, decoded, pos, sync_len, score)
        except Exception:
            continue

    label, decoded_bits, sync_pos, sync_len, score = best

    header_start = sync_pos
    header_end = sync_pos + sync_len
    payload_start = header_end
    payload_end = payload_start + payload_len

    recovered_preamble_junk = decoded_bits[:header_start]
    recovered_sync = decoded_bits[header_start:header_end]
    recovered_payload = decoded_bits[payload_start:payload_end]

    n = min(len(recovered_payload), len(truth_payload))
    errors = np.sum(recovered_payload[:n] != np.array(truth_payload[:n]))
    ber = errors / n * 100 if n else 100

    print("Estimated carrier:", round(carrier, 2), "Hz")
    print("Polarity used:", label)
    print("Total decoded bits:", len(decoded_bits))
    print()
    print("--- Bitstream correlation result ---")
    print("Sync word found at bit position:", sync_pos, "(correlation score:", int(score), "/", sync_len, "max)")
    print("Junk/preamble bits before frame:", header_start)
    print("Header (sync word) region: bits", header_start, "to", header_end)
    print("Payload region: bits", payload_start, "to", payload_end)
    print()
    print("Recovered sync word:  ", list(recovered_sync))
    print("Expected sync word:   ", sync_word)
    print()
    print("Payload bit errors:", errors, "out of", n)
    print("Payload BER:", round(ber, 2), "%")
    print("First 20 recovered payload bits:", list(recovered_payload[:20]))
    print("First 20 ground truth payload:  ", truth_payload[:20])


if __name__ == "__main__":
    main()
    input("Press Enter to close...")