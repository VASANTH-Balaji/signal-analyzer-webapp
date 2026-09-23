import json
import numpy as np
from fec_core import viterbi_decode, block_deinterleave
from signal_io import load_iq
from analysis_pipeline import (estimate_carrier_squaring as estimate_carrier_precise,
                               downconvert, demodulate_bpsk)

IQ_FILE = "test_signal_fec.iq"
META_FILE = "test_signal_fec_meta.json"


def main():
    with open(META_FILE) as f:
        meta = json.load(f)

    sample_rate = meta["sample_rate"]
    samples_per_symbol = meta["samples_per_symbol"]
    rows = meta["interleaver_rows"]
    cols = meta["interleaver_cols"]
    pad = meta["interleaver_pad"]
    num_message_bits = meta["num_message_bits"]
    num_encoded_bits = meta["num_encoded_bits"]
    truth_bits = meta["ground_truth_message_bits"]

    iq = load_iq(IQ_FILE)
    carrier = estimate_carrier_precise(iq, sample_rate)
    baseband = downconvert(iq, sample_rate, carrier)
    raw_bits = demodulate_bpsk(baseband, samples_per_symbol)

    # Try both polarities (phase ambiguity) - keep whichever decodes with fewer errors
    best_result = None
    for label, bits_variant in [("normal", raw_bits), ("inverted", 1 - raw_bits)]:
        try:
            deinterleaved = block_deinterleave(bits_variant, rows, cols, pad)
            decoded = viterbi_decode(deinterleaved, num_message_bits)
            n = min(len(decoded), len(truth_bits))
            errors = np.sum(decoded[:n] != np.array(truth_bits[:n]))
            if best_result is None or errors < best_result[2]:
                best_result = (label, decoded, errors)
        except Exception:
            continue

    label, decoded, errors = best_result
    ber = errors / num_message_bits * 100

    print("Estimated carrier:", round(carrier, 2), "Hz")
    print("Raw demodulated bits (before decode):", len(raw_bits))
    print("Polarity used:", label)
    print("Decoded message bits:", len(decoded))
    print("Bit errors after de-interleave + Viterbi decode:", errors)
    print("Final BER:", round(ber, 2), "%")
    print()
    print("First 20 decoded bits:  ", list(decoded[:20]))
    print("First 20 ground truth:  ", truth_bits[:20])


if __name__ == "__main__":
    main()
    input("Press Enter to close...")