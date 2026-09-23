import json
import numpy as np
from fec_core import conv_deinterleave, rs_decode
from signal_io import load_iq
from analysis_pipeline import (estimate_carrier_squaring as estimate_carrier_precise,
                               downconvert, demodulate_bpsk, bits_to_bytes)

IQ_FILE = "test_signal_rs.iq"
META_FILE = "test_signal_rs_meta.json"


def main():
    with open(META_FILE) as f:
        meta = json.load(f)

    sample_rate = meta["sample_rate"]
    samples_per_symbol = meta["samples_per_symbol"]
    B = meta["interleave_branches"]
    D = meta["interleave_delay"]
    delay = meta["interleave_total_delay"]
    codeword_len_bytes = meta["rs_codeword_len"]
    nsym = meta["rs_nsym"]
    truth_bytes = meta["ground_truth_message_bytes"]

    iq = load_iq(IQ_FILE)
    carrier = estimate_carrier_precise(iq, sample_rate)
    baseband = downconvert(iq, sample_rate, carrier)
    raw_bits = demodulate_bpsk(baseband, samples_per_symbol)

    best = None
    for label, variant in [("normal", raw_bits), ("inverted", 1 - raw_bits)]:
        try:
            deint = conv_deinterleave(variant, B, D)
            codeword_bits = deint[delay:delay + codeword_len_bytes * 8]
            codeword_bytes = bits_to_bytes(codeword_bits)
            decoded = rs_decode(codeword_bytes, nsym)
            n = min(len(decoded), len(truth_bytes))
            errors = sum(1 for a, b in zip(decoded[:n], truth_bytes[:n]) if a != b)
            if best is None or errors < best[2]:
                best = (label, decoded, errors)
        except Exception as e:
            if best is None:
                best = (label, None, 999, str(e))
            continue

    label = best[0]
    decoded_bytes = best[1]
    errors = best[2]

    print("Estimated carrier:", round(carrier, 2), "Hz")
    print("Raw demodulated bits:", len(raw_bits))
    print("Polarity used:", label)

    if decoded_bytes is None:
        print("RS decoding failed on both polarities:", best[3] if len(best) > 3 else "")
        return

    ber = errors / len(truth_bytes) * 100
    print("Decoded message bytes:", len(decoded_bytes))
    print("Byte errors after de-interleave + RS decode:", errors, "/", len(truth_bytes))
    print("Final byte error rate:", round(ber, 2), "%")
    print()
    print("Decoded bytes:     ", decoded_bytes)
    print("Ground truth bytes:", truth_bytes)


if __name__ == "__main__":
    main()
    input("Press Enter to close...")