import json
import numpy as np
from signal_io import load_iq
from analysis_pipeline import (estimate_carrier_squaring as estimate_carrier_precise,
                               downconvert, demodulate_bpsk)

IQ_FILE = "test_signal.iq"
META_FILE = "test_signal_meta.json"


def bit_error_rate(recovered, truth):
    n = min(len(recovered), len(truth))
    recovered = recovered[:n]
    truth = np.array(truth[:n])

    errors_normal = np.sum(recovered != truth)
    errors_inverted = np.sum((1 - recovered) != truth)

    if errors_inverted < errors_normal:
        return errors_inverted / n, "inverted", 1 - recovered
    else:
        return errors_normal / n, "normal", recovered


def main():
    with open(META_FILE) as f:
        meta = json.load(f)
    sample_rate = meta["sample_rate"]
    samples_per_symbol = meta["samples_per_symbol"]
    truth_bits = meta["ground_truth_bits"]

    iq = load_iq(IQ_FILE)
    carrier = estimate_carrier_precise(iq, sample_rate)
    baseband = downconvert(iq, sample_rate, carrier)
    recovered_bits = demodulate_bpsk(baseband, samples_per_symbol)

    ber, polarity, final_bits = bit_error_rate(recovered_bits, truth_bits)

    print("Estimated carrier (precise):", round(carrier, 2), "Hz")
    print("Ground truth carrier:", meta.get("carrier_freq"), "Hz")
    print("Recovered", len(recovered_bits), "bits")
    print("Polarity used:", polarity)
    print("Bit error rate:", round(ber * 100, 2), "%")
    print()
    print("First 20 recovered bits:", list(final_bits[:20]))
    print("First 20 ground truth bits:", truth_bits[:20])


if __name__ == "__main__":
    main()
    input("Press Enter to close...")