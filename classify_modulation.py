import json
import numpy as np

IQ_FILE = "test_signal.iq"
META_FILE = "test_signal_meta.json"


def load_iq(path, dtype=np.float32):
    raw = np.fromfile(path, dtype=dtype)
    return raw[0::2] + 1j * raw[1::2]


def estimate_carrier_and_downconvert(iq, sample_rate):
    # Find the strongest frequency component (carrier) and shift it to baseband
    spectrum = np.fft.fftshift(np.fft.fft(iq))
    freqs = np.fft.fftshift(np.fft.fftfreq(len(iq), d=1 / sample_rate))
    peak_freq = freqs[np.argmax(np.abs(spectrum))]
    t = np.arange(len(iq)) / sample_rate
    baseband = iq * np.exp(-1j * 2 * np.pi * peak_freq * t)
    return baseband, peak_freq


def extract_features(baseband):
    amplitude = np.abs(baseband)
    phase = np.angle(baseband)

    amp_mean = np.mean(amplitude)
    amp_std = np.std(amplitude)
    # Normalized amplitude variance - key feature for QAM vs PSK/FSK
    amp_variance_ratio = (amp_std / amp_mean) if amp_mean > 0 else 0

    # Phase jump statistics - PSK has discrete jumps, FSK has continuous drift
    phase_diff = np.diff(np.unwrap(phase))
    phase_diff_std = np.std(phase_diff)

    # Instantaneous frequency variance - high for FSK, low for PSK/QAM
    inst_freq = phase_diff
    freq_variance = np.var(inst_freq)

    return {
        "amp_variance_ratio": float(amp_variance_ratio),
        "phase_diff_std": float(phase_diff_std),
        "freq_variance": float(freq_variance),
    }


def classify(features):
    # Simple rule-based classifier using the extracted features.
    # Thresholds are approximate starting points - refine using labeled test data.
    if features["freq_variance"] > 0.5:
        return "FSK"
    elif features["amp_variance_ratio"] > 0.15:
        return "QAM"
    else:
        return "PSK"


def main():
    with open(META_FILE) as f:
        meta = json.load(f)
    sample_rate = meta["sample_rate"]

    iq = load_iq(IQ_FILE)
    baseband, carrier = estimate_carrier_and_downconvert(iq, sample_rate)
    features = extract_features(baseband)
    prediction = classify(features)

    print("Estimated carrier frequency:", round(carrier, 1), "Hz")
    print("Ground truth carrier (metadata):", meta.get("carrier_freq"), "Hz")
    print()
    print("Extracted features:")
    for k, v in features.items():
        print(" ", k, "=", round(v, 4))
    print()
    print("Predicted modulation:", prediction)
    print("Ground truth modulation (metadata):", meta.get("modulation"))


if __name__ == "__main__":
    main()
    input("Press Enter to close...")