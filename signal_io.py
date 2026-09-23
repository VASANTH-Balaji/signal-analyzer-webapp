import numpy as np
from scipy.io import wavfile
from scipy.signal import hilbert


IQ_FORMATS = {                     # name -> (numpy dtype, human label)
    "float32": (np.float32, "32-bit float I/Q (GNU Radio / .cfile)"),
    "float64": (np.float64, "64-bit float I/Q"),
    "int16":   (np.int16,   "16-bit signed integer I/Q (e.g. HackRF/USRP/SDR# int16)"),
    "uint8":   (np.uint8,   "8-bit unsigned I/Q, centred on 127.5 (RTL-SDR)"),
    "int8":    (np.int8,    "8-bit signed I/Q (HackRF)"),
}


def sniff_iq_format(path, nbytes=1 << 20):
    """Guess the sample format of a raw interleaved I/Q file. Returns
    (format_name, confidence 'high'|'low', notes list).

    The old loader silently assumed float32. Reading int16/uint8 data as float32
    does not raise - it produces a confidently wrong answer - so this checks the
    bytes first: float32 data has finite values of sane magnitude, while any other
    format reinterpreted as float32 gives NaN/Inf/denormal garbage spanning many
    decades. Integer formats are then told apart by their byte statistics."""
    with open(path, "rb") as f:
        raw = f.read(nbytes)
    notes = []
    if len(raw) < 256:
        return "float32", "low", ["file too small to sniff - assuming float32"]

    def plausible_float(dt):
        n = len(raw) // np.dtype(dt).itemsize
        with np.errstate(all="ignore"):
            x = np.frombuffer(raw[: n * np.dtype(dt).itemsize], dtype=dt).astype(np.float64)
        finite = np.isfinite(x)
        if finite.mean() < 0.9995:
            return False
        a = np.abs(x[finite])
        if a.max() == 0:
            return True
        nz = a[a > 0]
        if len(nz) < len(a) * 0.5:
            return False
        # Real float I/Q spans about a decade between its 25th and 95th percentile
        # magnitude; another format reinterpreted as float spans 10-100+ decades.
        p5, p50, p95 = np.percentile(nz, [25, 50, 95])   # 25th: tolerate exact zeros / tiny values
        span = np.log10(p95) - np.log10(p5)
        return 1e-8 < p50 < 1e8 and span < 6.0 and a.max() < 1e12

    for name in ("float32", "float64"):
        if plausible_float(IQ_FORMATS[name][0]):
            return name, "high", notes

    b = np.frombuffer(raw[: len(raw) // 2 * 2], dtype=np.uint8).astype(float)
    ev, od = b[0::2], b[1::2]

    def kurt(v):
        v = v - v.mean()
        return float(np.mean(v ** 4) / (np.mean(v ** 2) ** 2 + 1e-12))

    def flat(v):        # uniform bytes: std ~73.9, kurtosis ~1.8 (Gaussian data would be 3)
        return 64 < v.std() < 84 and kurt(v) < 2.3

    # int16 little-endian: the LOW byte of every sample is ~uniform whatever the signal is;
    # the HIGH byte carries the amplitude and is not.
    if flat(ev) and not flat(od):
        return "int16", "high", notes
    if not flat(ev) and not flat(od):
        if abs(ev.mean() - 127.5) < 40 and abs(od.mean() - 127.5) < 40 and ev.std() < 70 and od.std() < 70:
            return "uint8", "high", notes
        if ev.std() > 90 and od.std() > 90:
            return "int8", "high", notes      # small signed values wrap to both ends when read as unsigned
    notes.append("could not tell the sample format apart - assuming int16; verify manually")
    return "int16", "low", notes


def load_signal_ex(path, iq_dtype=None):
    """Like load_signal but also returns an info dict:
    {'format': ..., 'confidence': 'high'|'low'|'user', 'notes': [...], 'clipped_pct': ...}.
    iq_dtype: force a format name from IQ_FORMATS (skips sniffing)."""
    low = path.lower()
    if low.endswith(".wav"):
        iq, sr = _load_wav(path)
        return iq, sr, dict(format="wav", confidence="high", notes=[])
    if not low.endswith(".iq"):
        raise ValueError("Unsupported file type - expected .iq or .wav: " + path)
    if iq_dtype:
        if iq_dtype not in IQ_FORMATS:
            raise ValueError("unknown IQ format %r (choose from %s)" % (iq_dtype, ", ".join(IQ_FORMATS)))
        fmt, conf, notes = iq_dtype, "user", []
    else:
        fmt, conf, notes = sniff_iq_format(path)
    dt = IQ_FORMATS[fmt][0]
    raw = np.fromfile(path, dtype=dt)
    raw = raw[: len(raw) // 2 * 2]
    clipped = 0.0
    if fmt == "uint8":
        x = (raw.astype(np.float32) - 127.5) / 127.5
        clipped = float(np.mean((raw == 0) | (raw == 255)) * 100)
    elif fmt in ("int16", "int8"):
        full = float(np.iinfo(dt).max)
        x = raw.astype(np.float32) / full
        clipped = float(np.mean(np.abs(raw.astype(np.int64)) >= full) * 100)
    else:
        x = raw
    if len(x) == 0:
        raise ValueError("file contains no samples for format " + fmt)
    return x[0::2] + 1j * x[1::2], None, dict(format=fmt, confidence=conf, notes=notes,
                                              clipped_pct=clipped)


def load_signal(path):
    # Universal loader: accepts .iq (raw interleaved I/Q, format sniffed) or .wav files.
    # Always returns (complex_iq_array, sample_rate).
    iq, sr, _info = load_signal_ex(path)
    return iq, sr


def load_iq(path, dtype=np.float32):
    """Raw interleaved I/Q file -> complex array. Single shared implementation."""
    raw = np.fromfile(path, dtype=dtype)
    return raw[0::2] + 1j * raw[1::2]


_load_iq = load_iq  # backwards-compatible alias


def _load_wav(path):
    try:
        sample_rate, data = wavfile.read(path)
    except ValueError as e:
        if "Unknown wave file format" in str(e):
            sample_rate, data = _load_compressed_wav(path)
        else:
            raise

    # Normalize integer PCM formats to float range -1..1
    if np.issubdtype(data.dtype, np.integer):
        max_val = np.iinfo(data.dtype).max
        data = data.astype(np.float64) / max_val

    if data.ndim == 2 and data.shape[1] == 2:
        # Stereo wav: many SDR tools store I on left channel, Q on right channel
        iq = data[:, 0] + 1j * data[:, 1]
    else:
        # Mono wav: no explicit I/Q, so build an analytic (complex) signal
        # using the Hilbert transform - this lets the same downstream code
        # (spectrum, demod, etc) work on real-valued recordings too.
        mono = data if data.ndim == 1 else data[:, 0]
        iq = hilbert(mono)

    return iq, sample_rate


def _load_compressed_wav(path):
    # scipy.io.wavfile only supports PCM and IEEE_FLOAT. Real-world .wav
    # captures (especially telephony/VoIP recordings) often use compressed
    # codecs like A-law or mu-law instead. This manually parses the RIFF
    # chunks and decodes those two common cases as a fallback.
    import struct
    import audioop

    with open(path, "rb") as f:
        raw = f.read()

    if raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
        raise ValueError("Not a valid WAV file: " + path)

    pos = 12
    fmt_tag = channels = sample_rate = bits_per_sample = None
    data_bytes = None

    while pos < len(raw) - 8:
        chunk_id = raw[pos:pos + 4]
        chunk_size = struct.unpack("<I", raw[pos + 4:pos + 8])[0]
        chunk_data = raw[pos + 8:pos + 8 + chunk_size]

        if chunk_id == b"fmt ":
            fmt_tag, channels, sample_rate, _, _, bits_per_sample = \
                struct.unpack("<HHIIHH", chunk_data[:16])
        elif chunk_id == b"data":
            data_bytes = chunk_data

        pos += 8 + chunk_size + (chunk_size % 2)  # chunks are word-aligned

    if fmt_tag == 6:      # WAVE_FORMAT_ALAW
        pcm_bytes = audioop.alaw2lin(data_bytes, 2)
    elif fmt_tag == 7:    # WAVE_FORMAT_MULAW
        pcm_bytes = audioop.ulaw2lin(data_bytes, 2)
    else:
        raise ValueError(
            "Unsupported compressed WAV format tag: " + str(fmt_tag) +
            " (only PCM, IEEE_FLOAT, A-law, and mu-law are supported)"
        )

    data = np.frombuffer(pcm_bytes, dtype=np.int16)
    if channels and channels > 1:
        data = data.reshape(-1, channels)

    return sample_rate, data


if __name__ == "__main__":
    # Self-test: create a synthetic wav file and load it back
    sample_rate = 44100
    t = np.arange(sample_rate) / sample_rate
    tone = 0.5 * np.sin(2 * np.pi * 1000 * t)
    tone_int16 = (tone * 32767).astype(np.int16)
    wavfile.write("_selftest.wav", sample_rate, tone_int16)

    iq, sr = load_signal("_selftest.wav")
    print("Loaded wav - sample rate:", sr, "samples:", len(iq), "dtype:", iq.dtype)

    # Also test .iq path still works via _load_iq directly
    test_iq = (np.random.randn(100) + 1j * np.random.randn(100)).astype(np.complex64)
    interleaved = np.empty(200, dtype=np.float32)
    interleaved[0::2] = test_iq.real
    interleaved[1::2] = test_iq.imag
    interleaved.tofile("_selftest.iq")
    iq2, sr2 = load_signal("_selftest.iq")
    print("Loaded iq - samples:", len(iq2), "sample_rate (expected None):", sr2)

    import os
    os.remove("_selftest.wav")
    os.remove("_selftest.iq")
    print("Self-test passed")