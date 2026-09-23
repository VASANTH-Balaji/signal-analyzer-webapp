"""
analysis_pipeline.py - Qt-free analysis engine for SIH26147
(automated analysis of .IQ / .WAV files with signal-parameter extraction).

Everything that used to live inside the GUI class is here, so the same code
can drive the interactive window, the batch mode, or a plain script/test.

  * The blind-estimation functions (carrier, modulation, demodulation, sync
    word search ...) originate in the earlier gui_app_full.py (now removed),
    unchanged in maths.
  * run_pipeline() is the old run_pipeline / run_fec_stage flow, but instead of
    calling self.log_line() it reports through a PipelineContext, so it can run
    on a worker thread and feed parameters, stage status, plots and decoded
    output to whoever is listening.
  * render_plots() draws the four analysis plots on any matplotlib Figure.
"""
import json
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import signal

from signal_io import load_signal_ex
from fec_core import (viterbi_decode, block_deinterleave,
                      conv_deinterleave, diagonal_deinterleave,
                      prbs_deinterleave, rs_decode, concat_decode,
                      viterbi_decode_soft, conv_encode)
from fec_ldpc import (build_ldpc_code, ldpc_decode, ldpc_decode_codeword,
                      estimate_snr_db as ldpc_estimate_snr)
from demodulate_general import (demodulate_mpsk_or_qam, estimate_constellation_order,
                                estimate_timing_offset, estimate_order_and_phase,
                                ambiguity_angles, soft_demap_mpsk_or_qam)
from estimate_symbol_rate import estimate_symbol_rate


# ---------------------------------------------------------------------------
# Colours (shared by the Qt stylesheet and the matplotlib plots)
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Light Professional UI Theme
# ---------------------------------------------------------------------------
class C:
    # Main application
    BG = "#F6F8FB"
    PANEL = "#FFFFFF"
    RAISED = "#FFFFFF"
    BORDER = "#E3E8EF"

    # Typography
    TEXT = "#172033"
    MUTED = "#667085"
    DIM = "#98A2B3"

    # Brand / interactive
    ACCENT = "#2563EB"       # Primary blue
    ACCENT_LIGHT = "#EFF6FF"

    # Signal traces
    TRACE = "#7C3AED"        # Purple
    TRACE_LIGHT = "#F5F3FF"

    # Status
    GOOD = "#16A34A"
    GOOD_LIGHT = "#ECFDF3"

    WARN = "#D97706"
    WARN_LIGHT = "#FFF7ED"

    BAD = "#DC2626"
    BAD_LIGHT = "#FEF2F2"

    # Plot-specific
    GRID = "#E5E7EB"
TONE_COLORS = {
    "": C.TEXT,
    "good": C.GOOD,
    "warn": C.WARN,
    "bad": C.BAD,
    "muted": C.MUTED
}
MAX_SPECTROGRAM_SAMPLES = 2_000_000   # display only - analysis always uses every sample
MAX_SPECTROGRAM_COLUMNS = 1000
SNR_CAP_DB = 60.0

# (key, title) - order matters, the GUI shows them top to bottom
STAGES = [
    ("load", "Load file"),
    ("carrier", "Carrier estimation"),
    ("classify", "Modulation classification"),
    ("demod", "Timing and demodulation"),
    ("fec", "De-interleave and FEC decode"),
    ("sync", "Sync-word search"),
]

PARAM_GROUPS = ["Signal", "Carrier", "Spectrum", "Modulation", "Timing",
                "Decoding", "Validation"]


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
def eng_scale(v):
    v = abs(v)
    for div, prefix in ((1e9, "G"), (1e6, "M"), (1e3, "k")):
        if v >= div:
            return div, prefix
    return 1.0, ""


def fmt_eng(v, unit="Hz", decimals=1):
    """12500 -> '12,500.0 Hz (12.500 kHz)'; small values are shown plainly."""
    base = "{:,.{d}f} {u}".format(v, d=decimals, u=unit)
    div, prefix = eng_scale(v)
    if prefix:
        return "{} ({:.3f} {}{})".format(base, v / div, prefix, unit)
    return base


def fmt_short(v, unit="Hz"):
    """12500 -> '12.500 kHz' (compact form for the headline readouts)."""
    div, prefix = eng_scale(v)
    if prefix:
        return "{:.3f} {}{}".format(v / div, prefix, unit)
    return "{:.1f} {}".format(v, unit)


def time_scale(v):
    v = abs(v)
    if v >= 1:
        return 1.0, "s"
    if v >= 1e-3:
        return 1e-3, "ms"
    if v >= 1e-6:
        return 1e-6, "\u00b5s"
    return 1e-9, "ns"


def fmt_duration(seconds):
    div, unit = time_scale(seconds)
    return "{:.3f} {}".format(seconds / div, unit)


def fmt_snr(res):
    if res is None:
        return "n/a"
    snr, capped = res
    return "> {:.0f} dB".format(snr) if capped else "\u2248 {:.1f} dB".format(snr)


# ---------------------------------------------------------------------------
# Core signal processing (blind estimation) - carried over unchanged from the earlier GUI
# ---------------------------------------------------------------------------
def estimate_carrier_fft(iq, sample_rate):
    spectrum = np.fft.fftshift(np.fft.fft(iq))
    freqs = np.fft.fftshift(np.fft.fftfreq(len(iq), d=1 / sample_rate))
    return freqs[np.argmax(np.abs(spectrum))]


def estimate_carrier_squaring(iq, sample_rate):
    # More precise carrier estimate for PSK-type signals - squaring removes
    # binary phase modulation, leaving a clean tone at 2x carrier
    squared = iq ** 2
    n_fft = len(squared) * 8
    spectrum = np.fft.fftshift(np.fft.fft(squared, n=n_fft))
    freqs = np.fft.fftshift(np.fft.fftfreq(n_fft, d=1 / sample_rate))
    peak_idx = np.argmax(np.abs(spectrum))
    return freqs[peak_idx] / 2


def estimate_carrier_qam(iq, sample_rate):
    # Squaring only removes 2-fold (BPSK) phase ambiguity; square QAM
    # constellations (16/64/256-QAM) have 4-fold rotational symmetry
    # instead, so the same trick needs a 4th power here to collapse the
    # data modulation and leave a clean tone at 4x carrier. Using the
    # squaring estimate on a QAM signal leaves enough residual frequency
    # error that the constellation smears into a ring over the capture -
    # verified this against a real 16-QAM test file: order estimation
    # was garbage (guessed 256) with squaring, clean with this.
    raised = iq ** 4
    n_fft = len(raised) * 8
    spectrum = np.fft.fftshift(np.fft.fft(raised, n=n_fft))
    freqs = np.fft.fftshift(np.fft.fftfreq(n_fft, d=1 / sample_rate))
    peak_idx = np.argmax(np.abs(spectrum))
    return freqs[peak_idx] / 4


def downconvert(iq, sample_rate, carrier_freq):
    t = np.arange(len(iq)) / sample_rate
    return iq * np.exp(-1j * 2 * np.pi * carrier_freq * t)


def extract_features(baseband):
    amplitude = np.abs(baseband)
    phase = np.angle(baseband)
    amp_mean = np.mean(amplitude)
    amp_std = np.std(amplitude)
    amp_variance_ratio = (amp_std / amp_mean) if amp_mean > 0 else 0
    phase_diff = np.diff(np.unwrap(phase))
    freq_variance = np.var(phase_diff)
    return amp_variance_ratio, freq_variance


def detect_spectral_clusters(iq, sample_rate):
    # FSK shows two separated spectral peaks (one per tone frequency).
    # PSK/QAM show one dominant peak at the carrier. Counting clusters is a
    # much more reliable modulation cue than instantaneous frequency variance,
    # which is sensitive to the specific frequency spacing used.
    freqs, psd = signal.welch(iq, fs=sample_rate, nperseg=1024, return_onesided=False)
    freqs = np.fft.fftshift(freqs)
    psd = np.fft.fftshift(psd)
    psd_db = 10 * np.log10(psd + 1e-15)

    # A peak must be within 10 dB of the strongest one AND clearly above the noise
    # floor (median PSD + 6 dB). Without the second condition, noise ripples on a
    # noisy single-carrier signal all count as "peaks" and PSK gets called FSK.
    height = max(np.max(psd_db) - 10, float(np.median(psd_db)) + 6)
    peaks, _ = signal.find_peaks(psd_db, height=height, distance=10)
    peak_freqs = freqs[peaks]

    clusters = []
    for f in sorted(peak_freqs):
        if clusters and abs(f - clusters[-1][-1]) < sample_rate * 0.03:
            clusters[-1].append(f)
        else:
            clusters.append([f])
    cluster_centers = [float(np.mean(c)) for c in clusters]
    return cluster_centers


# PSK-vs-QAM decision thresholds on the amplitude spread (std/mean of |x|).
#  * per-SAMPLE spread (fallback): constant-envelope PSK reads ~0 only when noise is
#    tiny - noise_std 0.3 already gives ~0.28, so 0.15 turned any moderately noisy
#    BPSK/QPSK capture into "QAM".
#  * per-SYMBOL spread (preferred): averaging each symbol's samples suppresses noise
#    by ~sqrt(samples/symbol) while a QAM constellation keeps its ~0.34+ spread.
#    PSK stays below ~0.2 out to ~0 dB per-sample SNR at 20 samples/symbol, so 0.27
#    sits between the two clusters (see classifier_noise_sweep.py).
AMP_SPREAD_THRESHOLD_SAMPLE = 0.15
AMP_SPREAD_THRESHOLD_SYMBOL = 0.27


def amplitude_spread(baseband, sample_rate):
    """(spread, per_symbol_flag). Uses symbol-averaged points when blind timing works."""
    try:
        _, sps = estimate_symbol_rate(baseband, sample_rate)
        # align to the symbol grid first: averaging across a transition dips the
        # amplitude and makes constant-envelope PSK look like QAM
        pts = symbol_points(baseband[estimate_timing_offset(baseband, sps):], sps)
        if pts is not None:
            a = np.abs(pts)
            if np.mean(a) > 0:
                return float(np.std(a) / np.mean(a)), True
    except ValueError:
        pass
    ratio, _ = extract_features(baseband)
    return float(ratio), False


def classify_modulation(iq, sample_rate, baseband):
    cluster_centers = detect_spectral_clusters(iq, sample_rate)
    if len(cluster_centers) >= 2:
        return "FSK", cluster_centers
    spread, per_symbol = amplitude_spread(baseband, sample_rate)
    threshold = AMP_SPREAD_THRESHOLD_SYMBOL if per_symbol else AMP_SPREAD_THRESHOLD_SAMPLE
    if spread > threshold:
        return "QAM", cluster_centers
    return "PSK/BPSK", cluster_centers


def demodulate_fsk(iq, sample_rate, sps, f0, f1):
    # Non-coherent FSK demod: correlate each symbol against both candidate
    # frequencies and pick whichever has more energy. No carrier phase
    # recovery needed - the classic practical advantage of FSK.
    num_symbols = len(iq) // sps
    bits = np.zeros(num_symbols, dtype=int)
    t_sym = np.arange(sps) / sample_rate
    ref0 = np.exp(-1j * 2 * np.pi * f0 * t_sym)
    ref1 = np.exp(-1j * 2 * np.pi * f1 * t_sym)
    for i in range(num_symbols):
        seg = iq[i * sps:(i + 1) * sps]
        e0 = np.abs(np.sum(seg * ref0))
        e1 = np.abs(np.sum(seg * ref1))
        bits[i] = 1 if e1 > e0 else 0
    return bits


def demodulate_bpsk(baseband, samples_per_symbol):
    num_symbols = len(baseband) // samples_per_symbol
    bits = np.zeros(num_symbols, dtype=int)
    for i in range(num_symbols):
        start = i * samples_per_symbol
        segment = baseband[start:start + samples_per_symbol]
        bits[i] = 1 if np.mean(segment.real) > 0 else 0
    return bits


def demodulate_bpsk_soft(baseband, samples_per_symbol):
    # Same per-symbol averaging as demodulate_bpsk, but returns the
    # un-thresholded real value. LDPC's belief-propagation decoder needs
    # this soft information (how confident each symbol is, not just its
    # sign) - hard bits alone throw away exactly what BP uses.
    num_symbols = len(baseband) // samples_per_symbol
    soft = np.zeros(num_symbols, dtype=float)
    for i in range(num_symbols):
        start = i * samples_per_symbol
        segment = baseband[start:start + samples_per_symbol]
        soft[i] = np.mean(segment.real)
    return soft


def find_sync_word(bitstream, sync_word):
    # Same result as the original per-position loop (first best position wins),
    # but done as one correlation so long streams don't stall the analysis.
    bits_pm1 = 2 * np.asarray(bitstream).astype(int) - 1
    sync_pm1 = 2 * np.array(sync_word) - 1
    n = len(sync_word)
    if len(bits_pm1) < n:
        return 0, -1e9
    scores = np.correlate(bits_pm1, sync_pm1, mode="valid")
    best_pos = int(np.argmax(scores))
    return best_pos, scores[best_pos]


def bits_to_bytes(bits):
    bits = list(bits)
    pad = (-len(bits)) % 8
    bits = bits + [0] * pad
    out = []
    for i in range(0, len(bits), 8):
        byte_val = 0
        for b in bits[i:i + 8]:
            byte_val = (byte_val << 1) | int(b)
        out.append(byte_val)
    return out


def load_meta(file_path):
    for ext in (".iq", ".wav"):
        if file_path.lower().endswith(ext):
            meta_path = file_path[: -len(ext)] + "_meta.json"
            break
    else:
        meta_path = file_path + "_meta.json"
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            return json.load(f)
    return None


def fsk_symbol_discriminator(iq, sample_rate, f0, f1, win=24):
    # Short sliding correlation against each tone; energy DIFFERENCE
    # kept continuous (NOT hard-clipped to sign - see note below).
    #
    # win=24 (was 4): noise-robustness testing (noise_stress_test_fsk.py)
    # found win=4 breaks almost immediately - BER jumps to ~49% at
    # noise_std as low as 0.3, because a 4-sample correlation barely
    # averages out any noise. Sweeping window sizes against multiple
    # noise levels: win=24 holds 0% BER up to noise_std=1.0 (comparable
    # to the BPSK path's own robustness), degrading gracefully after -
    # not the aliasing failure win=20/40 have (those lock onto a
    # spurious half-symbol-period subharmonic even at zero noise).
    #
    # Continuous vs hard-clipped: estimate_symbol_rate uses an FFT
    # spectral-line technique on the squared difference of this signal.
    # Hard-clipping to +-1 creates sharp edges whose harmonic content
    # narrowly outcompetes the real symbol-rate tone (verified: 1024 vs
    # 1015 magnitude, wrong peak picked). Continuous avoids that.
    win = min(win, len(iq))
    t = np.arange(win) / sample_rate
    ref0 = np.exp(-1j * 2 * np.pi * f0 * t)
    ref1 = np.exp(-1j * 2 * np.pi * f1 * t)
    e0 = np.abs(np.correlate(iq, np.conj(ref0), mode="valid"))
    e1 = np.abs(np.correlate(iq, np.conj(ref1), mode="valid"))
    return (e1 - e0).astype(complex)


# ---------------------------------------------------------------------------
# Extra signal-parameter measurements (new)
# ---------------------------------------------------------------------------
def welch_psd(iq, sample_rate):
    freqs, psd = signal.welch(iq, fs=sample_rate, nperseg=min(1024, len(iq)),
                              return_onesided=False)
    return np.fft.fftshift(freqs), np.fft.fftshift(psd)


def occupied_bandwidth(freqs, psd, fraction=0.99):
    """Width of the band holding `fraction` of the total power. Returns (bw, lo, hi)."""
    total = np.sum(psd)
    if total <= 0 or len(freqs) < 3:
        return 0.0, float(freqs[0]), float(freqs[-1])
    c = np.cumsum(psd) / total
    tail = (1.0 - fraction) / 2.0
    lo_i = int(np.searchsorted(c, tail))
    hi_i = min(int(np.searchsorted(c, 1.0 - tail)), len(freqs) - 1)
    lo, hi = float(freqs[lo_i]), float(freqs[hi_i])
    return hi - lo, lo, hi


def snr_from_psd(freqs, psd, lo, hi):
    """Blind SNR from the spectrum: noise floor from outside the occupied band.
    Suits narrowband signals (FSK); returns None when the signal fills the band."""
    inband = (freqs >= lo) & (freqs <= hi)
    outband = ~inband
    if outband.sum() < 8 or inband.sum() < 2:
        return None
    df = float(np.mean(np.diff(freqs)))
    noise_density = float(np.median(psd[outband]))
    noise_power = noise_density * (hi - lo)
    total_power = float(np.sum(psd[inband]) * df)
    sig_power = total_power - noise_power
    if sig_power <= 0 or noise_power <= 0:
        return None
    snr = 10 * np.log10(sig_power / noise_power)
    return min(float(snr), SNR_CAP_DB), bool(snr >= SNR_CAP_DB)


# Kurtosis of the ideal (noise-free) constellation, needed by the M2M4 estimator
_QAM_KURTOSIS = {4: 1.0, 16: 1.32, 64: 1.381, 256: 1.395}


QAM_SNR_CAP_DB = 30.0   # M2M4 on QAM is unreliable above this with a few hundred symbols


def snr_m2m4(points, ka=1.0, cap=SNR_CAP_DB):
    """Blind SNR from 2nd/4th moments of symbol-rate samples (M2M4 estimator).
    Insensitive to carrier phase/frequency error, so it works before phase lock.
    ka = kurtosis of the clean constellation (1.0 for PSK)."""
    y = np.asarray(points)
    if len(y) < 16:
        return None
    m2 = float(np.mean(np.abs(y) ** 2))
    m4 = float(np.mean(np.abs(y) ** 4))
    if m2 <= 0:
        return None
    s2 = (m4 - 2 * m2 ** 2) / (ka - 2)
    if s2 <= 0:
        return None
    s = np.sqrt(s2)
    n = m2 - s
    if n <= 1e-9 * m2:
        return cap, True
    snr = 10 * np.log10(s / n)
    return min(float(snr), cap), bool(snr >= cap)


def symbol_points(baseband, sps):
    """One complex value per symbol (average of the middle half of each symbol)."""
    sps = int(sps)
    if sps < 1:
        return None
    n = len(baseband) // sps
    if n < 4:
        return None
    seg = np.asarray(baseband)[:n * sps].reshape(n, sps)
    lo = sps // 4
    hi = max(lo + 1, sps - sps // 4)
    return seg[:, lo:hi].mean(axis=1)


def mod_name(family, order=None):
    if family == "FSK":
        return "FSK"
    if family == "QAM":
        return "%d-QAM" % order
    return {2: "BPSK", 4: "QPSK"}.get(order, "%d-PSK" % order)


def _validate_carrier(ctx, meta, carrier, sample_rate, n_samples):
    if not (meta and "carrier_freq" in meta):
        return
    truth = float(meta["carrier_freq"])
    err = carrier - truth
    tol = 2 * sample_rate / (8 * n_samples)      # two bins of the 8x zero-padded FFT
    ctx.log("Ground truth (for validation only): " + str(meta["carrier_freq"]) + " Hz")
    ctx.param("Validation", "Carrier (metadata)", fmt_eng(truth))
    ctx.param("Validation", "Carrier error", "{:+,.1f} Hz".format(err),
              "good" if abs(err) <= tol else "warn")


def _family_matches(family, truth):
    return family.split("/")[0] in str(truth).upper()


# ---------------------------------------------------------------------------
# Plot data
# ---------------------------------------------------------------------------
def make_base_plot_data(iq, sample_rate):
    freqs, psd = welch_psd(iq, sample_rate)

    x = iq[:MAX_SPECTROGRAM_SAMPLES]
    nperseg = min(256, len(x))
    f, t, sxx = signal.spectrogram(x, fs=sample_rate, nperseg=nperseg,
                                   return_onesided=False)
    f = np.fft.fftshift(f)
    sxx = np.fft.fftshift(sxx, axes=0)
    sxx_db = 10 * np.log10(sxx + 1e-12)
    if sxx_db.shape[1] > MAX_SPECTROGRAM_COLUMNS:
        k = int(np.ceil(sxx_db.shape[1] / MAX_SPECTROGRAM_COLUMNS))
        m = (sxx_db.shape[1] // k) * k
        sxx_db = sxx_db[:, :m].reshape(sxx_db.shape[0], -1, k).mean(axis=2)
        t = t[:m].reshape(-1, k).mean(axis=1)

    return {
        "fs": sample_rate,
        "psd_f": freqs, "psd": psd, "psd_db": 10 * np.log10(psd + 1e-12),
        "spec_t": t, "spec_f": f, "spec_db": sxx_db,
        "points": iq[:2000], "points_kind": "raw",
    }


def make_time_trace(baseband, sample_rate, sps=None, label="Baseband"):
    n = int(24 * sps) if sps else 800
    n = max(2, min(len(baseband), n))
    seg = np.asarray(baseband[:n])
    return {"x": np.arange(n) / sample_rate, "i": seg.real, "q": seg.imag,
            "label": label}


# ---------------------------------------------------------------------------
# Pipeline context: how the pipeline talks to whoever runs it
# ---------------------------------------------------------------------------
@dataclass
class AnalysisOptions:
    sample_rate: Optional[float] = None    # manual override, Hz
    modulation: Optional[str] = None       # None = auto, else "FSK" | "PSK/BPSK" | "QAM"
    use_meta: bool = True                  # read the sidecar *_meta.json
    blind_fec: bool = True                 # when no FEC metadata: try to identify the code blind
    iq_format: Optional[str] = None        # None = sniff; or float32/float64/int16/uint8/int8


class PipelineContext:
    """Collects everything the pipeline reports and forwards it to optional callbacks
    (the GUI passes Qt signal emitters, batch mode passes none)."""

    def __init__(self, on_log=None, on_stage=None, on_param=None,
                 on_headline=None, on_plots=None):
        self.on_log, self.on_stage, self.on_param = on_log, on_stage, on_param
        self.on_headline, self.on_plots = on_headline, on_plots
        self.log_lines = []
        self.params = {}                       # {group: {label: value}}
        self.stages = {k: "pending" for k, _ in STAGES}
        self.summary = {}
        self.output = None                     # see set_output()
        self.current_stage = None
        self.blind_fec = True                  # set from AnalysisOptions by run_pipeline()

    def log(self, text=""):
        self.log_lines.append(text)
        if self.on_log:
            self.on_log(text)

    def stage(self, key, status, detail=""):
        self.stages[key] = status
        self.current_stage = key
        if self.on_stage:
            self.on_stage(key, status, detail)

    def param(self, group, label, value, tone=""):
        value = str(value)
        self.params.setdefault(group, {})[label] = value
        if self.on_param:
            self.on_param(group, label, value, tone)

    def headline(self, key, value, tone=""):
        if self.on_headline:
            self.on_headline(key, str(value), tone)

    def plots(self, data):
        if self.on_plots:
            self.on_plots(data)

    def set_summary(self, **kw):
        self.summary.update({k: (v.item() if isinstance(v, np.generic) else v)
                             for k, v in kw.items()})

    def set_output(self, label, bits=None, data=None):
        """Remember what the 'Decoded data' tab should show. Give bits (0/1 array)
        and/or data (bytes-like)."""
        if bits is not None:
            bits = np.asarray(bits).astype(np.uint8)
            if data is None:
                data = bytes(bits_to_bytes(bits))
        elif data is not None:
            data = bytes(bytearray(int(b) & 0xFF for b in data))
            bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
        self.output = {"label": label, "bits": bits, "data": data}

    def finalize(self):
        for key, status in list(self.stages.items()):
            if status in ("pending", "running"):
                self.stage(key, "skipped")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def run_pipeline(path, ctx, opts=None):
    """Analyse one file. Returns "ok" or "failed"; raises on unexpected errors."""
    opts = opts or AnalysisOptions()
    ctx.blind_fec = opts.blind_fec
    try:
        status = _run(path, ctx, opts)
        ctx.set_summary(status=status)
        return status
    except Exception as e:
        ctx.stage(ctx.current_stage or "load", "failed")
        ctx.set_summary(status="error", error=str(e))
        raise
    finally:
        ctx.finalize()


def _timing(signal_for_timing, sample_rate, meta, ctx):
    # Blind symbol-rate/timing estimate - this is what removes the
    # meta.json dependency for timing. meta is only used afterwards to
    # print a "ground truth" line for validation, and as a fallback if
    # the blind estimate itself fails (e.g. signal too short/noisy).
    try:
        symbol_rate, sps = estimate_symbol_rate(signal_for_timing, sample_rate)
        ctx.log("Blind symbol rate estimate: " + str(round(symbol_rate, 1)) +
                " Hz (samples/symbol = " + str(sps) + ")")
        if meta and "samples_per_symbol" in meta:
            ctx.log("Ground truth (for validation only): samples/symbol = " +
                    str(meta["samples_per_symbol"]))
            ctx.param("Validation", "Samples per symbol (metadata)",
                      meta["samples_per_symbol"],
                      "good" if int(meta["samples_per_symbol"]) == int(sps) else "bad")
        return sps, symbol_rate
    except ValueError as e:
        ctx.log("Blind symbol rate estimation failed: " + str(e))
        if meta and "samples_per_symbol" in meta:
            ctx.log("Falling back to samples_per_symbol from metadata.")
            sps = meta["samples_per_symbol"]
            return sps, sample_rate / sps
        ctx.log("No metadata fallback available - cannot determine symbol timing.")
        return None, None


def _preview(values, n=40):
    """First n values as plain ints (str(list(np_array)) prints np.int64(1) on NumPy 2)."""
    return str([int(v) for v in list(values)[:n]])


def _count_errors(decoded, truth):
    n = min(len(decoded), len(truth))
    if n == 0:
        return 0, 0
    return int(np.sum(np.asarray(decoded[:n]) != np.asarray(truth[:n]))), n


def _report_errors(ctx, decoded, truth, err_label="Bit errors vs ground truth",
                   rate_label="BER", errors=None, names=None):
    n = min(len(decoded), len(truth))
    if errors is None:
        errors, n = _count_errors(decoded, truth)
    rate = errors / n * 100 if n else 100
    ctx.log(err_label + ": " + str(errors) + " / " + str(n))
    ctx.log(rate_label + ": " + str(round(rate, 2)) + " %")
    tone = "good" if errors == 0 else ("warn" if rate < 5 else "bad")
    err_name, rate_name = names or (err_label.replace(" vs ground truth", ""), rate_label)
    ctx.param("Validation", err_name, "%d / %d" % (errors, n), tone)
    ctx.param("Validation", rate_name, "%.2f %%" % rate, tone)
    ctx.headline("result", "%s %.2f %%" % (rate_label, rate), tone)
    ctx.set_summary(ber_pct=rate)
    return rate


def _pick_polarity(candidates, truth=None, sync_scores=None, blind_scores=None):
    """candidates: [(label, decoded), ...] in preference order (normal first).
    Returns (label, decoded, how). Criteria in order: highest sync-word score; highest
    blind FEC-consistency score (needs no ground truth); fewest errors vs ground truth
    (only when the sidecar supplies it); else the first candidate."""
    if sync_scores:
        best = max(range(len(candidates)), key=lambda i: (sync_scores[i], -i))
        return candidates[best] + ("sync word",)
    if blind_scores:
        best = max(range(len(candidates)), key=lambda i: (round(blind_scores[i], 6), -i))
        return candidates[best] + ("FEC re-encode agreement (blind)",)
    if truth:
        best = min(range(len(candidates)),
                   key=lambda i: (_count_errors(candidates[i][1], truth)[0], i))
        return candidates[best] + ("ground truth from metadata",)
    return candidates[0] + ("default (no criterion available)",)


def _fit_length(bits, n):
    # QAM/M-PSK's symbol-centering trims a partial symbol at the signal
    # edge (correctly - it avoids sampling across a transition), which
    # can leave raw_bits a handful of bits short of rows*cols. block_
    # deinterleave's reshape needs that length exact, so pad/truncate to
    # fit rather than letting the whole decode fail over a few edge bits.
    if len(bits) == n:
        return bits
    if len(bits) > n:
        return bits[:n]
    return np.concatenate([bits, np.zeros(n - len(bits), dtype=int)])


def _run(path, ctx, opts):
    log = ctx.log

    # ---------------- STEP 1: load ----------------
    ctx.stage("load", "running")
    iq, wav_sample_rate, fmt_info = load_signal_ex(path, opts.iq_format)
    iq = np.asarray(iq)
    if iq.size < 64:
        raise ValueError("Signal is too short to analyse (%d samples)." % iq.size)
    if not np.all(np.isfinite(iq)):
        raise ValueError("Signal contains NaN/Inf - almost certainly the wrong sample format "
                         "(detected: %s). Set the I/Q format manually." % fmt_info["format"])
    if float(np.mean(np.abs(iq) ** 2)) == 0.0:
        raise ValueError("Signal is all zeros - wrong file, or wrong sample format "
                         "(detected: %s)." % fmt_info["format"])

    # SDR front ends leak their local oscillator to exactly 0 Hz. A plain FFT-peak /
    # power-of-N carrier estimator can lock onto that spike instead of the signal, so
    # remove a constant offset when it is a meaningful share of the power. Random-data
    # signals have mean ~0, so clean captures are untouched.
    dc = complex(np.mean(iq))
    dc_ratio = abs(dc) ** 2 / float(np.mean(np.abs(iq) ** 2))
    dc_removed = 1e-3 < dc_ratio < 0.5
    if dc_removed:
        iq = iq - dc
    meta = load_meta(path) if opts.use_meta else None

    if opts.sample_rate:
        v = float(opts.sample_rate)
        sample_rate = int(v) if v.is_integer() else v
        sr_source, sr_tone = "manual override", ""
    elif wav_sample_rate is not None:
        sample_rate, sr_source, sr_tone = wav_sample_rate, "WAV header", ""
    elif meta and "sample_rate" in meta:
        sample_rate, sr_source, sr_tone = meta["sample_rate"], "metadata file", ""
    else:
        sample_rate = 1000000
        sr_source = "assumed default - set it manually if this is wrong"
        sr_tone = "warn"

    power = np.abs(iq) ** 2
    mean_power = float(np.mean(power))
    papr = 10 * np.log10(float(np.max(power)) / mean_power) if mean_power > 0 else 0.0

    ctx.param("Signal", "File", os.path.basename(path))
    ctx.param("Signal", "Samples", "{:,}".format(len(iq)))
    ctx.param("Signal", "Duration", fmt_duration(len(iq) / sample_rate))
    ctx.param("Signal", "Sample rate", fmt_eng(sample_rate), sr_tone)
    ctx.param("Signal", "Sample-rate source", sr_source, sr_tone)
    fmt_tone = "warn" if (fmt_info["confidence"] == "low" or fmt_info.get("clipped_pct", 0) > 1) else ""
    ctx.param("Signal", "Sample format",
              "%s (%s)" % (fmt_info["format"], {"user": "set manually", "high": "auto-detected",
                                                 "low": "auto-detected, LOW confidence"}[fmt_info["confidence"]]),
              fmt_tone)
    if fmt_info.get("clipped_pct", 0) > 1:
        ctx.param("Signal", "Clipped samples", "%.1f %% - ADC overload, expect distortion" %
                  fmt_info["clipped_pct"], "warn")
    if dc_removed:
        ctx.param("Signal", "DC offset removed", "%.1f %% of power (LO leakage)" % (100 * dc_ratio))
    elif dc_ratio >= 0.5:
        ctx.param("Signal", "DC offset", "%.0f %% of power - NOT removed (could be the signal itself)" %
                  (100 * dc_ratio), "warn")
    ctx.param("Signal", "RMS amplitude", "%.4g" % np.sqrt(mean_power))
    ctx.param("Signal", "Peak-to-average power", "%.2f dB" % papr)
    if not opts.use_meta:
        ctx.param("Signal", "Sidecar metadata", "ignored (fully blind)")
    elif meta:
        ctx.param("Signal", "Sidecar metadata", "found")
    else:
        ctx.param("Signal", "Sidecar metadata", "none found (fully blind)")
    ctx.set_summary(file=os.path.basename(path), sample_rate=sample_rate)

    log("=== STEP 1: File loaded ===")
    log("File: " + os.path.basename(path))
    log("Samples: " + str(len(iq)))
    log("Sample rate: " + str(sample_rate) + " Hz")
    log("Sample format: %s (%s)" % (fmt_info["format"], fmt_info["confidence"]))
    for note in fmt_info.get("notes", []):
        log("Note: " + note)
    if dc_removed:
        log("DC offset removed: %.1f %% of total power." % (100 * dc_ratio))
    if sr_tone:
        log("Note: sample rate is " + sr_source)
    log("")

    base = make_base_plot_data(iq, sample_rate)
    obw, obw_lo, obw_hi = occupied_bandwidth(base["psd_f"], base["psd"])
    plot = dict(base)
    plot["obw"] = (obw_lo, obw_hi)
    ctx.stage("load", "done", "{:,} samples".format(len(iq)))

    # ---------------- STEP 2: carrier ----------------
    ctx.stage("carrier", "running")
    log("=== STEP 2: Blind carrier frequency estimation ===")
    carrier_fft = estimate_carrier_fft(iq, sample_rate)
    carrier_precise = estimate_carrier_squaring(iq, sample_rate)
    log("Coarse estimate (FFT peak): " + str(round(carrier_fft, 1)) + " Hz")
    log("Refined estimate (squaring trick): " + str(round(carrier_precise, 1)) + " Hz")
    ctx.param("Carrier", "Coarse (FFT peak)", fmt_eng(carrier_fft))
    ctx.param("Carrier", "Refined (squaring)", fmt_eng(carrier_precise))
    _validate_carrier(ctx, meta, carrier_precise, sample_rate, len(iq))
    log("")
    ctx.param("Spectrum", "Occupied bandwidth (99%)", fmt_eng(obw))
    ctx.param("Spectrum", "Band edges", "{:,.0f} Hz to {:,.0f} Hz".format(obw_lo, obw_hi))
    ctx.headline("carrier", fmt_short(carrier_precise))
    ctx.stage("carrier", "done", fmt_short(carrier_precise))

    baseband = downconvert(iq, sample_rate, carrier_precise)
    plot["carrier_hz"] = carrier_precise

    # ---------------- STEP 3: classify ----------------
    ctx.stage("classify", "running")
    log("=== STEP 3: Blind modulation classification ===")
    modulation, cluster_centers = classify_modulation(iq, sample_rate, baseband)
    if opts.modulation:
        log("Manual override: " + opts.modulation + " (auto-detect said " + modulation + ")")
        ctx.param("Modulation", "Auto-detect result", modulation)
        modulation = opts.modulation
    log("Predicted modulation: " + modulation)
    log("Detected spectral peak(s): " + str([round(c, 1) for c in cluster_centers]))
    amp_ratio, _ = extract_features(baseband)
    ctx.param("Modulation", "Family", modulation + ("  (manual)" if opts.modulation else ""))
    ctx.param("Modulation", "Spectral peaks", ", ".join("{:,.0f} Hz".format(c)
                                                       for c in cluster_centers) or "none")
    ctx.param("Modulation", "Amplitude spread (std/mean)", "%.3f" % amp_ratio)
    if meta and "modulation" in meta:
        log("Ground truth (for validation only): " + meta["modulation"])
        ctx.param("Validation", "Modulation (metadata)", meta["modulation"],
                  "good" if _family_matches(modulation, meta["modulation"]) else "bad")
    log("")
    ctx.headline("modulation", modulation)
    ctx.set_summary(modulation=modulation)
    ctx.stage("classify", "done", modulation)

    plot["time"] = make_time_trace(baseband, sample_rate, label="Baseband (carrier removed)")
    ctx.plots(dict(plot))

    # ---------------- STEP 4: FSK path ----------------
    if modulation == "FSK":
        ctx.stage("demod", "running")
        log("=== STEP 4: FSK demodulation ===")
        if len(cluster_centers) < 2:
            log("Could not resolve two distinct tone frequencies.")
            ctx.stage("demod", "failed", "one tone found")
            ctx.headline("result", "No tones", "bad")
            return "failed"
        f0_est, f1_est = sorted(cluster_centers[:2])
        log("Using tone frequencies: f0=" + str(round(f0_est, 1)) +
            " Hz, f1=" + str(round(f1_est, 1)) + " Hz")
        center = (f0_est + f1_est) / 2
        ctx.param("Modulation", "Tone 0", fmt_eng(f0_est))
        ctx.param("Modulation", "Tone 1", fmt_eng(f1_est))
        ctx.param("Modulation", "Tone spacing", fmt_eng(f1_est - f0_est))
        ctx.param("Modulation", "Centre frequency", fmt_eng(center))
        ctx.headline("carrier", fmt_short(center))
        plot["tones"] = [f0_est, f1_est]
        plot["carrier_hz"] = center

        # FSK-specific discriminator for blind timing: correlate short
        # sliding windows against each tone and take the energy difference.
        # It is flat within a symbol and flips only at real tone changes,
        # which is what the transition detector in estimate_symbol_rate needs.
        # (Raw instantaneous frequency was tried first and is NOT robust -
        # differentiating amplifies noise badly.)
        discriminator = fsk_symbol_discriminator(iq, sample_rate, f0_est, f1_est)
        sps, symbol_rate = _timing(discriminator, sample_rate, meta, ctx)
        if sps is None:
            ctx.stage("demod", "failed", "no timing")
            ctx.headline("result", "No timing", "bad")
            return "failed"
        ctx.param("Timing", "Symbol rate", fmt_eng(symbol_rate, "baud"))
        ctx.param("Timing", "Samples per symbol", sps)
        ctx.param("Timing", "Raw bit rate", fmt_eng(symbol_rate, "bit/s"))
        ctx.headline("symbol_rate", fmt_short(symbol_rate, "baud"))
        ctx.set_summary(carrier_hz=center, symbol_rate=symbol_rate)

        snr = snr_from_psd(base["psd_f"], base["psd"], obw_lo, obw_hi)
        ctx.param("Spectrum", "SNR (from spectrum)", fmt_snr(snr))
        ctx.headline("snr", fmt_snr(snr))
        if snr:
            ctx.set_summary(snr_db=snr[0], snr_capped=snr[1])

        bits = demodulate_fsk(iq, sample_rate, sps, f0_est, f1_est)
        log("Recovered " + str(len(bits)) + " bits")
        plot["sps"] = sps
        plot["time"] = make_time_trace(downconvert(iq, sample_rate, center), sample_rate, sps,
                                       "Baseband (centred between the two tones)")
        plot["disc"] = discriminator.real[:int(24 * sps) if sps else 800]
        plot["points_kind"] = "disc"
        ctx.plots(dict(plot))
        ctx.stage("demod", "done", "%d bits" % len(bits))
        ctx.param("Decoding", "Raw bits recovered", "{:,}".format(len(bits)))
        ctx.set_summary(fec="none (raw bits)")

        truth_bits = meta.get("ground_truth_bits") if meta else None
        if truth_bits:
            n = min(len(bits), len(truth_bits))
            errors = int(np.sum(bits[:n] != np.array(truth_bits[:n])))
            errors_inv = int(np.sum((1 - bits[:n]) != np.array(truth_bits[:n])))
            if errors_inv < errors:
                errors = errors_inv
            _report_errors(ctx, bits, truth_bits, errors=errors)
        else:
            ctx.headline("result", "Decoded %d bits" % len(bits), "")
        log("First 40 bits: " + _preview(bits))
        ctx.set_output("Raw demodulated bits (FSK, no FEC)", bits=bits)
        return "ok"

    # ---------------- STEP 4: QAM path ----------------
    if "PSK" not in modulation:
        ctx.stage("demod", "running")
        log("=== STEP 4: QAM demodulation ===")
        # Re-estimate carrier with the 4th-power method (see
        # estimate_carrier_qam) - the squaring-based baseband from STEP 2 is
        # BPSK-tuned and leaves enough residual frequency error to smear the
        # constellation into a ring over the capture.
        carrier_qam = estimate_carrier_qam(iq, sample_rate)
        log("Refined QAM carrier estimate (4th-power method): " +
            str(round(carrier_qam, 1)) + " Hz")
        ctx.param("Carrier", "Refined (4th power, QAM)", fmt_eng(carrier_qam))
        _validate_carrier(ctx, meta, carrier_qam, sample_rate, len(iq))
        ctx.headline("carrier", fmt_short(carrier_qam))
        baseband_qam = downconvert(iq, sample_rate, carrier_qam)
        plot["carrier_hz"] = carrier_qam

        sps, symbol_rate = _timing(baseband_qam, sample_rate, meta, ctx)
        if sps is None:
            ctx.stage("demod", "failed", "no timing")
            ctx.headline("result", "No timing", "bad")
            return "failed"

        # Real captures do not start on a symbol boundary and have an arbitrary carrier
        # phase; the frequency estimate above does not fix either. Align to the symbol
        # grid, then estimate order and phase together (see demodulate_general.py).
        t_off = estimate_timing_offset(baseband_qam, sps)
        baseband_qam = baseband_qam[t_off:]
        order, score, theta = estimate_order_and_phase(baseband_qam, sps, "QAM")
        order = int(order)
        baseband_qam = baseband_qam * np.exp(-1j * theta)
        log("Symbol timing offset: %d samples; carrier phase estimate: %.1f deg "
            "(known only modulo 90 deg - resolved by the FEC stage)" % (t_off, np.degrees(theta)))
        ctx.param("Timing", "Symbol timing offset", "%d samples" % t_off)
        ctx.param("Carrier", "Phase estimate", "%.1f deg (mod 90)" % np.degrees(theta))
        log("Blind constellation order estimate: " + str(order) +
            " (fit score " + str(round(score, 4)) + ", lower is better)")
        name = mod_name("QAM", order)
        ctx.param("Modulation", "Constellation", name)
        ctx.param("Modulation", "Order fit score", "%.4f (lower is better)" % score)
        ctx.headline("modulation", name)
        ctx.set_summary(modulation=name, order=order, carrier_hz=carrier_qam,
                        symbol_rate=symbol_rate)
        _timing_params(ctx, symbol_rate, sps, order)

        pts = symbol_points(baseband_qam, sps)
        if pts is not None:
            snr = snr_m2m4(pts, _QAM_KURTOSIS.get(order, 1.32), QAM_SNR_CAP_DB)
            ctx.param("Spectrum", "SNR (M2M4, blind)", fmt_snr(snr))
            ctx.headline("snr", fmt_snr(snr))
            if snr:
                ctx.set_summary(snr_db=snr[0], snr_capped=snr[1])
            plot["points"], plot["points_kind"] = pts[:4000], "symbols"
        plot["sps"] = sps
        plot["time"] = make_time_trace(baseband_qam, sample_rate, sps,
                                       "Baseband (carrier removed)")
        ctx.plots(dict(plot))

        raw_bits = demodulate_mpsk_or_qam(baseband_qam, sps, "QAM", order)
        log("Recovered " + str(len(raw_bits)) + " raw bits")
        log("")
        ctx.stage("demod", "done", name)
        ctx.param("Decoding", "Raw bits recovered", "{:,}".format(len(raw_bits)))

        variants = _build_variants(baseband_qam, sps, "QAM", order)
        truth_bits = meta.get("ground_truth_bits") if meta else None
        if truth_bits:
            _report_raw_ber(ctx, variants, truth_bits)
        _run_fec_stage(raw_bits, meta, baseband_qam, sps, ctx, variants)
        return ctx.summary.get("status", "ok")

    # ---------------- STEP 4: PSK path ----------------
    ctx.stage("demod", "running")
    log("=== STEP 4: PSK demodulation ===")
    samples_per_symbol, symbol_rate = _timing(baseband, sample_rate, meta, ctx)
    if samples_per_symbol is None:
        ctx.stage("demod", "failed", "no timing")
        ctx.headline("result", "No timing", "bad")
        return "failed"

    # Blind constellation-order estimate - covers BPSK/QPSK/8PSK/16PSK. The
    # squaring-based carrier only removes BPSK's 2-fold phase ambiguity; for
    # QPSK/8PSK/16PSK it leaves enough residual carrier error to smear the
    # constellation. So try both carrier estimates and keep whichever gives
    # the tighter constellation fit.
    carrier_psk_alt = estimate_carrier_qam(iq, sample_rate)
    baseband_alt = downconvert(iq, sample_rate, carrier_psk_alt)
    t_off = estimate_timing_offset(baseband, samples_per_symbol)
    baseband, baseband_alt = baseband[t_off:], baseband_alt[t_off:]
    order_sq, score_sq, theta_sq = estimate_order_and_phase(baseband, samples_per_symbol, "PSK")
    order_4th, score_4th, theta_4th = estimate_order_and_phase(baseband_alt, samples_per_symbol, "PSK")
    used_carrier = carrier_precise
    if score_4th < score_sq:
        psk_order, psk_score, theta = order_4th, score_4th, theta_4th
        baseband = baseband_alt
        used_carrier = carrier_psk_alt
        log("Using 4th-power carrier estimate (better constellation fit for M>2 PSK): " +
            str(round(carrier_psk_alt, 1)) + " Hz")
        ctx.param("Carrier", "Refined (4th power, M>2 PSK)", fmt_eng(carrier_psk_alt))
        _validate_carrier(ctx, meta, carrier_psk_alt, sample_rate, len(iq))
        ctx.headline("carrier", fmt_short(carrier_psk_alt))
    else:
        psk_order, psk_score, theta = order_sq, score_sq, theta_sq
    psk_order = int(psk_order)
    baseband = baseband * np.exp(-1j * theta)        # remove the constant carrier phase
    log("Symbol timing offset: %d samples; carrier phase estimate: %.1f deg "
        "(known only modulo the constellation symmetry - resolved by the FEC stage)" %
        (t_off, np.degrees(theta)))
    ctx.param("Timing", "Symbol timing offset", "%d samples" % t_off)
    ctx.param("Carrier", "Phase estimate", "%.1f deg" % np.degrees(theta))
    log("Blind PSK order estimate: " + str(psk_order) +
        " (fit score " + str(round(psk_score, 4)) + ", lower is better)")
    name = mod_name("PSK", psk_order)
    ctx.param("Modulation", "Constellation", name)
    ctx.param("Modulation", "Order fit score", "%.4f (lower is better)" % psk_score)
    ctx.headline("modulation", name)
    ctx.set_summary(modulation=name, order=psk_order, carrier_hz=used_carrier,
                    symbol_rate=symbol_rate)
    if meta and "psk_order" in meta:
        log("Ground truth (for validation only): psk_order=" + str(meta["psk_order"]))
        ctx.param("Validation", "PSK order (metadata)", meta["psk_order"],
                  "good" if int(meta["psk_order"]) == psk_order else "bad")
    _timing_params(ctx, symbol_rate, samples_per_symbol, psk_order)

    pts = symbol_points(baseband, samples_per_symbol)
    if pts is not None:
        snr = snr_m2m4(pts, 1.0)
        ctx.param("Spectrum", "SNR (M2M4, blind)", fmt_snr(snr))
        ctx.headline("snr", fmt_snr(snr))
        if snr:
            ctx.set_summary(snr_db=snr[0], snr_capped=snr[1])
        plot["points"], plot["points_kind"] = pts[:4000], "symbols"
    plot["carrier_hz"] = used_carrier
    plot["sps"] = samples_per_symbol
    plot["time"] = make_time_trace(baseband, sample_rate, samples_per_symbol,
                                   "Baseband (carrier removed)")
    ctx.plots(dict(plot))

    if psk_order == 2:
        # Keep the original hard-decision BPSK demod for order 2: its bit
        # polarity is what the downstream block/conv FEC branches were built
        # and tested against.
        raw_bits = demodulate_bpsk(baseband, samples_per_symbol)
    else:
        raw_bits = demodulate_mpsk_or_qam(baseband, samples_per_symbol, "PSK", psk_order)
    log("Recovered " + str(len(raw_bits)) + " raw bits")
    log("")
    ctx.stage("demod", "done", name)
    ctx.param("Decoding", "Raw bits recovered", "{:,}".format(len(raw_bits)))

    variants = _build_variants(baseband, samples_per_symbol, "PSK", psk_order)
    if psk_order != 2:
        truth_bits = meta.get("ground_truth_bits") if meta else None
        if truth_bits:
            _report_raw_ber(ctx, variants, truth_bits)
    _run_fec_stage(raw_bits, meta, baseband, samples_per_symbol, ctx, variants)
    return ctx.summary.get("status", "ok")


def _report_raw_ber(ctx, variants, truth_bits):
    """Raw (pre-FEC) BER for M-ary modulations. Blind phase recovery leaves a rotation
    ambiguity, so this reports the best of the N hypotheses - it measures demodulator
    quality. It does NOT mean the correct rotation is known: with no FEC, sync word or
    differential coding, nothing in the signal identifies it."""
    scored = [(_count_errors(bits, truth_bits)[0], i) for i, (_l, bits, _s) in enumerate(variants)]
    _, best = min(scored)
    ctx.log("Raw BER reported as best of %d phase-ambiguity hypotheses (%s) - the correct "
            "one cannot be identified blind without FEC / sync word / differential coding." %
            (len(variants), variants[best][0]))
    _report_errors(ctx, variants[best][1], truth_bits, names=("Raw bit errors", "Raw BER"))


def _build_variants(baseband, sps, mod_type, order):
    """Every hypothesis the FEC stage should try, as (label, hard_bits, soft_values).

    Blind carrier-phase recovery only fixes the phase modulo the constellation's
    symmetry, so the bit stream is ambiguous: BPSK by a sign flip, QPSK/QAM by 4
    rotations (90 deg), M-PSK by M rotations. Each rotation is demodulated separately
    (it permutes the bit labels, it is not a simple inversion). soft_values are
    "positive = 1" (BPSK: symbol mean; M-ary: max-log LLR) and feed the Viterbi
    decoder directly. The FEC stage picks the winner from the decoder's own
    consistency, not from ground truth."""
    if mod_type == "PSK" and order == 2:
        bits = demodulate_bpsk(baseband, sps)
        soft = demodulate_bpsk_soft(baseband, sps)
        return [("normal", bits, soft), ("inverted", 1 - bits, -soft)]
    variants = []
    for k, ang in enumerate(ambiguity_angles(mod_type, order)):
        rot = baseband * np.exp(-1j * ang)
        label = "normal" if k == 0 else "rotated %d deg" % round(np.degrees(ang))
        variants.append((label, demodulate_mpsk_or_qam(rot, sps, mod_type, order),
                         soft_demap_mpsk_or_qam(rot, sps, mod_type, order)))
    return variants


def _timing_params(ctx, symbol_rate, sps, order):
    bits_per_symbol = np.log2(order) if order and order > 1 else 1
    ctx.param("Timing", "Symbol rate", fmt_eng(symbol_rate, "baud"))
    ctx.param("Timing", "Samples per symbol", sps)
    ctx.param("Timing", "Raw bit rate", fmt_eng(symbol_rate * bits_per_symbol, "bit/s"))
    ctx.headline("symbol_rate", fmt_short(symbol_rate, "baud"))


def _viterbi_candidate(hard_deint, soft_deint, num_input_bits):
    """Viterbi-decode one hypothesis and score it WITHOUT ground truth: re-encode the
    decoded bits and measure agreement with what was received (1.0 = the received
    stream is a valid codeword; a wrong polarity/rotation/interleaver is far lower).
    Soft input is used when available; erased (0) soft values are ignored in the score."""
    if soft_deint is not None:
        dec = viterbi_decode_soft(soft_deint, num_input_bits)
        ref, keep = (np.asarray(soft_deint) > 0).astype(int), np.asarray(soft_deint) != 0
    else:
        dec = viterbi_decode(hard_deint, num_input_bits)
        ref, keep = np.asarray(hard_deint).astype(int), np.ones(len(hard_deint), dtype=bool)
    enc = conv_encode(dec)
    m = min(len(enc), len(ref))
    k = keep[:m]
    score = float(np.mean(enc[:m][k] == ref[:m][k])) if np.any(k) else 0.0
    return dec, score


BLIND_MAX_VARIANTS = 8        # rotation hypotheses searched blind (M-PSK above 8-PSK is skipped)


def _blind_fec_stage(ctx, raw_bits, variants):
    """No FEC metadata: try to identify a rate-1/2 convolutional code (and a block/diagonal
    interleaver in front of it) from the bits alone - see blind_detect.py for exactly what
    is and is not covered. Returns True when it produced a decode. Result is always
    labelled 'blind-detected' with its confidence; nothing is decoded on a guess."""
    import blind_detect as bd
    log = ctx.log
    truth = None
    log("=== STEP 5: Blind FEC identification (no FEC metadata available) ===")
    hyps = variants[:BLIND_MAX_VARIANTS]
    if len(variants) > BLIND_MAX_VARIANTS:
        log("Note: %d rotation hypotheses - only the first %d are searched blind." %
            (len(variants), BLIND_MAX_VARIANTS))

    def _vec(v):                                    # soft when we have it, else +-1 hard
        return np.asarray(v[2], dtype=float) if v[2] is not None else 2.0 * np.asarray(v[1], dtype=float) - 1.0

    # Stage A: convolutional code directly on the stream (no interleaver)
    best = None
    for h in hyps:
        r = bd.detect_convolutional_code(_vec(h))
        if r.get("candidate_name") and (best is None or r["confidence"] > best[1]["confidence"]):
            best = (h, r)
    if best and best[1]["detected"]:
        h, r = best
        log("Convolutional code identified: %s (agreement %.3f vs null %.3f, confidence %.2f), "
            "hypothesis '%s'" % (r["candidate_name"], r["agreement"], r["null_mean"],
                                 r["confidence"], h[0]))
        decoded = bd.blind_viterbi_decode(_vec(h), r["params"])
        return _finish_blind(ctx, decoded, "convolutional %s (no interleaver)" % r["candidate_name"],
                             r["confidence"], h[0])
    log("No convolutional code stands out on the raw stream%s." %
        (" (best %s: agreement %.3f vs null %.3f)" % (best[1]["candidate_name"], best[1]["agreement"],
                                                       best[1]["null_mean"]) if best else ""))

    # Stage B: block / diagonal interleaver + convolutional code
    found = None
    per_variant = max(150, 600 // max(1, len(hyps)))     # total search cost stays bounded
    for h in hyps:
        r = bd.detect_interleaver_and_code(_vec(h), max_decodes=per_variant)
        if r["detected"] and (found is None or r["confidence"] > found[1]["confidence"]):
            found = (h, r)
    if found:
        h, r = found
        b = r["best"]
        rows, cols, extra = b["dims"]
        x = np.concatenate([_vec(h), np.zeros(extra)])
        d = block_deinterleave(x, rows, cols, 0) if b["family"] == "block" else \
            diagonal_deinterleave(x, rows, cols, 0)
        params = bd.CODE_LIBRARY[b["code"]]
        log("%s interleaver %d x %d (+%d pad) with code %s identified (confidence %.2f, "
            "%d hypotheses tested), hypothesis '%s'" % (b["family"], rows, cols, extra, b["code"],
                                                        r["confidence"], r["hypotheses_tested"], h[0]))
        decoded = bd.blind_viterbi_decode(d, params)
        return _finish_blind(ctx, decoded, "%s interleaver %dx%d + convolutional %s" %
                             (b["family"], rows, cols, b["code"]), r["confidence"], h[0])
    log("No block/diagonal interleaver + convolutional code stood out either. "
        "(Not covered blind: pseudo-random interleavers, LDPC, Reed-Solomon.)")
    ctx.param("Decoding", "Blind FEC search", "ran - nothing identified with confidence")
    return False


def _finish_blind(ctx, decoded, scheme, confidence, hypothesis):
    name = scheme + "  [blind-detected]"
    ctx.param("Decoding", "FEC identification", "blind-detected (confidence %.2f)" % confidence,
              "good" if confidence >= 0.4 else "warn")
    ctx.param("Decoding", "FEC scheme (blind)", scheme)
    ctx.param("Decoding", "Polarity/rotation", hypothesis)
    ctx.param("Decoding", "Decoded bits (post-FEC)", "{:,}".format(len(decoded)))
    ctx.set_summary(fec=name, fec_source="blind-detected", fec_confidence=float(confidence))
    ctx.headline("result", "Blind decode, %d bits" % len(decoded), "good")
    ctx.log("Decoded %d bits with the blind-detected scheme (no metadata used)." % len(decoded))
    ctx.log("First 40 decoded bits: " + _preview(decoded))
    ctx.set_output("Decoded bits (%s)" % name, bits=decoded)
    ctx.stage("fec", "done", "blind-detected")
    return True


def _run_fec_stage(raw_bits, meta, baseband, samples_per_symbol, ctx, variants=None):
    # Shared by the QAM path and the order==2 (BPSK) PSK path: everything
    # from here on (LDPC / block+Viterbi / conv+RS / diagonal+Viterbi /
    # prbs+Viterbi + bitstream correlation) operates on a flat bit array
    # and doesn't care which demodulator produced it.
    log = ctx.log
    if variants is None:
        variants = [("normal", raw_bits, None), ("inverted", 1 - raw_bits, None)]
    ctx.stage("fec", "running")
    ctx.set_summary(status="ok")

    def fail(message, detail="failed"):
        log(message)
        ctx.stage("fec", "failed", detail)
        ctx.headline("result", "Decode failed", "bad")
        ctx.set_summary(status="failed")

    # --- LDPC decode path (checked first: needs soft symbols, not raw_bits) ---
    has_ldpc = meta and all(k in meta for k in ("ldpc_n", "ldpc_dv", "ldpc_dc", "ldpc_seed"))
    if has_ldpc:
        log("=== STEP 5: LDPC decode (belief propagation) ===")
        n = meta["ldpc_n"]
        d_v = meta["ldpc_dv"]
        d_c = meta["ldpc_dc"]
        ldpc_seed = meta["ldpc_seed"]
        truth_bits = meta.get("ground_truth_message_bits")

        H, G, perm = build_ldpc_code(n, d_v, d_c, ldpc_seed)
        # Prefer the first hypothesis whose belief-propagation result satisfies every
        # parity check. (With even check degree the all-ones word is a codeword, so a
        # complemented stream is also valid - that ambiguity cannot be resolved blind
        # and the un-inverted hypothesis, listed first, wins.)
        soft_symbols = None
        for _lbl, _hard, _soft in variants:
            cand = _soft if _soft is not None else (2.0 * _hard - 1.0)
            snr_c = ldpc_estimate_snr(cand[:n])
            d_hat, iters = ldpc_decode_codeword(H, cand[:n], snr_c)
            if np.all((H @ d_hat) % 2 == 0):
                soft_symbols = cand
                break
        if soft_symbols is None:
            soft_symbols = variants[0][2] if variants[0][2] is not None else 2.0 * variants[0][1] - 1.0
        decoded, snr_db = ldpc_decode(soft_symbols[:n], H, G, perm)

        log("LDPC params: n=" + str(n) + ", k=" + str(G.shape[1]) +
            ", d_v=" + str(d_v) + ", d_c=" + str(d_c))
        log("Blind SNR estimate used for decode: " + str(round(snr_db, 2)) + " dB")
        log("Decoded message bits: " + str(len(decoded)))
        scheme = "LDPC (n=%d, k=%d)" % (n, G.shape[1])
        ctx.param("Decoding", "FEC scheme (from metadata)", scheme)
        ctx.param("Decoding", "FEC identification", "metadata-assisted")
        ctx.set_summary(fec_source="metadata-assisted")
        ctx.param("Decoding", "LDPC decoder SNR estimate", "%.2f dB" % snr_db)
        ctx.param("Decoding", "Decoded message bits", "{:,}".format(len(decoded)))
        ctx.set_summary(fec=scheme, snr_db=float(snr_db), snr_capped=False)
        ctx.headline("snr", "\u2248 %.1f dB" % snr_db)
        if truth_bits:
            _report_errors(ctx, decoded, truth_bits)
        else:
            ctx.headline("result", "Decoded %d bits" % len(decoded), "")
        log("First 40 decoded bits: " + _preview(decoded))
        ctx.set_output("Decoded message bits (LDPC)", bits=decoded)
        ctx.stage("fec", "done", "LDPC")
        return

    # --- De-interleaving + FEC decode (scheme depends on which metadata is present) ---
    has_block_viterbi = meta and all(k in meta for k in
                                     ("interleaver_rows", "interleaver_cols", "interleaver_pad"))
    has_conv_rs = meta and all(k in meta for k in
                               ("interleave_branches", "interleave_delay", "interleave_total_delay",
                                "rs_nsym", "rs_codeword_len"))
    has_diagonal_viterbi = meta and all(k in meta for k in
                                        ("diag_interleaver_rows", "diag_interleaver_cols",
                                         "diag_interleaver_pad"))
    has_prbs_viterbi = meta and all(k in meta for k in ("prbs_seed", "prbs_pad"))
    has_concat = meta and all(k in meta for k in
                              ("concat_rs_nsym", "concat_rs_n", "concat_depth",
                               "concat_num_message_bytes"))

    if any((has_block_viterbi, has_conv_rs, has_diagonal_viterbi, has_prbs_viterbi, has_concat)):
        ctx.param("Decoding", "FEC identification", "metadata-assisted")
        ctx.set_summary(fec_source="metadata-assisted")

    if not any((has_block_viterbi, has_conv_rs, has_diagonal_viterbi, has_prbs_viterbi,
                has_concat)):
        if getattr(ctx, "blind_fec", True) and _blind_fec_stage(ctx, raw_bits, variants):
            return
        log("No interleaver/FEC metadata found - showing raw demodulated bits only.")
        log("First 40 bits: " + _preview(raw_bits))
        ctx.stage("fec", "skipped", "no FEC metadata")
        ctx.set_summary(fec="none (raw bits)")
        ctx.param("Decoding", "FEC", "none - raw bits shown")
        if ctx.summary.get("ber_pct") is None:
            ctx.headline("result", "Raw %d bits" % len(raw_bits), "")
        ctx.set_output("Raw demodulated bits (no FEC metadata)", bits=raw_bits)
        return

    if has_concat:
        # Genuine concatenation: TWO error-correcting codes in cascade.
        #   inner = rate-1/2 convolutional (Viterbi), outer = Reed-Solomon, joined by a
        #   byte interleaver that breaks up the bursts Viterbi leaves behind.
        # (Not to be confused with "Convolutional interleaver + RS" below, which has only
        # ONE code - RS - plus a convolutional *interleaver*.)
        log("=== STEP 5: Concatenated decode: Viterbi (inner) -> de-interleave -> Reed-Solomon (outer) ===")
        nsym, rs_n, depth = meta["concat_rs_nsym"], meta["concat_rs_n"], meta["concat_depth"]
        n_msg = meta["concat_num_message_bytes"]
        truth_bytes = meta.get("ground_truth_message_bytes")

        best = None
        use_soft = False
        for label, variant, soft_v in variants:
            try:
                if soft_v is not None:
                    msg, st = concat_decode(soft_v, n_msg, nsym, rs_n, depth, soft=True)
                else:
                    msg, st = concat_decode(variant, n_msg, nsym, rs_n, depth)
            except Exception:
                continue
            key = (st["failed_codewords"], st["rs_symbols_corrected"])
            if best is None or key < best[0]:
                best = (key, label, msg, st)
                use_soft = soft_v is not None
        if best is None:
            fail("Concatenated decoding failed on both polarities.", "concat failed")
            return
        _, label, decoded_bytes, st = best
        name = "Concatenated: conv(K=3, r=1/2) + RS(%d, %d) x%d interleaved" % (
            rs_n, rs_n - nsym, depth)
        log("Polarity used: " + label)
        log("Inner decoder input: " + ("soft samples" if use_soft else "hard bits"))
        log("RS codewords the outer decoder could not fix: %d / %d" %
            (st["failed_codewords"], st["total_codewords"]))
        log("Bytes corrected by the outer RS (i.e. left over after Viterbi): %d" %
            st["rs_symbols_corrected"])
        ctx.param("Decoding", "FEC scheme (from metadata)", name)
        ctx.param("Decoding", "Polarity", label)
        ctx.param("Decoding", "Inner decoder input", "soft" if use_soft else "hard")
        ctx.param("Decoding", "RS codewords failed",
                  "%d / %d" % (st["failed_codewords"], st["total_codewords"]),
                  "good" if st["failed_codewords"] == 0 else "bad")
        ctx.param("Decoding", "Bytes fixed by outer RS", st["rs_symbols_corrected"])
        ctx.param("Decoding", "Decoded message bytes", "{:,}".format(len(decoded_bytes)))
        ctx.set_summary(fec=name, rs_failed=st["failed_codewords"],
                        rs_corrected=st["rs_symbols_corrected"])
        if truth_bytes:
            errors = sum(1 for a, b in zip(decoded_bytes, truth_bytes) if a != b)
            _report_errors(ctx, decoded_bytes, truth_bytes, "Byte errors vs ground truth",
                           "Byte error rate", errors=errors)
        else:
            ctx.headline("result", "Decoded %d bytes" % len(decoded_bytes),
                         "good" if st["failed_codewords"] == 0 else "warn")
        ctx.set_output("Decoded message bytes (%s)" % name, data=decoded_bytes)
        ctx.stage("fec", "done", "Concatenated")
        return

    if has_diagonal_viterbi or has_prbs_viterbi:
        scheme = "Diagonal" if has_diagonal_viterbi else "PRBS"
        log("=== STEP 5: " + scheme + " de-interleaving + Viterbi FEC decode ===")
        num_total_bits = meta.get("num_total_bits") or meta.get("num_message_bits")
        truth_bits = meta.get("ground_truth_message_bits") or meta.get("ground_truth_bits")

        def _deint(x):
            if has_diagonal_viterbi:
                target_len = meta["diag_interleaver_rows"] * meta["diag_interleaver_cols"]
                return diagonal_deinterleave(
                    _fit_length(x, target_len), meta["diag_interleaver_rows"],
                    meta["diag_interleaver_cols"], meta["diag_interleaver_pad"])
            return prbs_deinterleave(x, meta["prbs_seed"], meta["prbs_pad"],
                                     block_len=meta.get("prbs_block_len"))

        candidates, blind, last_err = [], [], None
        for label, variant, soft_v in variants:
            try:
                dec, sc = _viterbi_candidate(_deint(variant),
                                             _deint(soft_v) if soft_v is not None else None,
                                             num_total_bits)
                candidates.append((label, dec))
                blind.append(sc)
            except Exception as e:
                last_err = e
                continue

        if not candidates:
            fail("Decoding failed" + (": " + str(last_err) if last_err else "."))
            return

        label, decoded_bits, how = _pick_polarity(candidates, truth_bits, blind_scores=blind)
        log("Polarity/rotation used: " + label + "  (chosen by: " + how + ")")
        log("Decoded bits (post-FEC): " + str(len(decoded_bits)))
        ctx.param("Decoding", "Hypothesis chosen by", how)
        name = scheme + " interleaver + Viterbi"
        ctx.param("Decoding", "FEC scheme (from metadata)", name)
        ctx.param("Decoding", "Polarity", label)
        ctx.param("Decoding", "Decoded bits (post-FEC)", "{:,}".format(len(decoded_bits)))
        ctx.set_summary(fec=name)
        if truth_bits:
            _report_errors(ctx, decoded_bits, truth_bits)
        else:
            ctx.headline("result", "Decoded %d bits" % len(decoded_bits), "")
        log("First 40 decoded bits: " + _preview(decoded_bits))
        ctx.set_output("Decoded bits (%s)" % name, bits=decoded_bits)
        ctx.stage("fec", "done", scheme + " + Viterbi")
        return

    if has_conv_rs:
        log("=== STEP 5: Convolutional de-interleaving + Reed-Solomon decode ===")
        B = meta["interleave_branches"]
        D = meta["interleave_delay"]
        delay = meta["interleave_total_delay"]
        codeword_len_bytes = meta["rs_codeword_len"]
        nsym = meta["rs_nsym"]
        truth_bytes = meta.get("ground_truth_message_bytes")

        best = None
        for label, variant, _soft in variants:
            try:
                deint = conv_deinterleave(variant, B, D)
                codeword_bits = deint[delay:delay + codeword_len_bytes * 8]
                codeword_bytes = bits_to_bytes(codeword_bits)
                decoded = rs_decode(codeword_bytes, nsym)
                # Blind criterion: a correct hypothesis needs the fewest byte corrections
                # (a wrong one either fails outright or "corrects" many bytes).
                corrected = sum(1 for a, b in zip(codeword_bytes, decoded) if a != b)
                errors = 0
                if truth_bytes:
                    n = min(len(decoded), len(truth_bytes))
                    errors = sum(1 for a, b in zip(decoded[:n], truth_bytes[:n]) if a != b)
                if best is None or corrected < best[3]:
                    best = (label, decoded, errors, corrected)
            except Exception:
                continue

        if best is None:
            fail("Reed-Solomon decoding failed on both polarities (too many errors).",
                 "RS failed")
            return

        label, decoded_bytes, errors, _corr = best
        log("Polarity/rotation used: " + label + "  (chosen by: fewest RS byte corrections, blind)")
        ctx.param("Decoding", "Hypothesis chosen by", "RS correction count (blind)")
        log("Decoded message bytes: " + str(len(decoded_bytes)))
        name = "Convolutional interleaver + RS(%d, %d)" % (
            codeword_len_bytes, codeword_len_bytes - nsym)
        ctx.param("Decoding", "FEC scheme (from metadata)", name)
        ctx.param("Decoding", "Polarity", label)
        ctx.param("Decoding", "Decoded message bytes", "{:,}".format(len(decoded_bytes)))
        ctx.set_summary(fec=name)
        if truth_bytes:
            n = min(len(decoded_bytes), len(truth_bytes))
            _report_errors(ctx, decoded_bytes, truth_bytes,
                           "Byte errors vs ground truth", "Byte error rate", errors=errors)
        else:
            ctx.headline("result", "Decoded %d bytes" % len(decoded_bytes), "")
        log("Decoded bytes: " + _preview(decoded_bytes, len(decoded_bytes)))
        ctx.set_output("Decoded message bytes (%s)" % name, data=decoded_bytes)
        ctx.stage("fec", "done", "RS decoded")
        return

    log("=== STEP 5: Block de-interleaving + Viterbi FEC decode ===")
    rows = meta["interleaver_rows"]
    cols = meta["interleaver_cols"]
    pad = meta["interleaver_pad"]
    num_total_bits = meta.get("num_total_bits") or meta.get("num_message_bits")
    target_len = rows * cols

    truth_key = "ground_truth_payload_bits" if "ground_truth_payload_bits" in meta \
        else "ground_truth_message_bits" if "ground_truth_message_bits" in meta \
        else "ground_truth_bits"
    truth_bits = meta.get(truth_key)
    sync_word = meta.get("sync_word")

    candidates, scores, blind, last_err = [], [], [], None
    for label, variant, soft_v in variants:
        try:
            deint = block_deinterleave(_fit_length(variant, target_len), rows, cols, pad)
            deint_soft = (block_deinterleave(_fit_length(soft_v, target_len), rows, cols, pad)
                          if soft_v is not None else None)
            decoded, sc = _viterbi_candidate(deint, deint_soft, num_total_bits)
            if sync_word is not None:
                _, score = find_sync_word(decoded, sync_word)
                scores.append(score)
            candidates.append((label, decoded))
            blind.append(sc)
        except Exception as e:
            last_err = e
            continue

    if not candidates:
        fail("Decoding failed" + (": " + str(last_err) if last_err else "."))
        return

    label, decoded_bits, how = _pick_polarity(
        candidates, truth_bits if sync_word is None else None,
        scores if sync_word is not None else None, blind_scores=blind)
    log("Polarity/rotation used: " + label + "  (chosen by: " + how + ")")
    ctx.param("Decoding", "Hypothesis chosen by", how)
    log("Decoded bits (post-FEC): " + str(len(decoded_bits)))
    name = "Block interleaver + Viterbi"
    ctx.param("Decoding", "FEC scheme (from metadata)", name)
    ctx.param("Decoding", "Polarity", label)
    ctx.param("Decoding", "Decoded bits (post-FEC)", "{:,}".format(len(decoded_bits)))
    ctx.set_summary(fec=name)
    ctx.stage("fec", "done", "Block + Viterbi")

    # --- Bitstream correlation (if a sync word is defined) ---
    if sync_word is not None:
        ctx.stage("sync", "running")
        log("")
        log("=== STEP 6: Bitstream correlation (header detection) ===")
        pos, score = find_sync_word(decoded_bits, sync_word)
        header_end = pos + len(sync_word)
        payload_len = meta.get("payload_len", len(decoded_bits) - header_end)
        payload = decoded_bits[header_end: header_end + payload_len]

        log("Sync word found at bit position: " + str(pos) +
            " (score " + str(int(score)) + "/" + str(len(sync_word)) + ")")
        log("Header region: bits " + str(pos) + " to " + str(header_end))
        log("Payload region: bits " + str(header_end) + " to " + str(header_end + payload_len))
        perfect = int(score) == len(sync_word)
        ctx.param("Decoding", "Sync word position", "bit %d" % pos)
        ctx.param("Decoding", "Sync word score", "%d / %d" % (int(score), len(sync_word)),
                  "good" if perfect else "warn")
        ctx.param("Decoding", "Payload region", "bits %d to %d" % (header_end, header_end + payload_len))

        if truth_bits:
            log("")
            _report_errors(ctx, payload, truth_bits, "Payload bit errors", "Payload BER")
        else:
            ctx.headline("result", "Sync %d/%d" % (int(score), len(sync_word)),
                         "good" if perfect else "warn")
        log("First 40 payload bits: " + _preview(payload))
        ctx.set_output("Payload bits (after sync word, %s)" % name, bits=payload)
        ctx.stage("sync", "done", "at bit %d" % pos)
    else:
        if truth_bits:
            _report_errors(ctx, decoded_bits, truth_bits)
        else:
            ctx.headline("result", "Decoded %d bits" % len(decoded_bits), "")
        log("First 40 decoded bits: " + _preview(decoded_bits))
        ctx.set_output("Decoded bits (%s)" % name, bits=decoded_bits)


# ---------------------------------------------------------------------------
# Plot rendering (matplotlib only - the Qt window just hands over its Figure)
# ---------------------------------------------------------------------------
def _style_axes(ax, title):
    ax.set_facecolor(C.PANEL)
    ax.set_title(title, color=C.TEXT, fontsize=10, loc="left", pad=8)
    ax.tick_params(colors=C.MUTED, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(C.BORDER)
    ax.xaxis.label.set_color(C.MUTED)
    ax.yaxis.label.set_color(C.MUTED)
    ax.xaxis.label.set_size(9)
    ax.yaxis.label.set_size(9)
    ax.grid(True, color=C.BORDER, linewidth=0.5, alpha=0.6)


def render_plots(fig, d):
    """Draw the four analysis panels (spectrum, waterfall, time trace,
    constellation/discriminator) onto `fig`."""
    fig.clear()
    fig.set_facecolor(C.BG)
    if not d:
        fig.text(0.5, 0.5, "Open a .iq or .wav file to see its spectrum, waterfall,\n"
                           "time trace and constellation here.",
                 ha="center", va="center", color=C.MUTED, fontsize=11)
        return

    gs = fig.add_gridspec(2, 2)
    ax_psd = fig.add_subplot(gs[0, 0])
    ax_wf = fig.add_subplot(gs[0, 1])
    ax_t = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])

    # --- spectrum ---
    f = np.asarray(d["psd_f"])
    fdiv, fpre = eng_scale(np.max(np.abs(f)))
    _style_axes(ax_psd, "Spectrum")
    if d.get("obw"):
        ax_psd.axvspan(d["obw"][0] / fdiv, d["obw"][1] / fdiv, color=C.ACCENT, alpha=0.08,
                       lw=0, label="99% bandwidth")
    ax_psd.plot(f / fdiv, d["psd_db"], color=C.TRACE, lw=1.0)
    if d.get("carrier_hz") is not None:
        ax_psd.axvline(d["carrier_hz"] / fdiv, color=C.ACCENT, ls="--", lw=1.0, label="carrier")
    for tone in d.get("tones", []):
        ax_psd.axvline(tone / fdiv, color=C.ACCENT, ls=":", lw=1.0)
    ax_psd.set_xlabel("Frequency (%sHz)" % fpre)
    ax_psd.set_ylabel("Power (dB)")
    handles, labels = ax_psd.get_legend_handles_labels()
    if handles:
        leg = ax_psd.legend(loc="upper right", fontsize=7, frameon=False)
        for text in leg.get_texts():
            text.set_color(C.MUTED)

    # --- waterfall ---
    t = np.asarray(d["spec_t"])
    fw = np.asarray(d["spec_f"])
    sdb = np.asarray(d["spec_db"])
    tdiv, tunit = time_scale(t.max() if len(t) else 1.0)
    wdiv, wpre = eng_scale(np.max(np.abs(fw)))
    _style_axes(ax_wf, "Waterfall")
    ax_wf.grid(False)
    lo = float(np.percentile(sdb, 5))
    try:
        ax_wf.pcolormesh(t / tdiv, fw / wdiv, sdb, shading="auto", cmap="magma",
                         vmin=lo, vmax=float(np.max(sdb)))
    except Exception:   # e.g. a file so short the spectrogram has one column
        ax_wf.imshow(sdb, aspect="auto", origin="lower", cmap="magma", vmin=lo,
                     extent=[float(t.min()) / tdiv, float(t.max()) / tdiv + 1e-9,
                             float(fw.min()) / wdiv, float(fw.max()) / wdiv])
    ax_wf.set_xlabel("Time (%s)" % tunit)
    ax_wf.set_ylabel("Frequency (%sHz)" % wpre)

    # --- time trace ---
    tr = d.get("time")
    _style_axes(ax_t, tr["label"] if tr else "Time trace")
    if tr:
        xdiv, xunit = time_scale(tr["x"][-1] if len(tr["x"]) else 1.0)
        ax_t.plot(tr["x"] / xdiv, tr["i"], color=C.TRACE, lw=1.0, label="I")
        ax_t.plot(tr["x"] / xdiv, tr["q"], color=C.ACCENT, lw=1.0, label="Q")
        sps = d.get("sps")
        if sps and len(tr["x"]) > 1:
            fs = d["fs"]
            k = 0
            while k * sps < len(tr["x"]):
                ax_t.axvline(k * sps / fs / xdiv, color=C.BORDER, lw=0.6, zorder=0)
                k += 1
        ax_t.set_xlabel("Time (%s)" % xunit)
        ax_t.set_ylabel("Amplitude")
        ax_t.margins(y=0.3)
        leg = ax_t.legend(loc="upper right", fontsize=7, ncol=2, frameon=True,
                          facecolor=C.PANEL, edgecolor=C.BORDER, framealpha=0.9)
        for text in leg.get_texts():
            text.set_color(C.TEXT)

    # --- constellation / discriminator ---
    kind = d.get("points_kind", "raw")
    if kind == "disc" and d.get("disc") is not None:
        _style_axes(ax_c, "FSK discriminator (energy at f1 minus f0)")
        disc = np.asarray(d["disc"])
        ax_c.plot(np.arange(len(disc)), disc, color=C.TRACE, lw=1.0)
        ax_c.axhline(0, color=C.BORDER, lw=0.8)
        ax_c.set_xlabel("Sample")
    else:
        pts = np.asarray(d["points"])
        if len(pts) > 4000:
            pts = pts[:4000]
        title = "Recovered symbols" if kind == "symbols" else "Raw IQ (before carrier removal)"
        _style_axes(ax_c, title)
        ax_c.axhline(0, color=C.BORDER, lw=0.8)
        ax_c.axvline(0, color=C.BORDER, lw=0.8)
        ax_c.scatter(pts.real, pts.imag, s=5, alpha=0.55, linewidths=0, color=C.TRACE)
        ax_c.set_xlabel("In-phase")
        ax_c.set_ylabel("Quadrature")
        ax_c.set_aspect("equal", adjustable="datalim")


# ---------------------------------------------------------------------------
# Text views of decoded data, and the exported report
# ---------------------------------------------------------------------------
def format_bits(bits, group=8, per_line=64, max_bits=200_000):
    """'000000  01010101 11001100 ...' - 64 bits per line, grouped by byte."""
    bits = np.asarray(bits).astype(int)
    total = len(bits)
    if max_bits is not None:
        bits = bits[:max_bits]
    lines = []
    for off in range(0, len(bits), per_line):
        s = "".join(str(int(b)) for b in bits[off:off + per_line])
        lines.append("%06d  %s" % (off, " ".join(s[i:i + group] for i in range(0, len(s), group))))
    if len(bits) < total:
        lines.append("... showing the first {:,} of {:,} bits".format(len(bits), total))
    return "\n".join(lines)


def _printable(b, keep_newlines=False):
    if 32 <= b < 127 or (keep_newlines and b in (9, 10, 13)):
        return chr(b)
    return "."


def format_hexdump(data, max_bytes=65536):
    data = bytes(data)
    total = len(data)
    data = data[:max_bytes]
    lines = []
    for off in range(0, len(data), 16):
        chunk = data[off:off + 16]
        lines.append("%06x  %-47s  |%s|" % (off, " ".join("%02x" % b for b in chunk),
                                           "".join(_printable(b) for b in chunk)))
    if len(data) < total:
        lines.append("... showing the first {:,} of {:,} bytes".format(len(data), total))
    return "\n".join(lines)


def format_ascii(data, max_bytes=100_000):
    data = bytes(data)
    text = "".join(_printable(b, keep_newlines=True) for b in data[:max_bytes])
    if len(data) > max_bytes:
        text += "\n... showing the first {:,} of {:,} bytes".format(max_bytes, len(data))
    return text


def build_report(path, opts, ctx, elapsed=None):
    """Everything worth keeping from one analysis, as a JSON-friendly dict."""
    import datetime
    out = ctx.output
    return {
        "tool": "Signal Analyzer (SIH26147)",
        "generated": datetime.datetime.now().isoformat(timespec="seconds"),
        "file": os.path.abspath(path),
        "options": {"sample_rate_override_hz": opts.sample_rate,
                    "modulation_override": opts.modulation,
                    "use_sidecar_metadata": opts.use_meta},
        "status": ctx.summary.get("status"),
        "elapsed_seconds": None if elapsed is None else round(elapsed, 3),
        "summary": ctx.summary,
        "parameters": ctx.params,
        "stages": ctx.stages,
        "decoded_output": None if not out else {
            "description": out["label"],
            "num_bits": None if out["bits"] is None else int(len(out["bits"])),
            "num_bytes": len(out["data"]),
            "hex_preview_first_512_bytes": bytes(out["data"][:512]).hex(),
        },
        "log": ctx.log_lines,
    }


def report_to_text(rep):
    lines = ["Signal Analyzer report (SIH26147)", "=" * 34,
             "File:      " + rep["file"], "Generated: " + rep["generated"],
             "Status:    " + str(rep["status"]), ""]
    for group, rows in rep["parameters"].items():
        lines.append(group)
        lines.append("-" * len(group))
        width = max(len(k) for k in rows) if rows else 0
        for k, v in rows.items():
            lines.append("  %-*s  %s" % (width, k, v))
        lines.append("")
    if rep["decoded_output"]:
        d = rep["decoded_output"]
        lines += ["Decoded output", "--------------",
                  "  %s (%s bytes)" % (d["description"], d["num_bytes"]), ""]
    lines += ["Log", "---"] + rep["log"]
    return "\n".join(lines) + "\n"
