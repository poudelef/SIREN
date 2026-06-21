"""
Audio loading, visualization, and low-level signal feature helpers for SIREN.
"""

import numpy as np
import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TARGET_SR = 16000  # YAMNet requires 16 kHz mono


def load_audio(file_path: str, target_sr: int = TARGET_SR):
    """
    Load any audio file librosa/audioread can handle (mp3, wav, m4a, flac...).
    Always returns mono float32 audio resampled to target_sr.
    """
    wav_data, sr = librosa.load(file_path, sr=target_sr, mono=True)
    wav_data = wav_data.astype(np.float32)
    duration = len(wav_data) / sr
    return wav_data, sr, duration


def denoise_audio(wav_data: np.ndarray, sr: int) -> np.ndarray:
    """
    Light noise reduction pass, useful before transcription on noisy calls.
    Falls back to the original signal if `noisereduce` isn't installed.
    """
    try:
        import noisereduce as nr
        return nr.reduce_noise(y=wav_data, sr=sr).astype(np.float32)
    except Exception:
        return wav_data


def compute_rms_envelope(wav_data: np.ndarray, sr: int, frame_length: int = 2048, hop_length: int = 512):
    rms = librosa.feature.rms(y=wav_data, frame_length=frame_length, hop_length=hop_length)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
    return times, rms


def compute_pitch_stats(wav_data: np.ndarray, sr: int):
    """
    Rough fundamental-frequency stats via librosa.pyin. One input signal for
    panic estimation (raised / variable pitch correlates with vocal stress).
    Returns (mean_f0, std_f0, voiced_fraction); zeros if no voiced frames found.
    """
    try:
        f0, voiced_flag, _voiced_prob = librosa.pyin(
            wav_data,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C6"),
            sr=sr,
        )
        voiced = f0[voiced_flag]
        if len(voiced) == 0:
            return 0.0, 0.0, 0.0
        return float(np.nanmean(voiced)), float(np.nanstd(voiced)), float(voiced_flag.mean())
    except Exception:
        return 0.0, 0.0, 0.0


def lowpass_naive(wav_data: np.ndarray, sr: int, cutoff_hz: float = 300.0) -> np.ndarray:
    """
    Naive frequency-domain low-pass via direct FFT bin zeroing.

    NOTE: useful to *visualize* spectral content below a cutoff, but it is
    NOT a reliable way to "isolate background noise." Speech and background
    sounds overlap across almost the whole spectrum, so hard-zeroing bins
    mostly just distorts the signal (this is why running a classifier on a
    low-passed clip gives nonsense labels). Real foreground/background
    separation needs source-separation models, not frequency masking. Kept
    here for the spectrum demo / judge Q&A, not for the actual pipeline.
    """
    fft = np.fft.fft(wav_data)
    freqs = np.fft.fftfreq(len(wav_data), d=1 / sr)
    filtered = fft.copy()
    filtered[np.abs(freqs) > cutoff_hz] = 0
    return np.fft.ifft(filtered).real.astype(np.float32)


def plot_waveform(wav_data: np.ndarray, sr: int):
    fig, ax = plt.subplots(figsize=(10, 3))
    librosa.display.waveshow(wav_data, sr=sr, ax=ax)
    ax.set_title("Waveform")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    fig.tight_layout()
    return fig


def plot_spectrum(wav_data: np.ndarray, sr: int):
    fft = np.fft.fft(wav_data)
    freqs = np.fft.fftfreq(len(fft), d=1 / sr)
    positive = freqs >= 0

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(freqs[positive], np.abs(fft[positive]))
    ax.set_title("Frequency Spectrum (FFT)")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude")
    fig.tight_layout()
    return fig


def plot_spectrogram(wav_data: np.ndarray, sr: int):
    D = librosa.amplitude_to_db(np.abs(librosa.stft(wav_data)), ref=np.max)
    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(D, sr=sr, x_axis="time", y_axis="hz", ax=ax)
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title("Spectrogram")
    fig.tight_layout()
    return fig

# ---------------------------------------------------------------------------
# Dispatcher-facing charts: plain-language, labeled for a non-technical
# reader (dispatcher / responding officer), not an audio engineer. These are
# meant to replace the raw FFT/spectrogram view as the *primary* visuals;
# the technical plots above stay available for anyone who wants them.
# ---------------------------------------------------------------------------
 
SOUND_COLORS = {
    "Siren": "#d62728",
    "Gunshot": "#7f0000",
    "Explosion": "#7f0000",
    "Glass breaking": "#e6550d",
    "Alarm": "#e6550d",
    "Screaming / shouting": "#a31515",
    "Crying": "#9467bd",
    "Dog": "#8c6d31",
    "Traffic / vehicles": "#1f77b4",
    "Construction": "#7f7f7f",
    "Door / impact": "#bcbd22",
    "Footsteps": "#7f7f7f",
    "Speech": "#2ca02c",
    "Silence": "#cccccc",
}
 
 
def _format_mmss(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    m, s = divmod(seconds, 60)
    return f"{m}:{s:02d}"
 
 
def _apply_mmss_xaxis(ax, duration_s: float):
    ax.set_xlim(0, max(duration_s, 1))
    ticks = ax.get_xticks()
    ticks = [t for t in ticks if 0 <= t <= duration_s]
    if not ticks:
        ticks = [0, duration_s]
    ax.set_xticks(ticks)
    ax.set_xticklabels([_format_mmss(t) for t in ticks])
    ax.set_xlabel("Time into call (mm:ss)")
 
 
def plot_voice_energy_timeline(rms_times: np.ndarray, rms: np.ndarray, duration_s: float):
    """
    Plain-language version of the waveform: how loud / energetic the
    caller's voice was, over time, normalized to THIS call's own loudest
    moment (0-100), with shaded "calm / raised / very loud" bands instead
    of raw, unitless amplitude numbers.
    """
    peak = float(rms.max()) if len(rms) and rms.max() > 0 else 1.0
    levels = (rms / peak) * 100.0
 
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.axhspan(0, 33, color="#2ca02c", alpha=0.06, zorder=0)
    ax.axhspan(33, 66, color="#ff7f0e", alpha=0.08, zorder=0)
    ax.axhspan(66, 100, color="#d62728", alpha=0.10, zorder=0)
 
    ax.fill_between(rms_times, levels, color="#1f4e8c", alpha=0.30, zorder=1)
    ax.plot(rms_times, levels, color="#1f4e8c", linewidth=1.5, zorder=2)
 
    ax.set_ylim(0, 100)
    ax.set_ylabel("Caller voice energy\n(relative to this call, 0-100)")
    ax.set_title("How loud / energetic the caller sounded, over time")
    _apply_mmss_xaxis(ax, duration_s)
 
    ax.text(0.995, 0.14, "Calm", transform=ax.transAxes, ha="right", fontsize=9, color="#2ca02c")
    ax.text(0.995, 0.47, "Raised", transform=ax.transAxes, ha="right", fontsize=9, color="#cc6600")
    ax.text(0.995, 0.80, "Very loud / distressed", transform=ax.transAxes, ha="right", fontsize=9, color="#a31515")
 
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig
 
 
def plot_sound_event_timeline(events: list, duration_s: float):
    """
    Timeline (one row per detected background-sound category) showing WHEN
    each sound happened during the call. Reads like a Gantt chart -- far
    easier to scan at a glance than a spectrogram for a non-technical
    reader, and ties directly back to the "Detected background sound
    events" list shown elsewhere in the report.
    """
    if not events:
        fig, ax = plt.subplots(figsize=(10, 1.6))
        ax.text(0.5, 0.5, "No notable background sounds detected above threshold",
                ha="center", va="center", fontsize=11)
        ax.axis("off")
        return fig
 
    categories = sorted({e["category"] for e in events})
    fig, ax = plt.subplots(figsize=(10, 0.55 * len(categories) + 1.6))
 
    for i, cat in enumerate(categories):
        intervals = [(e["start"], e["end"] - e["start"]) for e in events if e["category"] == cat]
        color = SOUND_COLORS.get(cat, "#999999")
        ax.broken_barh(intervals, (i - 0.35, 0.7), facecolors=color, edgecolors="white", linewidth=0.5)
 
    ax.set_yticks(range(len(categories)))
    ax.set_yticklabels(categories)
    ax.set_ylim(-0.6, len(categories) - 0.4)
    ax.set_title("When each background sound happened during the call")
    _apply_mmss_xaxis(ax, duration_s)
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig
 
 
def plot_sound_duration_bar(events: list):
    """
    Bar chart: total seconds each background-sound category was present
    during the call, longest first -- a quick "what did we hear, and how
    much of it" summary, labeled with exact seconds on each bar.
    """
    if not events:
        fig, ax = plt.subplots(figsize=(10, 1.6))
        ax.text(0.5, 0.5, "No notable background sounds detected above threshold",
                ha="center", va="center", fontsize=11)
        ax.axis("off")
        return fig
 
    totals = {}
    for e in events:
        totals[e["category"]] = totals.get(e["category"], 0.0) + (e["end"] - e["start"])
    items = sorted(totals.items(), key=lambda kv: kv[1], reverse=True)
    cats = [c for c, _ in items]
    durations = [d for _, d in items]
    colors = [SOUND_COLORS.get(c, "#999999") for c in cats]
 
    fig, ax = plt.subplots(figsize=(10, 0.5 * len(cats) + 1.5))
    bars = ax.barh(cats, durations, color=colors)
    ax.invert_yaxis()
    ax.set_xlabel("Total time detected (seconds)")
    ax.set_title("Which background sounds were heard, and for how long")
 
    max_dur = max(durations) if durations else 1.0
    for bar, d in zip(bars, durations):
        ax.text(bar.get_width() + max_dur * 0.02, bar.get_y() + bar.get_height() / 2,
                f"{d:.1f}s", va="center", fontsize=9)
 
    ax.set_xlim(0, max_dur * 1.18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig
 