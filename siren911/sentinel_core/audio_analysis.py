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
