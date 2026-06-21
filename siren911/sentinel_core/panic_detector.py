"""
Heuristic 'panic score' for a 911 caller, combining acoustic and linguistic
signals. This is NOT a clinically validated stress/emotion detector -- it's
a transparent, explainable heuristic meant to flag calls for closer human
attention, with a full breakdown so a dispatcher can sanity-check the score.
"""

import re
import numpy as np

EMOTIONAL_KEYWORDS = [
    "help", "please", "hurry", "scared", "afraid", "dying", "blood",
    "gun", "fire", "stop", "can't breathe", "hurts", "crying", "screaming",
    "now", "quick", "oh my god", "oh god", "somebody", "anybody",
]


def _normalize(value, low, high):
    if high <= low:
        return 0.0
    return float(np.clip((value - low) / (high - low), 0.0, 1.0))


def repeated_phrase_score(text: str) -> float:
    """Detects a caller repeating the same short phrase multiple times
    (a common panic marker, e.g. 'please hurry please hurry please hurry')."""
    words = re.findall(r"[a-zA-Z']+", text.lower())
    if len(words) < 6:
        return 0.0
    trigrams = [" ".join(words[i:i + 3]) for i in range(len(words) - 2)]
    if not trigrams:
        return 0.0
    counts = {}
    for tri in trigrams:
        counts[tri] = counts.get(tri, 0) + 1
    max_repeat = max(counts.values())
    return _normalize(max_repeat, 1, 5)


def keyword_score(text: str) -> float:
    low = text.lower()
    hits = sum(low.count(kw) for kw in EMOTIONAL_KEYWORDS)
    return _normalize(hits, 0, 8)


def speech_rate_score(text: str, duration_s: float) -> float:
    word_count = len(re.findall(r"[a-zA-Z']+", text))
    if duration_s <= 0:
        return 0.0
    wpm = word_count / (duration_s / 60.0)
    # ~110-150 wpm is calm conversational speech; 180+ reads as rushed/panicked
    return _normalize(wpm, 130, 220)


def energy_variability_score(rms: np.ndarray) -> float:
    if len(rms) == 0 or np.mean(rms) == 0:
        return 0.0
    cv = np.std(rms) / (np.mean(rms) + 1e-8)
    return _normalize(cv, 0.3, 1.2)


def pitch_variability_score(std_f0: float, mean_f0: float) -> float:
    if mean_f0 <= 0:
        return 0.0
    cv = std_f0 / mean_f0
    return _normalize(cv, 0.15, 0.5)


def estimate_panic(transcript_text: str, duration_s: float, rms: np.ndarray,
                    mean_f0: float, std_f0: float) -> dict:
    components = {
        "speech_rate": speech_rate_score(transcript_text, duration_s),
        "energy_variability": energy_variability_score(rms),
        "pitch_variability": pitch_variability_score(std_f0, mean_f0),
        "repeated_phrases": repeated_phrase_score(transcript_text),
        "emotional_language": keyword_score(transcript_text),
    }
    weights = {
        "speech_rate": 0.20,
        "energy_variability": 0.25,
        "pitch_variability": 0.15,
        "repeated_phrases": 0.15,
        "emotional_language": 0.25,
    }
    weighted = sum(components[k] * weights[k] for k in components)
    score_10 = round(float(np.clip(weighted * 10, 0, 10)), 1)
    return {
        "panic_score": score_10,
        "breakdown": components,
        "weights": weights,
    }
