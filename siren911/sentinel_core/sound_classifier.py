"""
Environmental sound detection using YAMNet (Google's AudioSet model).

Rather than only looking at each frame's single top-1 class (which mostly
just says "Speech" for an entire call, since a talking caller usually
dominates loudness), this module scores EVERY category of interest in
EVERY frame independently. That lets us flag a siren or alarm quietly
happening underneath the caller's voice, not just whatever sound happens
to be loudest at that instant.
"""

import csv
import numpy as np

YAMNET_HANDLE = "https://tfhub.dev/google/yamnet/1"
FRAME_HOP_SECONDS = 0.48  # YAMNet's fixed patch hop

# category -> substrings matched against AudioSet display_name (case-insensitive)
CATEGORY_KEYWORDS = {
    "Siren": ["siren", "emergency vehicle"],
    "Alarm": ["alarm", "buzzer", "smoke detector"],
    "Gunshot": ["gunshot", "gunfire", "machine gun", "cap gun", "artillery"],
    "Explosion": ["explosion", "boom", "blast"],
    "Glass breaking": ["glass", "shatter"],
    "Dog": ["dog", "bark", "growl", "howl", "bow-wow"],
    "Screaming / shouting": ["scream", "shout", "yell", "bellow"],
    "Crying": ["crying", "sob", "whimper", "baby cry"],
    "Traffic / vehicles": ["traffic", "vehicle", "car", "truck", "motor", "engine", "bus", "motorcycle"],
    "Construction": ["jackhammer", "power tool", "drill", "hammer", "saw"],
    "Door / impact": ["door", "slam", "knock", "thump", "bang"],
    "Footsteps": ["footstep", "walk"],
    "Speech": ["speech", "conversation", "narration", "child speech"],
    "Silence": ["silence"],
}

CRITICAL_CATEGORIES = {"Gunshot", "Glass breaking", "Explosion", "Screaming / shouting"}


def _label_to_category(label: str):
    low = label.lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(kw in low for kw in keywords):
            return category
    return None


class YamnetClassifier:
    """Cache-friendly wrapper around the YAMNet TF-Hub model."""

    def __init__(self):
        import tensorflow_hub as hub

        self.model = hub.load(YAMNET_HANDLE)
        self.class_names = self._load_class_names()

    def _load_class_names(self):
        import tensorflow as tf

        class_map_path = self.model.class_map_path().numpy()
        names = []
        with tf.io.gfile.GFile(class_map_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                names.append(row["display_name"])
        return names

    def classify(self, wav_data: np.ndarray):
        """
        Run YAMNet over the full clip.
        Returns: frame_times, frame_top_label, frame_top_score, mean_scores, scores_np
        """
        scores, _embeddings, _spectrogram = self.model(wav_data)
        scores_np = scores.numpy()
        top_idx = scores_np.argmax(axis=1)
        top_score = scores_np[np.arange(len(top_idx)), top_idx]
        frame_labels = [self.class_names[i] for i in top_idx]
        frame_times = np.arange(len(frame_labels)) * FRAME_HOP_SECONDS
        mean_scores = scores_np.mean(axis=0)
        return frame_times, frame_labels, top_score, mean_scores, scores_np

    def top_n_overall(self, mean_scores, n=10):
        top_idx = np.argsort(mean_scores)[::-1][:n]
        return [(self.class_names[i], float(mean_scores[i])) for i in top_idx]


def _category_score_matrix(scores_np, class_names):
    category_indices = {cat: [] for cat in CATEGORY_KEYWORDS}
    for idx, name in enumerate(class_names):
        cat = _label_to_category(name)
        if cat:
            category_indices[cat].append(idx)
    return {
        cat: scores_np[:, idxs].max(axis=1)
        for cat, idxs in category_indices.items()
        if idxs
    }


def extract_events(frame_times, scores_np, class_names, score_threshold=0.15, min_frames=1):
    """
    Collapse per-category, per-frame scores into timeline events, allowing
    multiple overlapping categories to be active at the same time (e.g.
    "Speech" and "Siren" both active during the same window).
    """
    category_scores = _category_score_matrix(scores_np, class_names)
    events = []

    for category, score_series in category_scores.items():
        active = score_series >= score_threshold
        i, n = 0, len(active)
        while i < n:
            if not active[i]:
                i += 1
                continue
            j = i
            while j < n and active[j]:
                j += 1
            duration_frames = j - i
            if duration_frames >= min_frames or category in CRITICAL_CATEGORIES:
                events.append({
                    "category": category,
                    "start": float(frame_times[i]),
                    "end": float(frame_times[j - 1] + FRAME_HOP_SECONDS),
                    "max_score": float(score_series[i:j].max()),
                    "raw_labels": [category],
                })
            i = j

    events.sort(key=lambda e: e["start"])
    return events


def guess_environment(events):
    """Cheap local heuristic; Claude refines this further using full context."""
    categories_present = {e["category"] for e in events}
    if "Traffic / vehicles" in categories_present:
        return "Likely outdoors / near a road (traffic detected)"
    if "Dog" in categories_present and "Construction" not in categories_present:
        return "Possibly residential, indoors or in a yard (animal sounds detected)"
    if "Construction" in categories_present:
        return "Likely outdoors near construction or industrial activity"
    if not categories_present or categories_present == {"Speech"}:
        return "Likely a quiet indoor space (little background sound detected)"
    return "Mixed environment — see detected background sounds"
