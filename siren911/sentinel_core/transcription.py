"""
Speech-to-text backends.

Two interchangeable engines so the demo never depends on having both an
internet connection AND a paid API key at the same time:

  - Deepgram (cloud, high accuracy, speaker diarization) -- needs an API key
  - faster-whisper (local, free, works fully offline once the model weights
    are cached on disk) -- the safer default for a live hackathon demo

Both return the same TranscriptSegment shape so the rest of the app doesn't
care which one ran.
"""

from dataclasses import dataclass
from typing import List


@dataclass
class TranscriptSegment:
    start: float
    end: float
    speaker: str
    text: str


def segments_to_text(segments: List[TranscriptSegment]) -> str:
    return " ".join(s.text for s in segments).strip()


def transcribe_with_deepgram(audio_path: str, api_key: str) -> List[TranscriptSegment]:
    from deepgram import DeepgramClient, PrerecordedOptions

    client = DeepgramClient(api_key)
    with open(audio_path, "rb") as f:
        buffer_data = f.read()

    options = PrerecordedOptions(
        model="nova-2",
        smart_format=True,
        diarize=True,
        punctuate=True,
        utterances=True,
    )
    response = client.listen.prerecorded.v("1").transcribe_file(
        {"buffer": buffer_data}, options
    )

    segments = []
    utterances = getattr(response.results, "utterances", None) or []
    for utt in utterances:
        segments.append(
            TranscriptSegment(
                start=float(utt.start),
                end=float(utt.end),
                speaker=f"Speaker {utt.speaker}",
                text=utt.transcript,
            )
        )
    if not segments:
        # fall back to the flat transcript if utterance-level data is empty
        text = response.results.channels[0].alternatives[0].transcript
        segments = [TranscriptSegment(0.0, 0.0, "Caller", text)]
    return segments


# _whisper_model = None
# _whisper_model_size = None


# def _load_whisper_model(model_size: str):
#     """Try GPU first (huge speedup if an NVIDIA GPU + CUDA/cuDNN are present),
#     fall back to CPU automatically if that fails for any reason."""
#     from faster_whisper import WhisperModel
#     try:
#         return WhisperModel(model_size, device="auto", compute_type="default")
#     except Exception:
#         return WhisperModel(model_size, device="cpu", compute_type="int8")


# def transcribe_with_whisper(wav_data, sr: int, model_size: str = "small") -> List[TranscriptSegment]:
#     """wav_data must already be mono float32 at 16 kHz (see audio_analysis.load_audio)."""
#     global _whisper_model, _whisper_model_size

#     if _whisper_model is None or _whisper_model_size != model_size:
#         _whisper_model = _load_whisper_model(model_size)
#         _whisper_model_size = model_size

#     segments_out, _info = _whisper_model.transcribe(wav_data, language="en", vad_filter=True)
#     segments = []
#     for seg in segments_out:
#         segments.append(
#             TranscriptSegment(start=seg.start, end=seg.end, speaker="Caller", text=seg.text.strip())
#         )
#     return segments
