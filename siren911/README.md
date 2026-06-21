# 🚨 SIREN — Smart Incident Recognition & Emergency Network

**An AI co-pilot for 911 dispatchers.** Upload an emergency call recording and
get back a structured, triage-ready incident report: transcript, background
sound events, an acoustic panic score, threat assessment, and a recommended
priority — all generated automatically and laid out for a human dispatcher to
review in seconds, not minutes.

> ⚠️ **This is a hackathon prototype.** It's decision-support only, not a
> replacement for trained dispatcher judgment, and it should only ever be run
> on audio you're authorized to process (e.g. licensed/public research
> datasets like the Kaggle 911-call corpus this was built against) — never on
> live calls without proper authorization and consent.

---

## Why "SIREN"

It's a literal feature: the system detects sirens (and gunshots, glass
breaking, alarms, dogs, traffic, screaming...) in the background of a call.
It also reads well as an acronym: **S**mart **I**ncident **R**ecognition &
**E**mergency **N**etwork.

## What it does

1. **Loads** any call recording (mp3/wav/m4a/flac/ogg) with `librosa`.
2. **Transcribes** it — locally and free with `faster-whisper`, or via
   Deepgram if you have an API key (more accurate, gives speaker
   diarization).
3. **Classifies background sound** with Google's YAMNet, frame by frame,
   *per category* — not just "whatever's loudest" — so a quiet siren under a
   caller's voice still gets flagged. See "A design note" below for why this
   matters.
4. **Scores caller panic** (0–10) using a transparent, explainable blend of
   speech rate, vocal energy variability, pitch variability, repeated
   phrases, and emotional language — with the full breakdown shown, not a
   black-box number.
5. **Reasons over everything with Claude**: incident type, threat level,
   why, victim/suspect info, environment guess, a merged timeline, and an
   explicit confidence + human-review note (Claude is instructed to say
   "I'm not sure" rather than hallucinate certainty).
6. **Triages multiple calls at once** in a Dispatch Queue view, sorted by
   threat level — because real dispatchers manage a queue, not one call.

## Architecture

```
Audio file
   │
   ├─► librosa  ───────────► waveform / FFT spectrum / spectrogram
   │
   ├─► faster-whisper or Deepgram ─► timestamped transcript
   │
   ├─► YAMNet (TF-Hub) ───────────► per-frame, per-category sound scores
   │        └─► extract_events()  ─► background sound timeline
   │
   ├─► panic_detector.py ─────────► explainable 0–10 panic score
   │
   └─► Claude (Anthropic API) ────► threat assessment + timeline + report
            (sees transcript + sound events + panic breakdown)
                  │
                  ▼
          Streamlit dashboard (app.py)
```

## A design note: why we don't filter audio to find "background noise"

Our first instinct was to low-pass filter the audio (e.g. keep only <300 Hz)
to try to isolate background sound from speech. **This doesn't work well** —
speech and background sounds overlap across almost the whole frequency
range, so hard-zeroing FFT bins mostly distorts the signal rather than
separating sources (you'll see a `lowpass_naive()` helper in
`sentinel_core/audio_analysis.py`, kept around for the spectrum demo, with a
docstring explaining exactly why it's not used in the real pipeline).

The fix: run YAMNet on the **original, full-bandwidth audio**, and instead of
only taking each frame's single top-1 label (which is almost always
"Speech" — it's usually the loudest thing), score *every category of
interest independently per frame*. That's what `extract_events()` in
`sentinel_core/sound_classifier.py` does, and it's why SIREN can flag a siren
or alarm happening **underneath** a caller talking, not just whatever sound
wins a popularity contest at each instant.

## Setup

```bash
git clone <this folder, or unzip it>
cd siren911
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env   # optional — or just paste keys into the sidebar
streamlit run app.py
```

Open the local URL Streamlit prints (usually `http://localhost:8501`).

**You need:**
- An [Anthropic API key](https://console.anthropic.com) for the threat
  assessment / report step (many hackathons hand these out via sponsor
  credits — check if yours does).
- *Nothing else is required.* Transcription defaults to local
  `faster-whisper`, so the demo works even with no internet and no Deepgram
  key. Toggle Deepgram in the sidebar if you want diarized, higher-accuracy
  transcripts and have a key.

**Troubleshooting:**
- TensorFlow installs most reliably on Python 3.10–3.11. If `pip install` for
  `tensorflow` fails on your Python version, create the venv with a pinned
  earlier Python (`python3.11 -m venv venv`).
- First YAMNet run downloads the model from TF-Hub — needs internet once,
  then it's cached.
- First `faster-whisper` run downloads model weights — same deal, cached
  after that.

## Demo script (≈60 seconds, for judges)

1. Open the app, point out the disclaimer banner (judges like seeing
   responsible-AI framing front and center, not as an afterthought).
2. Upload one dramatic call. Hit Analyze.
3. While it processes: narrate the pipeline ("librosa → YAMNet → Claude").
4. Land on the **Report** tab — lead with threat level + the *why*.
5. Flip to **Sound Timeline** — show a background event (e.g. glass
   breaking, siren) with its timestamp, and mention the overlapping-category
   detection design choice above — it's a good "most technical" talking
   point.
6. Upload a second, calmer call. Show the **Dispatch Queue** view sorting
   both by threat level — this is the "why this matters for a real
   dispatcher" moment.

## Hackathon track alignment

- **Public safety / real-world impact:** decision support for emergency
  dispatch triage.
- **Deepgram:** pluggable cloud transcription with diarization.
- **Anthropic:** the reasoning layer — threat assessment, timeline
  synthesis, structured report generation, with explicit confidence and
  human-review framing.
- **Most technical:** combines speech-to-text, FFT/STFT signal processing,
  multi-label overlapping audio-event classification, an explainable
  heuristic scoring model, and LLM reasoning into one pipeline.

## Known limitations / good "future work" answers for Q&A

- The panic score is a heuristic, not a validated clinical measure — it's
  designed to be transparent and inspectable, not to claim more certainty
  than it has.
- Local `faster-whisper` is slower than a cloud API on CPU-only machines;
  for a live multi-call demo, pre-process calls beforehand if your judging
  laptop is slow.
- `pyin` pitch tracking is the slowest single step on long calls (tens of
  seconds for a 100s+ clip) — fine for a single demo call, would need
  windowing/streaming for true real-time use.
- Next steps we'd build with more time: real source-separation (e.g. a
  speech-enhancement model) instead of frequency masking, a live-streaming
  mode that updates the dashboard as audio arrives, and a PDF export of the
  dispatcher report (the markdown export already in the Report tab is the
  easy first step toward that).

## Project layout

```
siren911/
├── app.py                       # Streamlit UI + pipeline orchestration
├── requirements.txt
├── .env.example
└── sentinel_core/
    ├── audio_analysis.py        # load/denoise/plot, FFT & spectrogram
    ├── sound_classifier.py      # YAMNet wrapper + event timeline extraction
    ├── panic_detector.py        # explainable panic-score heuristic
    ├── transcription.py        # Deepgram + faster-whisper backends
    ├── claude_brain.py          # Claude prompt + structured-JSON parsing
    └── report.py                 # combine signals -> markdown report
```
