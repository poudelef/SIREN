"""
SIREN — Smart Incident Recognition & Emergency Network
AI co-pilot for 911 dispatchers: turns raw call audio into structured,
triage-ready incident intelligence.

Run with:  streamlit run app.py

NOTE: every step in process_call() prints a timestamped line to the
terminal (the one running `streamlit run app.py`), e.g.

    [   0.0s] call_1.mp3: starting
    [   0.4s] call_1.mp3: audio loaded (101.7s @ 16000Hz)
    [   0.4s] call_1.mp3: transcribing via Whisper...
    [  42.1s] call_1.mp3: transcription done (187 words)
    ...

Watch that terminal while the UI looks "stuck" — it tells you exactly
which step is slow.
"""

import os
import tempfile
import time

from dotenv import load_dotenv
load_dotenv()

import pandas as pd
import streamlit as st

from sentinel_core.audio_analysis import (
    load_audio, denoise_audio, compute_rms_envelope, compute_pitch_stats,
    plot_waveform, plot_spectrum, plot_spectrogram,
    plot_voice_energy_timeline, plot_sound_event_timeline, plot_sound_duration_bar,  # ← added
)
from sentinel_core.sound_classifier import YamnetClassifier, extract_events, guess_environment
from sentinel_core.panic_detector import estimate_panic
from sentinel_core.transcription import transcribe_with_deepgram, segments_to_text
from sentinel_core.claude_brain import analyze_call, DEFAULT_MODEL
from sentinel_core.report import build_report, to_markdown

st.set_page_config(page_title="SIREN — 911 Call Intelligence", page_icon="🚨", layout="wide")

THREAT_RANK = {"Critical": 4, "High": 3, "Medium": 2, "Low": 1, "Unknown": 0}
THREAT_COLOR = {"Critical": "🔴", "High": "🟠", "Medium": "🟡", "Low": "🟢", "Unknown": "⚪"}


@st.cache_resource(show_spinner="Loading YAMNet (first run only)…")
def get_yamnet():
    print("[yamnet] loading model from TF-Hub (first run downloads it — can take a while)...", flush=True)
    model = YamnetClassifier()
    print("[yamnet] model loaded", flush=True)
    return model


@st.cache_data(show_spinner=False)
def cache_audio_analysis(audio_path, sr, wav_data):
    """Cache expensive audio computations."""
    print(f"[{os.path.basename(audio_path)}]   -> computing RMS envelope...", flush=True)
    rms_times, rms = compute_rms_envelope(wav_data, sr)
    print(f"[{os.path.basename(audio_path)}]   -> RMS done, now pitch tracking (pyin — the slow one)...", flush=True)
    mean_f0, std_f0, voiced_fraction = compute_pitch_stats(wav_data, sr)
    print(f"[{os.path.basename(audio_path)}]   -> pitch tracking done", flush=True)
    return rms_times, rms, mean_f0, std_f0, voiced_fraction


@st.cache_data(show_spinner=False)
def cache_plots(wav_data, sr):
    """Cache plot generation to avoid regeneration."""
    return (
        plot_waveform(wav_data, sr),
        plot_spectrum(wav_data, sr),
        plot_spectrogram(wav_data, sr),
    )


def save_upload_to_tmp(uploaded_file) -> str:
    suffix = os.path.splitext(uploaded_file.name)[1] or ".wav"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.getvalue())
    tmp.flush()
    return tmp.name


def process_call(uploaded_file, settings) -> dict:
    t0 = time.time()
    name = uploaded_file.name

    def log(msg):
        print(f"[{time.time() - t0:6.1f}s] {name}: {msg}", flush=True)

    log("starting")

    audio_path = save_upload_to_tmp(uploaded_file)
    log("saved upload to temp file")

    wav_data, sr, duration = load_audio(audio_path)
    log(f"audio loaded ({duration:.1f}s @ {sr}Hz)")

    if settings["denoise"]:
        log("running noise reduction...")
        wav_data = denoise_audio(wav_data, sr)
        log("noise reduction done")

    # --- transcription ---
    log("transcribing via Deepgram...")

    if not settings["deepgram_key"]:
        raise ValueError("Deepgram API key is required.")

    segments = transcribe_with_deepgram(
        audio_path,
        settings["deepgram_key"]
    )

    transcript_text = segments_to_text(segments)
    log(f"transcription done ({len(transcript_text.split())} words)")

    # --- background sound events ---
    log("loading YAMNet / running sound classification...")
    yamnet = get_yamnet()
    frame_times, frame_labels, frame_top_scores, mean_scores, scores_np = yamnet.classify(wav_data)
    log("YAMNet inference complete")
    events = extract_events(frame_times, scores_np, yamnet.class_names, score_threshold=settings["event_threshold"])
    environment_guess_local = guess_environment(events)
    top10 = yamnet.top_n_overall(mean_scores, n=10)
    log(f"found {len(events)} background sound event(s)")

    # --- panic score ---
    log("computing RMS + pitch for panic score (pitch tracking is the slowest step on long clips)...")
    rms_times, rms, mean_f0, std_f0, voiced_fraction = cache_audio_analysis(audio_path, sr, wav_data)
    panic_result = estimate_panic(transcript_text, duration, rms, mean_f0, std_f0)
    log(f"panic score: {panic_result['panic_score']}/10")

    # --- Claude reasoning layer ---
    claude_result = {"error": "Anthropic API key not provided — add it in the sidebar to enable AI threat assessment."}
    if settings["anthropic_key"]:
        log(f"calling Claude ({settings['claude_model']}) for threat assessment...")
        import anthropic
        client = anthropic.Anthropic(api_key=settings["anthropic_key"])
        claude_result = analyze_call(
            client, transcript_text, events, panic_result, duration, model=settings["claude_model"]
        )
        log("Claude response received")
    else:
        log("no Anthropic key provided — skipping Claude step")

    report = build_report(uploaded_file.name, duration, transcript_text, events, panic_result, claude_result)
    report["_audio_path"] = audio_path
    report["_wav_data"] = wav_data
    report["_sr"] = sr
    report["_segments"] = segments
    report["_top10_sounds"] = top10
    report["_environment_guess_local"] = environment_guess_local
    report["_rms_times"] = rms_times
    report["_rms"] = rms
    report["_frame_times"] = frame_times
    report["_frame_labels"] = frame_labels
    report["_frame_top_scores"] = frame_top_scores

    log(f"DONE — total {time.time() - t0:.1f}s")
    return report

def _fact(col, label, value):
    with col:
        st.caption(f" {label}")
        st.markdown(f"**{value}**")

def render_call(report: dict):
    ai = report.get("ai_assessment", {}) or {}
    threat = ai.get("threat_level", "Unknown")

    st.markdown(
        f"### {THREAT_COLOR.get(threat, '⚪')} {report['call_name']} "
        f"— Threat: **{threat}** | Panic: **{report['panic_analysis']['panic_score']}/10**"
    )

    tabs = st.tabs([
        "Report", "Transcript", "Audio Visuals",
        "Sound Timeline", "Panic Detail", "Raw JSON",
    ])

    with tabs[0]:
        if "error" in ai:
            st.warning(ai["error"])
        else:
            # --- one-line summary banner ---
            st.markdown(f"#### {ai.get('one_line_summary', '—')}")
            st.divider()

            # --- key facts grid: scannable in seconds ---
            r1 = st.columns(4)
            _fact(r1[0], "Location", ai.get("location", "—"))
            _fact(r1[1], "Environment", ai.get("environment", "—"))
            _fact(r1[2], "People", ai.get("people_count", "—"))
            names = ai.get("people_names", [])
            _fact(r1[3], "Names mentioned", ", ".join(names) if names else "None")

            r2 = st.columns(4)
            _fact(r2[0], "Gender(s)", ai.get("genders", "—"))
            _fact(r2[1], "Weapon", ai.get("weapon", "—"))
            _fact(r2[2], "Suspect status", ai.get("suspect_status", "—"))
            _fact(r2[3], "Injuries", ai.get("injuries", "—"))

            st.divider()

            # --- priority action, front and center since it's the call to action ---
            priority_box = {"Critical": st.error, "High": st.warning}.get(threat, st.info)
            priority_box(f"**{ai.get('recommended_priority', '—')}**")

            c1, c2 = st.columns(2)
            c1.metric("Incident Type", ai.get("incident_type", "—"))
            c2.metric("AI Confidence", ai.get("confidence", "—"))

            with st.expander("Why this threat level"):
                for r in ai.get("threat_reasons", []):
                    st.markdown(f"- {r}")
                st.markdown(f"**Victims (estimate):** {ai.get('victims_estimate', '—')}")
                st.markdown(f"**Suspect info:** {ai.get('suspect_info', '—')}")
                st.markdown(
                    f"**Environment guess (detail):** "
                    f"{ai.get('environment_guess', report['_environment_guess_local'])}"
                )

            with st.expander("Merged timeline"):
                for item in ai.get("timeline", []):
                    st.markdown(
                        f"`{item.get('time_s', 0):>6.1f}s` · "
                        f"[{item.get('source', '?')}] {item.get('event', '')}"
                    )

            with st.expander("Human review notes", expanded=False):
                st.info(ai.get("human_review_notes", "None"))

        md_report = to_markdown(report)
        st.download_button("Download report (Markdown)", md_report, file_name=f"{report['call_name']}_report.md")

    with tabs[1]:
        st.audio(report["_audio_path"])
        for seg in report["_segments"]:
            st.markdown(f"`{seg.start:>6.1f}s` **{seg.speaker}:** {seg.text}")

    with tabs[2]:
        # waveform_fig, spectrum_fig, spectrogram_fig = cache_plots(report["_wav_data"], report["_sr"])
        st.caption(
            "Plain-language view of the call's audio — built for a quick read by "
            "a dispatcher or responding officer, not an audio engineer."
        )
        st.markdown("**Caller voice energy over time**")
        st.pyplot(plot_voice_energy_timeline(
            report["_rms_times"], report["_rms"], report["duration_seconds"]
        ))

        st.markdown("**When each background sound happened**")
        st.pyplot(plot_sound_event_timeline(
            report["background_sound_events"], report["duration_seconds"]
        ))

        st.markdown("**How much of the call had each background sound**")
        st.pyplot(plot_sound_duration_bar(report["background_sound_events"]))

        with st.expander("🔬 Technical audio views (waveform / FFT / spectrogram)"):
            st.pyplot(plot_waveform(report["_wav_data"], report["_sr"]))
            st.pyplot(plot_spectrum(report["_wav_data"], report["_sr"]))
            st.pyplot(plot_spectrogram(report["_wav_data"], report["_sr"]))

    with tabs[3]:
        if report["background_sound_events"]:
            for e in report["background_sound_events"]:
                st.markdown(
                    f"`{e['start']:>6.1f}s–{e['end']:>6.1f}s` **{e['category']}** "
                    f"(confidence {e['max_score']:.2f})"
                )
        else:
            st.write("No notable background events above threshold.")

        with st.expander("Top 10 overall sound classes (whole clip, mean score)"):
            for label, score in report["_top10_sounds"]:
                st.write(f"{label}: {score:.3f}")

        with st.expander("Raw YAMNet top-1 label per frame (debug view)"):
            st.dataframe(
                pd.DataFrame({
                    "time_s": report["_frame_times"],
                    "top_label": report["_frame_labels"],
                    "score": report["_frame_top_scores"],
                }),
                use_container_width=True,
                height=250,
            )

    with tabs[4]:
        st.write(f"**Panic score: {report['panic_analysis']['panic_score']} / 10**")
        st.bar_chart(pd.Series(report["panic_analysis"]["breakdown"]))
        st.caption(
            "Heuristic blend of speech rate, vocal energy variability, pitch "
            "variability, repeated phrases, and emotional keywords. Not a "
            "clinical stress measurement — use as a triage signal only."
        )
        st.markdown("**Vocal energy (RMS) over time:**")
        st.line_chart(pd.DataFrame({"RMS energy": report["_rms"]}, index=report["_rms_times"]))

    with tabs[5]:
        st.json({k: v for k, v in report.items() if not k.startswith("_")})


def main():
    st.title("🚨 SIREN")
    st.caption("Smart Incident Recognition & Emergency Network — AI co-pilot for 911 dispatchers")

    st.warning(
        "**Prototype — decision support only.** Every output here must be "
        "verified by a trained dispatcher before action is taken. Use only "
        "with audio you're authorized to process (e.g. licensed/public "
        "research datasets) — never live calls without proper authorization.",
        icon="⚠️",
    )

    with st.sidebar:
        st.header("⚙️ Settings")
        anthropic_key = st.text_input(
            "Anthropic API key", type="password", value=os.environ.get("ANTHROPIC_API_KEY", "")
        )
        claude_model = st.selectbox(
            "Claude model",
            [DEFAULT_MODEL, "claude-haiku-4-5-20251001", "claude-opus-4-7"],
            index=0,
        )

        st.subheader("🎤 Deepgram")

        deepgram_key = st.text_input(
            "Deepgram API Key",
            type="password",
            value=os.environ.get("DEEPGRAM_API_KEY", "")
        )

        st.divider()
        denoise = st.checkbox("Apply noise reduction before transcription", value=False)
        event_threshold = st.slider("Sound event confidence threshold", 0.05, 0.5, 0.15, 0.05)

    if "reports" not in st.session_state:
        st.session_state.reports = []

    uploaded_files = st.file_uploader(
        "Upload one or more 911 call recordings",
        type=["wav", "mp3", "m4a", "flac", "ogg"],
        accept_multiple_files=True,
    )

    if uploaded_files and st.button("🔍 Analyze call(s)", type="primary"):
        settings = dict(
            anthropic_key=anthropic_key, claude_model=claude_model,
            deepgram_key=deepgram_key,
            denoise=denoise, event_threshold=event_threshold,
        )
        progress = st.progress(0.0, text="Starting…")
        print(f"\n=== Analyzing {len(uploaded_files)} call(s) ===", flush=True)
        for i, f in enumerate(uploaded_files):
            print(f"--- File {i + 1}/{len(uploaded_files)}: {f.name} ---", flush=True)
            progress.progress(i / len(uploaded_files), text=f"Processing {f.name}…")
            try:
                report = process_call(f, settings)
                st.session_state.reports.append(report)
            except Exception as exc:
                print(f"!!! {f.name} FAILED: {exc}", flush=True)
                st.error(f"Failed to process {f.name}: {exc}")
        progress.progress(1.0, text="Done.")
        print("=== Batch complete ===\n", flush=True)

    if st.session_state.reports:
        if len(st.session_state.reports) > 1:
            st.subheader("📊 Dispatch Queue")
            rows = []
            for r in st.session_state.reports:
                ai = r.get("ai_assessment", {}) or {}
                rows.append({
                    "Call": r["call_name"],
                    "Threat": ai.get("threat_level", "Unknown"),
                    "Priority": ai.get("recommended_priority", "—"),
                    "Panic": r["panic_analysis"]["panic_score"],
                    "Incident Type": ai.get("incident_type", "—"),
                })
            rows.sort(key=lambda row: THREAT_RANK.get(row["Threat"], 0), reverse=True)
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            st.divider()

        st.subheader("📞 Call Details")
        names = [r["call_name"] for r in st.session_state.reports]
        choice = st.selectbox("Select a call to inspect", names, index=len(names) - 1)
        selected = next(r for r in st.session_state.reports if r["call_name"] == choice)
        render_call(selected)

        if st.button("🗑️ Clear all processed calls"):
            st.session_state.reports = []
            st.rerun()
    else:
        st.info("Upload a recording and click **Analyze** to get started.")


if __name__ == "__main__":
    main()
