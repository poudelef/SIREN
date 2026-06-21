"""
Combine every signal into one incident report object, and render it as
clean markdown for download / display.
"""

from datetime import datetime, timezone


def build_report(call_name, duration_s, transcript_text, sound_events, panic_result, claude_result):
    return {
        "call_name": call_name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": round(duration_s, 1),
        "transcript": transcript_text,
        "background_sound_events": sound_events,
        "panic_analysis": panic_result,
        "ai_assessment": claude_result,
    }


def to_markdown(report: dict) -> str:
    ai = report.get("ai_assessment", {}) or {}
    lines = [
        f"# Incident Report — {report['call_name']}",
        f"_Generated {report['generated_at']} • AI-assisted, requires human dispatcher review_",
        "",
        f"**Duration:** {report['duration_seconds']}s &nbsp;&nbsp;|&nbsp;&nbsp; "
        f"**Panic Score:** {report['panic_analysis']['panic_score']}/10",
        "",
    ]

    if "error" in ai:
        lines += ["## AI Assessment", "_Could not parse AI assessment for this call._", ""]
    else:
        lines += [
            f"## Incident Type: {ai.get('incident_type', 'Unknown')}",
            f"**Threat Level:** {ai.get('threat_level', 'Unknown')}  ",
            f"**Recommended Priority:** {ai.get('recommended_priority', 'Unknown')}  ",
            f"**Confidence:** {ai.get('confidence', 'Unknown')}",
            "",
            "### Why",
        ]
        for reason in ai.get("threat_reasons", []):
            lines.append(f"- {reason}")
        lines += [
            "",
            f"**Victims (estimate):** {ai.get('victims_estimate', 'Unknown')}  ",
            f"**Suspect info:** {ai.get('suspect_info', 'Unknown')}  ",
            f"**Environment guess:** {ai.get('environment_guess', 'Unknown')}",
            "",
            "### Timeline",
        ]
        for item in ai.get("timeline", []):
            lines.append(f"- `{item.get('time_s', 0):>6.1f}s` [{item.get('source', '?')}] {item.get('event', '')}")
        lines += [
            "",
            f"**Human review notes:** {ai.get('human_review_notes', '')}",
            "",
        ]

    lines += ["## Background Sound Events"]
    if report["background_sound_events"]:
        for e in report["background_sound_events"]:
            lines.append(
                f"- **{e['category']}** {e['start']:.1f}s–{e['end']:.1f}s "
                f"(confidence {e['max_score']:.2f})"
            )
    else:
        lines.append("- None detected above threshold.")

    lines += ["", "## Full Transcript", "", report["transcript"] or "_(none)_"]

    return "\n".join(lines)
