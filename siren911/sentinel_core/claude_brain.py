"""
LLM reasoning layer: turns raw signals (transcript + detected background
sounds + acoustic panic score) into dispatcher-facing intelligence using
Claude.
"""

import json
from textwrap import dedent

DEFAULT_MODEL = "claude-sonnet-4-6"

SYSTEM_PROMPT = dedent("""\
    You are an assistant that helps 911 dispatchers triage emergency calls.
    You are a DECISION-SUPPORT tool, not a replacement for a trained
    dispatcher. Never invent facts that aren't supported by the transcript
    or detected audio events. When information is missing or ambiguous,
    say so explicitly instead of guessing with false confidence.

    You will be given:
      - a call transcript (may contain transcription errors)
      - a list of background sound events detected by an audio classifier,
        with start/end timestamps in seconds and confidence scores
      - an acoustic "panic score" (0-10) for the caller's vocal stress, with
        a breakdown of the signals that produced it

    Respond with ONLY valid JSON (no prose, no markdown fences) matching
    this schema:

    {
      "incident_type": str,
      "threat_level": "Low" | "Medium" | "High" | "Critical",
      "threat_reasons": [str, ...],
      "victims_estimate": str,
      "suspect_info": str,
      "environment_guess": str,
      "recommended_priority": str,
      "timeline": [{"time_s": number, "event": str, "source": "transcript"|"audio"|"inference"}, ...],
      "confidence": "Low" | "Medium" | "High",
      "human_review_notes": str
    }
""")


def build_user_prompt(transcript_text, sound_events, panic_result, duration_s):
    events_str = "\n".join(
        f"- {e['category']} from {e['start']:.1f}s to {e['end']:.1f}s "
        f"(confidence {e['max_score']:.2f})"
        for e in sound_events
    ) or "- (no notable background events detected)"

    return dedent(f"""\
        CALL DURATION: {duration_s:.1f} seconds

        TRANSCRIPT:
        {transcript_text or "(no transcript available)"}

        DETECTED BACKGROUND SOUND EVENTS:
        {events_str}

        PANIC SCORE: {panic_result['panic_score']} / 10
        Breakdown: {json.dumps(panic_result['breakdown'])}

        Produce the JSON incident report described in your instructions.
    """)


def analyze_call(client, transcript_text, sound_events, panic_result, duration_s, model=DEFAULT_MODEL):
    """
    client: an instantiated anthropic.Anthropic client.
    Returns a dict matching the schema in SYSTEM_PROMPT, or a dict with an
    'error' key if Claude's response couldn't be parsed as JSON.
    """
    user_prompt = build_user_prompt(transcript_text, sound_events, panic_result, duration_s)

    response = client.messages.create(
        model=model,
        max_tokens=1500,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )

    raw_text = "".join(
        block.text for block in response.content if getattr(block, "type", None) == "text"
    ).strip()

    # Claude is instructed to return raw JSON, but strip fences defensively.
    cleaned = raw_text.strip("`")
    if cleaned.lower().startswith("json"):
        cleaned = cleaned[4:].strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {
            "error": "Claude's response could not be parsed as JSON.",
            "raw_response": raw_text,
        }
