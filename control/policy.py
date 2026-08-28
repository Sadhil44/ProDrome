"""Policy table: failure class + confidence -> action.

Deliberately a lookup table, not a learned model - an operator must be able
to answer "why did this restart my pod at 3am," and a table answers that
where a policy network doesn't. See docs/guides/sadhil.md Part 4.1 and
PRD.md S7.2.

Owned by Sadhil (the table's content and thresholds); consumed by Shravan's
controller loop. Don't change the values here without telling him.
"""

POLICY = {
    "CPU_HOG": {"action": "scale_out", "min_confidence": 0.50},
    "MEMORY_LEAK": {"action": "rolling_restart", "min_confidence": 0.80},
    "DISK_STRESS": {"action": "alert_only", "min_confidence": 0.90},
    "NORMAL": {"action": "nothing", "min_confidence": None},
    "UNKNOWN": {"action": "nothing", "min_confidence": None},
}

ABSTENTION_FLOOR = 0.50


def decide(predicted_class: str, confidence: float) -> str:
    """Map a classifier prediction to an action, per the table above."""
    if confidence < ABSTENTION_FLOOR:
        predicted_class = "UNKNOWN"

    entry = POLICY.get(predicted_class, POLICY["UNKNOWN"])

    if entry["min_confidence"] is not None and confidence < entry["min_confidence"]:
        return "nothing"

    return entry["action"]
