# Policy Table

POLICY = {
    "CPU_HOG": {"action": "scale_out", "min_confidence": 0.50},
    "MEMORY_LEAK": {"action": "rolling_restart", "min_confidence": 0.80},
    "DISK_STRESS": {"action": "alert_only", "min_confidence": 0.90},
    "NORMAL": {"action": "nothing", "min_confidence": None},
    "UNKNOWN": {"action": "nothing", "min_confidence": None},
}

ABSTENTION_FLOOR = 0.50


def decide(predicted_class, confidence):
    if confidence < ABSTENTION_FLOOR:
        predicted_class = "UNKNOWN"

    entry = POLICY.get(predicted_class, POLICY["UNKNOWN"])

    if entry["min_confidence"] is not None and confidence < entry["min_confidence"]:
        return "nothing"

    return entry["action"]
