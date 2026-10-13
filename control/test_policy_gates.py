"""Are the policy table's per-action confidence gates actually reachable?

`policy.py` is a hand-written lookup table and control/CLAUDE.md says the
per-action thresholds differ "on purpose, because being wrong costs different
amounts", with the intended consequence that "early in a failure only the cheap
hedging action is available, and the expensive one unlocks as the diagnosis
firms up".

That intent is only real if each gate can actually block something. A row whose
`min_confidence` equals `ABSTENTION_FLOOR` has a gate that no input can ever
trip: anything below the floor already became UNKNOWN, and anything at or above
it clears the row's own threshold too. The gate is dead code, and the class is
effectively unthresholded while still *looking* thresholded in the table.

These tests are owned by diagnosis (policy.py is diagnosis's file per
control/CLAUDE.md), and they deliberately do NOT assert a preferred threshold
value. Picking one is a project decision, raised on crosstalk/decisions. What
they pin is the structural property: a gate that cannot fire is a defect
regardless of what the right number turns out to be.
"""

import pytest

from control.policy import ABSTENTION_FLOOR, POLICY, decide

# Every class whose row claims a per-action threshold.
GATED = sorted(name for name, row in POLICY.items() if row["min_confidence"] is not None)

# Actions that change the cluster, as opposed to telling a human about it.
MUTATING = {"scale_out", "rolling_restart", "restart"}


def _blocked_band(label):
    """Confidences that clear abstention but are still refused for `label`."""
    return [
        c / 100
        for c in range(int(ABSTENTION_FLOOR * 100), 101)
        if decide(label, c / 100) == "nothing"
    ]


def test_the_abstention_floor_is_what_turns_a_low_confidence_into_unknown():
    """Baseline, so the tests below are not vacuous if the floor moves."""
    assert decide("CPU_HOG", ABSTENTION_FLOOR - 0.01) == "nothing"
    assert POLICY["UNKNOWN"]["action"] == "nothing"


@pytest.mark.parametrize("label", GATED)
def test_each_gated_class_authorizes_its_action_somewhere(label):
    """A gate that blocks everything would be a different defect: a lever that
    exists in the table and can never be pulled.
    """
    action = POLICY[label]["action"]
    assert decide(label, 1.0) == action, f"{label} can never reach {action}"


@pytest.mark.parametrize("label", GATED)
def test_each_gated_class_has_a_reachable_gate(label):
    """The finding, recorded as a strict-ish xfail so it cannot rot.

    CPU_HOG is min_confidence 0.50 against an ABSTENTION_FLOOR of 0.50, so its
    per-action gate is dead code: the lowest confidence that authorizes
    `scale_out` is exactly the lowest confidence that is not abstained on.

    This matters more than a tidiness complaint, because of what the attribution
    work on 2026-10-03 showed about which classes are trustworthy:

        DISK_STRESS  93.0% of attribution mass on its own fs_* metrics
                     -> gate 0.90, action alert_only          (safest action, strictest gate)
        CPU_HOG       6.0% on cpu_cores, ~80% on mem_*/fs_*,
                     and on a real sampled window the largest
                     single contribution was cpu_cores_last
                     NEGATIVE while CPU_HOG was still predicted
                     -> gate 0.50 == floor, action scale_out   (mutating action, no gate)

    So the class the model understands best is held to the highest bar and only
    allowed to alert, while the class it understands worst is held to no bar at
    all and allowed to change the cluster. That ordering is inverted with
    respect to the evidence.

    Marked non-strict on purpose, unlike this project's other recorded findings:
    the correct fix is a threshold value, and that is a project decision on
    crosstalk/decisions rather than something a test should force. When someone
    sets CPU_HOG above the floor this starts passing; make it strict then, so it
    cannot silently regress.
    """
    blocked = _blocked_band(label)
    if not blocked:
        pytest.xfail(
            f"{label}: min_confidence={POLICY[label]['min_confidence']} equals "
            f"ABSTENTION_FLOOR={ABSTENTION_FLOOR}, so its gate is dead code. Raised on "
            f"crosstalk/decisions; the threshold value is a project decision, not a test's."
        )
    assert blocked, (
        f"{label}: min_confidence={POLICY[label]['min_confidence']} equals "
        f"ABSTENTION_FLOOR={ABSTENTION_FLOOR}, so no confidence is ever refused by "
        f"this row's own gate -- it is effectively unthresholded"
    )


def test_a_mutating_action_is_never_the_cheapest_thing_to_unlock():
    """Ground rule: being wrong costs different amounts.

    If the lowest confidence that mutates the cluster is at or below the lowest
    confidence that merely raises an alert, then the table's whole rationale is
    inverted -- the expensive action unlocks before the cheap one.

    Also recorded as an expected failure today: CPU_HOG mutates at 0.50 while
    DISK_STRESS only alerts at 0.90.
    """
    cheapest_mutation = min(
        (row["min_confidence"] for row in POLICY.values()
         if row["action"] in MUTATING and row["min_confidence"] is not None),
        default=None,
    )
    cheapest_alert = min(
        (row["min_confidence"] for row in POLICY.values()
         if row["action"] == "alert_only" and row["min_confidence"] is not None),
        default=None,
    )
    if cheapest_mutation is None or cheapest_alert is None:
        pytest.skip("table has no mutating action or no alert_only action to compare")

    pytest.xfail(
        f"mutating action unlocks at {cheapest_mutation} but alert_only needs "
        f"{cheapest_alert} -- the expensive action is cheaper to trigger than the "
        f"cheap one. Raised on crosstalk/decisions; not fixed unilaterally because "
        f"the threshold value is a project decision."
    ) if cheapest_mutation <= cheapest_alert else None

    assert cheapest_mutation > cheapest_alert
