"""Does CI actually run the tests this repo contains?

For most of this project it did not. `.github/workflows/tests.yml` ran
`pytest tests`, while `control/test_safety_rails.py` (the kill switch, the
cooldown, MAX_REPLICAS, no-mutation-in-shadow), `control/test_policy_gates.py`
and `ml/test_explain.py` all live next to the code they cover and therefore
never ran on a push. Cluster's own handoff put it plainly: "Until testing acts,
nothing stops these rails regressing."

The fix was to widen the invocation rather than move the files -- tests/CLAUDE.md
gives `tests/` to this seat and says other seats' tests live beside their code,
so CI is a consumer of those files, not their owner. The invocation names each
file explicitly, because the workflow installs only what the suite imports and a
directory glob would make that import surface unbounded.

That trade has exactly one failure mode: a seat adds a test file and forgets to
add it to the workflow, so it silently never runs -- which is the original bug,
returning quietly. This file is the guard against that. It is the only test here
that asserts about CI configuration rather than about the pipeline, and it earns
its place by making a silent gap loud.

Deliberately text-parsed rather than YAML-parsed: pyyaml is not in the CI
install list, and adding a dependency so that a test about the dependency list
can run is the wrong direction.
"""

import re

import pytest

from conftest import REPO_ROOT

WORKFLOW = REPO_ROOT / ".github" / "workflows" / "tests.yml"

# Directories that are not part of the repo's source tree.
SKIP_DIRS = {".git", ".venv", ".claude", "__pycache__", ".pytest_cache", "node_modules"}

# Test files CI knowingly does NOT run, and why. An entry here is a statement
# that someone looked at it and decided; an absence is a gap. Keep the reasons
# specific enough that a future reader can tell whether they still hold.
KNOWN_UNENFORCED = {
    "ml/test_restart_suppression.py":
        "reads data/healthy/metrics.parquet, which is gitignored and absent, so it "
        "cannot run from a fresh clone (ground rule 8). It also has no test_ "
        "functions -- it is a main() script -- so collecting it today would add "
        "zero tests. A synthetic fallback would fix both; that is signal's file, "
        "so it is a request on crosstalk, not an edit from here.",
    "collect/test_twoarm.py":
        "imports only pandas/pytest and would run clean, but collect/ and eval/ are "
        "mid-campaign and are being rewritten under us. Making a moving target a "
        "merge gate turns someone else's work-in-progress into everyone's red "
        "build. Asked collect on crosstalk to say when it should be enforced.",
    "eval/test_recovery.py":
        "same as collect/test_twoarm.py: ready to enforce, held back only until the "
        "campaign in collect/ and eval/ settles.",
}


def _pytest_invocation() -> str:
    """The `python -m pytest ...` command the workflow runs, flattened.

    The invocation is a folded YAML block, so it spans several lines at one
    indent; continuation lines are taken while they stay at least that deep.
    """
    lines = WORKFLOW.read_text().splitlines()
    for i, line in enumerate(lines):
        if "python -m pytest" not in line:
            continue
        indent = len(line) - len(line.lstrip())
        parts = [line.strip()]
        for following in lines[i + 1:]:
            if not following.strip():
                break
            if len(following) - len(following.lstrip()) < indent:
                break
            parts.append(following.strip())
        return " ".join(parts)
    raise AssertionError(f"{WORKFLOW.name} no longer runs `python -m pytest` at all")


def _repo_test_files() -> set[str]:
    found = set()
    for path in REPO_ROOT.rglob("test_*.py"):
        if SKIP_DIRS & set(path.relative_to(REPO_ROOT).parts):
            continue
        found.add(path.relative_to(REPO_ROOT).as_posix())
    return found


def test_every_test_file_in_the_repo_is_either_run_by_ci_or_documented_as_not():
    """The guard. A new test file that CI does not run has to say so out loud.

    `pytest tests` ran 83 of the repo's tests and silently skipped 89 more,
    including every assertion standing between an untested model and a real
    cluster. Nothing in the repo reported that. This does.
    """
    invocation = _pytest_invocation()
    present = _repo_test_files()
    covered = {
        path for path in present
        if path.startswith("tests/") or path in invocation
    }

    undocumented = present - covered - set(KNOWN_UNENFORCED)
    assert undocumented == set(), (
        "these test files exist but CI never runs them: "
        f"{sorted(undocumented)}. Add each path to the pytest invocation in "
        f"{WORKFLOW.name}, or to KNOWN_UNENFORCED here with the reason."
    )

    stale = covered & set(KNOWN_UNENFORCED)
    assert stale == set(), (
        f"KNOWN_UNENFORCED still excuses {sorted(stale)}, but CI runs them now. "
        "Drop the entry so the list keeps meaning something."
    )


@pytest.mark.parametrize("path", sorted(KNOWN_UNENFORCED))
def test_the_unenforced_list_only_names_files_that_exist(path):
    """A stale exclusion is how a list like this stops meaning anything."""
    assert (REPO_ROOT / path).exists(), (
        f"{path} is excused from CI but no longer exists -- remove the entry"
    )


def test_ci_runs_the_safety_rails_the_policy_gates_and_the_attribution_identity():
    """Named, not inferred, because these three are the point of the exercise.

    control/test_safety_rails.py is the only thing standing between a model's
    guess and a production cluster; control/test_policy_gates.py records that
    CPU_HOG's confidence gate is unreachable dead code; ml/test_explain.py pins
    the per-prediction attribution identity (mean root probability plus summed
    contributions must reproduce predict_proba exactly). If a future edit
    narrows the invocation, the generic guard above would still pass as long as
    the file were added to KNOWN_UNENFORCED. This one would not.
    """
    invocation = _pytest_invocation()
    for required in (
        "control/test_safety_rails.py",
        "control/test_policy_gates.py",
        "ml/test_explain.py",
    ):
        assert required in invocation, f"CI no longer runs {required}"


def test_ci_does_not_install_the_whole_requirements_file():
    """tests/CLAUDE.md's rule, as a check rather than a note.

    `pip install -r requirements.txt` pulls torch (~2 GB, CUDA wheels by
    default) for ml/cnn.py, which no test imports, plus the kubernetes client,
    which is worth nothing without a cluster -- and installing it would hide the
    very regression test_integration.py's offline-import test exists to catch.
    """
    # Comments are stripped first: the workflow explains at length why it does
    # NOT do this, and a test that cannot tell an explanation from an
    # instruction is a test that fires on its own documentation.
    text = "\n".join(
        line for line in WORKFLOW.read_text().splitlines()
        if not line.lstrip().startswith("#")
    )
    assert "-r requirements.txt" not in text, (
        "CI installs requirements.txt: that is ~2 GB of torch for code no test "
        "touches, and a kubernetes client that would mask a module-scope import "
        "regression in control/controller.py"
    )
    for banned in ("torch", "kubernetes"):
        assert not re.search(rf"pip install[^\n]*\b{banned}\b", text), (
            f"CI installs {banned}; the suite does not import it"
        )
