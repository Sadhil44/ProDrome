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
        "defines NO test_ functions -- it is a main() script with a helper -- so "
        "naming it in the workflow would collect zero tests and report a green "
        "file that asserts nothing, which is worse than a documented gap. The "
        "data half of the old reason has changed and the conclusion has not: "
        "data/healthy/metrics.parquet exists locally now (465 rows) but data/ is "
        "gitignored, so on the fresh clone CI runs from it is still absent, and "
        "main() would raise FileNotFoundError at the first line if it ever "
        "became collectable. Both halves are signal's to fix and were posted to "
        "crosstalk rather than edited from here. What the script actually "
        "checks is NOT unenforced: tests/test_units.py's "
        "test_a_restart_is_suppressed_for_the_full_window_and_then_lifts is the "
        "fresh-clone counterpart and does run in CI. The one half it omits -- "
        "that the restart shape WOULD fire without on_restart -- is a coverage "
        "gap worth closing, NOT a reason to exclude this file: the exclusion "
        "rests entirely on the zero test_ functions above and would still stand "
        "if that half were fully covered. (The old reason claimed that half "
        "'does not hold through update() on any data, synthetic or real'. That "
        "was true of the pre-fix MetricDetector, which appended each tick's "
        "error to the reference BEFORE scoring against it and so could not fire "
        "on a step change of any magnitude. ml/detector.py:135-146 now scores "
        "first and folds the error in afterwards, only when the tick looked "
        "healthy, so the arithmetic that made the claim true is gone.)",
}

# Files whose excuse above is "it defines no test_ functions". That reason is the
# kind that rots silently -- signal adds one assertion, the file becomes worth
# running, and this list goes on excusing it forever. Checked, not trusted.
UNENFORCED_BECAUSE_NO_TESTS = {"ml/test_restart_suppression.py"}


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


@pytest.mark.parametrize("path", sorted(UNENFORCED_BECAUSE_NO_TESTS))
def test_a_file_excused_for_having_no_tests_still_has_none(path):
    """The reason, checked rather than taken on trust.

    Two of the three entries on this list expired quietly in October: the
    campaign that held collect/ and eval/ back finished, and the data file
    ml/test_restart_suppression.py wanted appeared. Nobody was told; the
    exclusions simply went on excusing. The `stale` check above catches a reason
    that expires because CI started running the file, which is the easy
    direction. This catches the other one -- a reason that expires because the
    file changed underneath it.

    Parsed with `ast` rather than imported: importing it would run
    `from ml.detector import ...` and, in the version that still has a main(),
    nothing else -- but the next version might, and a guard that executes the
    thing it is guarding is not a guard. Matches pytest's default
    `python_functions = test*` collection rule, including methods on Test*
    classes, so adding either kind trips this.
    """
    import ast

    tree = ast.parse((REPO_ROOT / path).read_text())
    found = [
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test")
    ]
    assert found == [], (
        f"{path} is excused from CI on the grounds that it defines no test "
        f"functions, but it now defines {sorted(found)}. Either add it to the "
        f"pytest invocation in {WORKFLOW.name} or rewrite the reason -- a file "
        f"with assertions nobody runs is the gap this list exists to prevent."
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
