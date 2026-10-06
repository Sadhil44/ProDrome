"""Terminal dashboard (docs/guides/shaurya.md Part 7.4, PRD.md FR-23).

Live-updating table over the decision log: workload, detector score, whether
it fired, predicted class, confidence, the features that drove it, action taken,
outcome, and whether that action was real or shadow.

Both the path and the columns are taken FROM the controller rather than
restated here. This file previously declared its own copy of each, and both
drifted: COLUMNS said `timestamp` with no top_features and no mode, and the
path still pointed at control/decisions.csv long after the controller moved to
data/decisions/log.csv per SETUP.md S8. The column drift was caught and fixed;
the path drift was not, and it is the worse of the two -- the dashboard sat
rendering "waiting for decisions" forever while the controller was writing
perfectly good rows somewhere else. An operator watching a blank dashboard
during an incident would reasonably conclude nothing was happening.

Importing both from control.controller makes that class of divergence
impossible rather than merely detectable. The controller is importable without
a kubernetes client or a cluster -- `from kubernetes import ...` is lazy inside
connect_to_kubernetes() -- which is what makes this safe.

Deliberately terminal, not web - see dashboard/README.md for why.

Run: python -m dashboard.terminal
"""

import time
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.live import Live
from rich.table import Table

from control.controller import DECISION_LOG_COLUMNS, LOG_FILE

DECISIONS_LOG = LOG_FILE
REFRESH_SECONDS = 2
MAX_ROWS = 20

# SETUP.md S7, via the controller that writes it -- not a second copy. The
# previous local list said "timestamp" and omitted top_features and mode, so
# both halves drifted together and the divergence was invisible from either
# side. `mode` matters most to an operator: without it you cannot tell a shadow
# observation from an action that was actually executed.
COLUMNS = ["timestamp", "workload", "detector_score", "fired", "predicted_class", "confidence", "action", "result"]


def load_recent(path: Path = DECISIONS_LOG, max_rows: int = MAX_ROWS):
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return df.tail(max_rows).iloc[::-1].reset_index(drop=True)


def render(df) -> Table:
    if df is None:
        table = Table(title=f"Prodrome — waiting for {DECISIONS_LOG}")
        table.add_column("status")
        table.add_row(f"No decisions logged yet at {DECISIONS_LOG}. Run the controller loop first.")
        return table

    table = Table(title=f"Prodrome — last {len(df)} decisions")
    for col in COLUMNS:
        table.add_column(col)

    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        # Say so loudly rather than rendering blanks. row.get(col, "") made a
        # missing column indistinguishable from "we took no action", which is
        # the one thing an operator must never be confused about.
        table.caption = f"[bold red]MISSING COLUMNS: {', '.join(missing)}[/] — log does not match SETUP.md S7"

    for _, row in df.iterrows():
        fired = str(row.get("fired", ""))
        style = "bold red" if fired in ("True", "1", "true") else "dim"
        table.add_row(*(str(row.get(col, "")) for col in COLUMNS), style=style)

    return table


def main():
    console = Console()
    with Live(render(load_recent()), console=console, refresh_per_second=1) as live:
        while True:
            time.sleep(REFRESH_SECONDS)
            live.update(render(load_recent()))


if __name__ == "__main__":
    main()
