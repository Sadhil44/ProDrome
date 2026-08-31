"""Terminal dashboard (docs/guides/shaurya.md Part 7.4, PRD.md FR-23).

Live-updating table over control/decisions.csv: workload, detector
score, whether it fired, predicted class, confidence, action taken,
outcome. Matches the exact columns Shravan's controller.log_decision()
writes.

Deliberately terminal, not web - see dashboard/README.md for why.

Run: python -m dashboard.terminal
"""

import time
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.live import Live
from rich.table import Table

DECISIONS_LOG = Path("control/decisions.csv")
REFRESH_SECONDS = 2
MAX_ROWS = 20

COLUMNS = [
    "timestamp", "workload", "detector_score", "fired",
    "predicted_class", "confidence", "action", "result",
]


def load_recent(path: Path = DECISIONS_LOG, max_rows: int = MAX_ROWS):
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return df.tail(max_rows).iloc[::-1].reset_index(drop=True)


def render(df) -> Table:
    if df is None:
        table = Table(title="Prodrome — waiting for control/decisions.csv")
        table.add_column("status")
        table.add_row("No decisions logged yet. Run the controller loop first.")
        return table

    table = Table(title=f"Prodrome — last {len(df)} decisions")
    for col in COLUMNS:
        table.add_column(col)

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
