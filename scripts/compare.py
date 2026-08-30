"""Print every run in results/ as a compact table, best first (task compare)."""

from __future__ import annotations

import sys

from sconixlib import load_runs

df = load_runs("results/*")
if not len(df):
    print("no runs yet — try: task run")
    sys.exit(0)

try:
    import pandas as pd

    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 30)
except ImportError:
    pass

prefer = ["name", "status", "final_loss", "final_metric", "duration_s", "git_sha", "run_dir"]
cols = [c for c in prefer if c in df.columns]
sort_key = next((c for c in ("final_loss", "final_metric") if c in df.columns), None)
if sort_key:
    df = df.sort_values(sort_key, na_position="last")
print(df[cols].to_string(index=False))
