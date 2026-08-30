"""Scaffold experiments/expNNN_<name>/ from experiments/_template/.

Usage:  python scripts/new_experiment.py "learning rate sweep"
        task exp -- learning-rate-sweep
"""

from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

EXPS = Path(__file__).resolve().parents[1] / "experiments"


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def next_number() -> int:
    nums = [
        int(m.group(1))
        for p in EXPS.iterdir()
        if p.is_dir() and (m := re.match(r"exp(\d+)", p.name))
    ]
    return max(nums, default=0) + 1


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit("give the experiment a short name")
    name = slug(" ".join(sys.argv[1:]))
    folder = EXPS / f"exp{next_number():03d}_{name}"
    if folder.exists():
        sys.exit(f"{folder} already exists")
    shutil.copytree(EXPS / "_template", folder)
    for f in folder.rglob("*"):
        if f.is_file():
            f.write_text(f.read_text().replace("EXPERIMENT", folder.name))
    print(f"created {folder}")
    print(f"  edit  {folder}/config.yaml  and  {folder}/run.py")
    print(f"  run   task run -- {folder.name}")


if __name__ == "__main__":
    main()
