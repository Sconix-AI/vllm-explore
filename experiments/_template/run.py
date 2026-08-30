"""Experiment entrypoint. Copy of the pattern from exp001_smoke — edit freely.

Run it:  task run -- THIS_FOLDER_NAME
"""

from __future__ import annotations

from pathlib import Path

from sconixlib import Run, load_config, set_seed

HERE = Path(__file__).parent
ROOT = HERE.parents[1]

EXP = HERE.name


def main() -> None:
    cfg = load_config(ROOT / "configs/default.yaml", HERE / "config.yaml")
    set_seed(cfg["seed"])

    with Run(EXP, config=cfg) as run:
        # ---- your work here ------------------------------------------------
        for step in range(cfg["epochs"]):
            metric = 1.0 / (step + 1)
            run.log(step=step, metric=metric)
        run.summary(final_metric=metric)
        # -----------------------------------------------------------------


if __name__ == "__main__":
    main()
