"""Experiment entrypoint. Copy of the pattern from exp001_smoke — edit freely.

Run it:  task run -- THIS_FOLDER_NAME
"""

from __future__ import annotations

from pathlib import Path

from sconixlib import Run, load_config, set_seed
from vllm import LLM, SamplingParams

HERE = Path(__file__).parent
ROOT = HERE.parents[1]

EXP = HERE.name


def main() -> None:
    cfg = load_config(ROOT / "configs/default.yaml", HERE / "config.yaml")
    set_seed(cfg["seed"])

    with Run(EXP, config=cfg) as run:
        # ---- your work here ------------------------------------------------
        llm = LLM(model=cfg["model"])
        params = SamplingParams(temperature=cfg["temperature"], max_tokens=cfg["max_tokens"])

        outputs = llm.generate([cfg["prompt"]], params)
        text = outputs[0].outputs[0].text

        print(f"prompt: {cfg['prompt']!r}")
        print(f"output: {text!r}")

        run.summary(model=cfg["model"], prompt=cfg["prompt"], output=text)
        # -----------------------------------------------------------------


if __name__ == "__main__":
    main()
