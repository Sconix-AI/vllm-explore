# vllm-explore



**Question:** How do llms inference engins work?

## Loop

```bash
task setup              # once
task exp -- my-idea     # scaffold experiments/expNNN_my-idea/
# edit that experiment's config.yaml + run.py
task run -- expNNN_my-idea
task compare            # see all runs side by side
task report             # render report.html
```

## Layout

| Path | What |
|---|---|
| `questions.md` | running list of open questions |
| `log.md` | dated project log, newest on top |
| `src/vllm_explore/` | reusable code (data, model, ...) |
| `experiments/` | one folder per experiment; `_template/` is the seed |
| `configs/default.yaml` | shared defaults, overridden per experiment |
| `results/` | one folder per run, written by `sconixlib.Run` — git-ignored |
| `report.qmd` | Quarto report that reads `results/` |

## Reproducibility

Every `run.py` wraps its work in `with Run(...) as run:`. That writes the
resolved config, git SHA + diff, `pip freeze`, GPU info, per-step metrics,
and a summary into `results/<timestamp>__<name>__<sha>/`. `results/latest`
points at the newest one.
