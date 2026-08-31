"""exp005_batch-size-sweep — where does aggregate decode throughput stop scaling?

Load Qwen2.5-7B-Instruct once, then run one timed generate() per batch size in
cfg["batch_sizes"]. Fixed 128 output tokens (ignore_eos) so every point is
comparable. Emits a tok/s-vs-batch curve plus a knee estimate.

Run it:  task run -- exp005_batch-size-sweep
"""

from __future__ import annotations

import subprocess
import threading
import time
from pathlib import Path

from sconixlib import Run, load_config, set_seed
from vllm import LLM, SamplingParams


def _gpu_mem_used_mib() -> float:
    """Device-wide VRAM in use, via nvidia-smi. vLLM runs the model in a spawned
    subprocess so torch.cuda.* in this process sees nothing — poll the device."""
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        text=True,
    )
    return max(float(x) for x in out.split())


class _VramSampler(threading.Thread):
    """Poll device VRAM in the background; keep the peak since the last reset."""

    def __init__(self, period_s: float = 0.1):
        super().__init__(daemon=True)
        self.period_s = period_s
        self.peak_mib = 0.0
        self._stop_evt = threading.Event()

    def run(self) -> None:
        while not self._stop_evt.is_set():
            try:
                self.peak_mib = max(self.peak_mib, _gpu_mem_used_mib())
            except Exception:
                pass
            self._stop_evt.wait(self.period_s)

    def reset(self) -> None:
        self.peak_mib = 0.0

    def stop(self) -> float:
        self._stop_evt.set()
        self.join(timeout=2)
        return self.peak_mib


HERE = Path(__file__).parent
ROOT = HERE.parents[1]
EXP = HERE.name

PROMPT_POOL = [
    "Explain, in 3 sentences, how a KV cache speeds up LLM inference.",
    "Write a short haiku about a GPU running hot.",
    "What is the difference between latency and throughput? Keep it brief.",
    "Give me three practical tips for reducing cloud compute costs.",
    "Summarize the plot of Romeo and Juliet in four sentences.",
    "Describe how paged attention manages the KV cache in vLLM.",
    "List five uses for a large language model in a research lab.",
    "In one paragraph, explain why batch size matters for inference throughput.",
]


def _plot(rows: list[dict]):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bs = [r["batch"] for r in rows]
    agg = [r["out_tok_per_s"] for r in rows]
    per_req = [r["per_req_tok_per_s"] for r in rows]

    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.plot(bs, agg, "o-", color="tab:blue", label="aggregate tok/s")
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(bs)
    ax1.set_xticklabels([str(b) for b in bs])
    ax1.set_xlabel("batch size (concurrent requests)")
    ax1.set_ylabel("aggregate output tok/s", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(True, which="both", alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(bs, per_req, "s--", color="tab:red", label="per-request tok/s")
    ax2.set_ylabel("per-request output tok/s", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    fig.suptitle("Qwen2.5-7B-Instruct decode throughput vs batch size (RTX 5090)")
    fig.tight_layout()
    return fig


def main() -> None:
    cfg = load_config(ROOT / "configs/default.yaml", HERE / "config.yaml")
    set_seed(cfg["seed"])

    batch_sizes = [int(b) for b in cfg["batch_sizes"]]
    max_bs = max(batch_sizes)
    max_tokens = int(cfg["max_tokens"])

    idle_mib = _gpu_mem_used_mib()

    with Run(EXP, config=cfg) as run:
        t0 = time.perf_counter()
        llm = LLM(
            model=cfg["model"],
            max_model_len=int(cfg["max_model_len"]),
            gpu_memory_utilization=float(cfg["gpu_memory_utilization"]),
            max_num_seqs=max_bs,
            seed=int(cfg["seed"]),
        )
        load_s = time.perf_counter() - t0
        print(f"[load] {cfg['model']} ready in {load_s:.1f}s  (max_num_seqs={max_bs})")

        params = SamplingParams(
            temperature=float(cfg["temperature"]),
            max_tokens=max_tokens,
            ignore_eos=True,  # force exactly max_tokens so every point is comparable
            seed=int(cfg["seed"]),
        )

        # one unmeasured warmup at the largest shape — covers first-call JIT
        # (triton gumbel-sample kernel, flashinfer autotune) so it doesn't land
        # inside a timed point.
        warm = [PROMPT_POOL[i % len(PROMPT_POOL)] for i in range(max_bs)]
        llm.generate(warm, params, use_tqdm=False)

        sampler = _VramSampler()
        sampler.start()

        rows: list[dict] = []
        for i, bs in enumerate(batch_sizes):
            prompts = [PROMPT_POOL[j % len(PROMPT_POOL)] for j in range(bs)]

            sampler.reset()
            g0 = time.perf_counter()
            outputs = llm.generate(prompts, params, use_tqdm=False)
            gen_s = time.perf_counter() - g0
            peak_mib = sampler.peak_mib

            out_tok = sum(len(o.outputs[0].token_ids) for o in outputs)
            in_tok = sum(len(o.prompt_token_ids) for o in outputs)
            agg_tok_s = out_tok / gen_s
            per_req_tok_s = agg_tok_s / bs
            req_s = bs / gen_s

            row = {
                "batch": bs,
                "gen_s": round(gen_s, 3),
                "out_tok": out_tok,
                "in_tok": in_tok,
                "out_tok_per_s": round(agg_tok_s, 1),
                "per_req_tok_per_s": round(per_req_tok_s, 1),
                "req_per_s": round(req_s, 2),
                "peak_vram_gb": round(peak_mib / 1024, 2),
                "vram_over_idle_gb": round((peak_mib - idle_mib) / 1024, 2),
            }
            rows.append(row)
            run.log(step=i, **row)
            print(
                f"[bs {bs:>4}]  {agg_tok_s:>8,.0f} tok/s agg  |  "
                f"{per_req_tok_s:>6.1f} tok/s/req  |  {req_s:>6.2f} req/s  |  "
                f"{gen_s:>6.3f}s  |  peak {peak_mib / 1024:.1f} GB"
            )

        sampler.stop()

        # knee: first batch size whose next doubling buys < 20% more aggregate tok/s
        knee = None
        for a, b in zip(rows, rows[1:], strict=False):
            gain = b["out_tok_per_s"] / a["out_tok_per_s"] - 1.0
            if gain < 0.20:
                knee = a["batch"]
                break
        peak_row = max(rows, key=lambda r: r["out_tok_per_s"])

        print()
        print(f"[knee]  aggregate tok/s stops scaling (<20% per doubling) at batch {knee}")
        print(
            f"[peak]  {peak_row['out_tok_per_s']:,.0f} tok/s at batch "
            f"{peak_row['batch']}  ({peak_row['per_req_tok_per_s']:.1f} tok/s/req)"
        )

        run.save_fig(_plot(rows), "throughput_vs_batch")

        run.summary(
            model=cfg["model"],
            max_tokens=max_tokens,
            load_s=round(load_s, 2),
            batch_sizes=batch_sizes,
            sweep=rows,
            knee_batch=knee,
            peak_tok_per_s=peak_row["out_tok_per_s"],
            peak_tok_per_s_batch=peak_row["batch"],
            single_stream_tok_per_s=rows[0]["out_tok_per_s"],
        )


if __name__ == "__main__":
    main()
