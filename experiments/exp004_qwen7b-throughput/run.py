"""exp004_qwen7b-throughput — batched-decode throughput of Qwen2.5-7B-Instruct on the 5090.

The real-model version of exp002. One generate() call with n_prompts concurrent
requests, fixed output length, measure aggregate tokens/sec, request/sec, TTFT
(if vLLM exposes per-request metrics) and peak VRAM.

Run it:  task run -- exp004_qwen7b-throughput
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
    """Poll device VRAM in the background; keep the peak."""

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

    def stop(self) -> float:
        self._stop_evt.set()
        self.join(timeout=2)
        return self.peak_mib


HERE = Path(__file__).parent
ROOT = HERE.parents[1]
EXP = HERE.name

# A small pool of chat prompts, cycled up to n_prompts. Varied lengths so the
# batch isn't perfectly uniform.
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


def main() -> None:
    cfg = load_config(ROOT / "configs/default.yaml", HERE / "config.yaml")
    set_seed(cfg["seed"])

    n_prompts = int(cfg["n_prompts"])
    prompts = [PROMPT_POOL[i % len(PROMPT_POOL)] for i in range(n_prompts)]

    mem_before_mib = _gpu_mem_used_mib()

    with Run(EXP, config=cfg) as run:
        t0 = time.perf_counter()
        llm = LLM(
            model=cfg["model"],
            max_model_len=int(cfg["max_model_len"]),
            gpu_memory_utilization=float(cfg["gpu_memory_utilization"]),
            seed=int(cfg["seed"]),
        )
        load_s = time.perf_counter() - t0
        print(f"[load] {cfg['model']} ready in {load_s:.1f}s")

        params = SamplingParams(
            temperature=float(cfg["temperature"]),
            max_tokens=int(cfg["max_tokens"]),
            ignore_eos=True,  # force exactly max_tokens so runs are comparable
            seed=int(cfg["seed"]),
        )

        # warmup — compiles kernels / fills caches, not measured
        llm.generate(prompts[: min(8, n_prompts)], params, use_tqdm=False)

        sampler = _VramSampler()
        sampler.start()
        g0 = time.perf_counter()
        outputs = llm.generate(prompts, params, use_tqdm=False)
        gen_s = time.perf_counter() - g0
        peak_vram_mib = sampler.stop()

        out_tok = sum(len(o.outputs[0].token_ids) for o in outputs)
        in_tok = sum(len(o.prompt_token_ids) for o in outputs)
        peak_vram_gb = peak_vram_mib / 1024
        # gpu_memory_utilization * 32 GiB is what vLLM reserves up front; the
        # sampled peak is the honest "how full did the card get" number.
        used_gb = (peak_vram_mib - mem_before_mib) / 1024

        out_tok_per_s = out_tok / gen_s
        req_per_s = n_prompts / gen_s

        # per-request TTFT, if this vLLM build populates it
        ttfts: list[float] = []
        for o in outputs:
            m = getattr(o, "metrics", None)
            ft = getattr(m, "first_token_time", None) if m else None
            at = getattr(m, "arrival_time", None) if m else None
            if ft and at:
                ttfts.append(ft - at)
        ttfts.sort()

        def pct(xs: list[float], p: float) -> float | None:
            return xs[min(len(xs) - 1, int(len(xs) * p))] if xs else None

        ttft_p50 = pct(ttfts, 0.50)
        ttft_p99 = pct(ttfts, 0.99)

        run.log(
            step=0,
            gen_s=gen_s,
            out_tok=out_tok,
            out_tok_per_s=out_tok_per_s,
            req_per_s=req_per_s,
            peak_vram_gb=peak_vram_gb,
        )

        print(f"\n[gen] {n_prompts} reqs x {cfg['max_tokens']} tok  ->  {out_tok} out tok in {gen_s:.2f}s")
        print(f"[gen] {out_tok_per_s:,.0f} output tok/s   |   {req_per_s:.2f} req/s   |   {in_tok} prompt tok")
        if ttfts:
            print(f"[gen] TTFT  p50 {ttft_p50 * 1000:.0f} ms   p99 {ttft_p99 * 1000:.0f} ms")
        else:
            print("[gen] TTFT  n/a (no per-request metrics in this vLLM build)")
        print(f"[mem] device VRAM peak {peak_vram_gb:.2f} GB   |   +{used_gb:.2f} GB over idle")
        print("\n--- sample output (req 0) ---")
        print(outputs[0].outputs[0].text.strip()[:600])

        run.summary(
            model=cfg["model"],
            n_prompts=n_prompts,
            max_tokens=int(cfg["max_tokens"]),
            load_s=round(load_s, 2),
            gen_s=round(gen_s, 2),
            out_tokens=out_tok,
            in_tokens=in_tok,
            out_tok_per_s=round(out_tok_per_s, 1),
            req_per_s=round(req_per_s, 2),
            peak_vram_gb=round(peak_vram_gb, 2),
            vram_over_idle_gb=round(used_gb, 2),
            ttft_p50_ms=round(ttft_p50 * 1000, 1) if ttft_p50 else None,
            ttft_p99_ms=round(ttft_p99 * 1000, 1) if ttft_p99 else None,
        )


if __name__ == "__main__":
    main()
