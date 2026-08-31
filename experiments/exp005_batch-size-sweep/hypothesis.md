# exp005_batch-size-sweep

**Hypothesis:** Decode on a 7B bf16 model is memory-bandwidth bound: each step re-reads
all ~15 GB of weights regardless of how many requests share it. So aggregate output
throughput should climb almost linearly with batch size while the batch is small
(weight reads amortised over more tokens), then flatten into a knee once either
(a) compute / KV-cache-bandwidth catches up with weight bandwidth, or (b) vLLM's
scheduler / KV pool can't keep the whole batch resident and starts chunking.

**Prediction:** near-linear tok/s scaling from batch 1 -> ~32, a knee somewhere around
batch 32-64, and <1.5x further gain from 64 -> 256. Per-request tok/s falls
monotonically the whole way. exp004 already clocked batch 64 at ~5,560 tok/s, so the
plateau should sit in the 6-8k tok/s range. Peak VRAM roughly flat across the sweep
(the KV pool is pre-reserved by gpu_memory_utilization=0.9, not grown per request).

**Setup:** Qwen2.5-7B-Instruct, vLLM 0.28.0, torch 2.13+cu130, RTX 5090. Model loaded
**once**; then one timed `generate()` per batch size in {1,4,8,16,32,64,128,256}, each
preceded by an unmeasured warmup call. `ignore_eos=True`, exactly 128 output tokens
per request, prompts cycled from a pool of 8. Changed vs exp004: loop over batch size
instead of a single point, load amortised across the sweep, emit a tok/s-vs-batch curve.

**Result:** Sweep {1..1024}, 128 out tok/req, model loaded once.

| batch |     1 |    4 |    8 |   16 |   32 |   64 |  128 |   256 |   512 |  1024 |
|-------|------:|-----:|-----:|-----:|-----:|-----:|-----:|------:|------:|------:|
| agg tok/s   |  101 |  381 |  776 | 1426 | 2866 | 5426 | 9347 | 11523 | 12393 | 12489 |
| tok/s / req  | 101  |  95  |  97  |  89  |  90  |  85  |  73  |   45  |   24  |   12  |
| gen wall (s) | 1.27 | 1.35 | 1.32 | 1.44 | 1.43 | 1.51 | 1.75 |  2.84 |  5.29 | 10.50 |

- **Near-linear** aggregate scaling up to batch 64 (~1.9x per doubling) — decode really
  is weight-bandwidth bound in that regime, exactly as predicted.
- **Knee at batch 128-256.** 64->128 still buys 1.7x; 128->256 only 1.23x; 256->512
  1.08x; 512->1024 1.008x. Aggregate throughput **saturates at ~12,500 out tok/s**.
- Per-request rate holds ~85-100 tok/s through batch 64, then collapses (73 -> 45 -> 24
  -> 12) as the batch stops fitting in one decode step's compute budget. Wall-clock for
  the same 128 tokens goes 1.5s -> 2.8s -> 5.3s -> 10.5s.
- Peak VRAM flat at 31.4 GB across the whole sweep (KV pool pre-reserved; higher than
  exp004's 29.7 only because max_num_seqs=1024 makes vLLM grab more up front).
- Prediction was ~6-8k plateau; actual plateau ~12.5k (2x exp004's batch-64 point).
  The prediction of "knee at 32-64" was too low — it's 128-256.

**Serving sweet spot:** batch 64-128 — 5.4-9.3k aggregate tok/s while keeping
per-request rate at a usable 73-85 tok/s. Past 256 you pay pure latency for ~1% more
aggregate.

**Decision:** keep. Knee pinned at batch 128-256, plateau ~12.5k tok/s. Next: repeat
with a realistic output-length distribution (not fixed 128) and get TTFT via the async
engine so the latency axis is first-token, not whole-request.
