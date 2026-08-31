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

**Result:** _(after running)_

**Decision:** _(keep / change / drop — one line)_
