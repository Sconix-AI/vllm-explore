# exp004_qwen7b-throughput

**Hypothesis:** A 7B model on the 5090 is memory-bandwidth bound in decode. Batching
64 requests through vLLM's paged attention should push aggregate output throughput
well past the single-stream rate, since decode steps amortise weight reads across
the batch.

**Prediction:** ~1500-3000 output tok/s aggregate at batch 64 (bf16, no quant),
single-stream would be ~40-60 tok/s. Peak VRAM in the 16-20 GB range with
gpu_memory_utilization=0.90 and max_model_len=4096. Load ~15-30 s from local cache.

**Setup:** Qwen2.5-7B-Instruct (just pulled + sha256-verified into $HF_HOME),
vLLM 0.28.0, torch 2.13+cu130, RTX 5090. 64 chat prompts cycled from a pool of 8,
`ignore_eos=True`, exactly 128 output tokens each. One `generate()` call, timed
around a `torch.cuda.synchronize()`. Changed vs exp003: real batch + throughput
metrics instead of a single-prompt chat smoke test.

**Result:** Batch-64 decode hits **~5,560 output tok/s** (8192 tok in 1.47 s), 43 req/s,
reproducible across two runs (5,557 / 5,628). Well above the predicted 1.5-3k — the
5090's bandwidth eats a 7B bf16 decode batch easily. Cold start 111 s (torch.compile +
CUDA-graph capture + FlashInfer autotune); warm start 21 s once those caches exist.
VRAM: the process sits at ~29.5 GB because `gpu_memory_utilization=0.9` pre-grabs the
pool — vLLM's own accounting is 15.6 GB weights, 1.1 GB activation, 0.4 GB CUDA graphs,
12.0 GB KV-cache pool (≈224k tokens, 55x concurrency at 4096 ctx). True working set ≈ 17 GB.
TTFT not captured — `RequestOutput.metrics` is empty on vLLM 0.28 V1.

**Decision:** keep. Baseline established. Next: sweep batch size to find the tok/s knee,
and get TTFT via the async engine or Prometheus stats.
