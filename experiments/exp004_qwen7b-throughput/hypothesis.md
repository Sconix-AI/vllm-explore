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

**Result:** _(after running)_

**Decision:** _(keep / change / drop — one line)_
