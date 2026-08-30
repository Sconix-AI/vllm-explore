# Notes — exp004_qwen7b-throughput

## Runs
- 150710 (cold): load 111s, gen 5,628 tok/s — VRAM instrumentation broken (torch.cuda
  in main proc sees nothing; vLLM model lives in a spawned EngineCore subprocess).
- 151043: crashed — named a `threading.Event` attr `self._stop`, which shadows
  `threading.Thread._stop` and breaks `.join()`. Renamed to `_stop_evt`.
- 151155 (warm, good): load 21.2s, gen 5,557 tok/s, VRAM peak 29.5 GB via nvidia-smi poller.

## What surprised me
- Throughput ~2x my upper estimate. 7B bf16 decode at batch 64 is trivial for the 5090.
- Cold vs warm load is 111s vs 21s — almost all of the cold cost is torch.compile +
  CUDA-graph capture (54s) + FlashInfer autotune (20s), NOT weight load (13s). Caches:
  ~/.cache/vllm/{torch_compile_cache,flashinfer_autotune_cache,torch_aot_compile}.

## Gotchas for next vLLM experiment
- torch.cuda.* is useless for VRAM here — poll nvidia-smi from a thread instead.
- `gpu_memory_utilization` pre-allocates the whole pool, so "VRAM used" ~= util * 32GB.
  To measure the model's real floor, drop util or read vLLM's startup accounting line.
- TTFT / per-request timing: `RequestOutput.metrics` is None on vLLM 0.28 V1.

## Next
- Batch sweep {1, 4, 16, 32, 64, 128, 256} -> tok/s and per-request latency curve.
- TTFT via AsyncLLM or the /metrics Prometheus endpoint.
