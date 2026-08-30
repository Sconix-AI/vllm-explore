- **2026-08-30** — exp004 qwen7b-throughput: Qwen2.5-7B-Instruct batch-64 decode = ~5,560 out tok/s (43 req/s) on 5090, reproducible. Cold load 111s / warm 21s (torch.compile + cudagraph + flashinfer autotune caches). VRAM working set ~17GB. Next: batch sweep + TTFT.
# Project log

Newest on top. One line per session: what you did, what you learned, next step.
