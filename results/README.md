# Results — Apple M3 Ultra

Benchmarked on **Apple M3 Ultra** (192GB unified memory) using MLX backend.

## Qwen3.5-0.8B-MLX-8bit

| Metric | Sequential (20 runs) | 16x Concurrent (20 runs) |
|---|---|---|
| Agent Score | **0.727** | 0.456 |
| Action Score | 0.455 | 0.512 |
| Restraint Score | **1.000** | 0.400 |
| Hard Prompts | 0.8/3 | 0.6/3 |
| Avg Latency | 532ms | 2,936ms |
| P50 Latency | 401ms | 2,899ms |
| Wall Time | 2m 8s | **56s** |
| Total Requests | 240 | 240 |

### Key Findings

1. **Concurrency degrades judgment** — restraint drops from 1.000 to 0.400 under load. The model starts hallucinating tool calls when it shouldn't.
2. **Sequential is the fair accuracy comparison** — 0.727 agent score with perfect restraint.
3. **Latency**: 401ms P50 on Apple Silicon GPU vs 1,567-10,000ms on CPU (original benchmark).
4. **Throughput**: 240 requests in 56 seconds at 16x concurrent.

### Comparison with Original Benchmark (CPU, Framework Laptop)

| Model | Agent Score | Latency | Hardware |
|---|---|---|---|
| qwen3:1.7b | **0.960** | 10,637ms | CPU (Ryzen AI 7) |
| lfm2.5:1.2b | 0.920 | 1,567ms | CPU |
| qwen3:0.6b | 0.880 | 3,037ms | CPU |
| **Qwen3.5-0.8B** | **0.727** | **401ms** | **GPU (M3 Ultra)** |

### Hardware
- Apple M3 Ultra, 192GB unified memory
- Backend: mlx_lm.server (MLX 0.31.0)
- Model: mlx-community/Qwen3.5-0.8B-MLX-8bit (~1 GB)
- Server config: 16 concurrent slots
