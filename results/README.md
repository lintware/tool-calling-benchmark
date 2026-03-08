# Results

Benchmarked on **Apple M3 Ultra** (192GB unified memory) using MLX and llama.cpp backends.

## Qwen3.5-0.8B-MLX-8bit

| Metric | Sequential (20 runs) | 16x Concurrent (20 runs) |
|---|---|---|
| Agent Score | **0.700** | 0.476 |
| Action Score | 0.425 | 0.503 |
| Restraint Score | 0.975 | 0.450 |
| Hard Prompts | 0.7/3 | 0.6/3 |
| Avg Latency | 502ms | 2,896ms |
| P50 Latency | 399ms | 2,780ms |
| Wall Time | ~120s | **53s** |

### Key Finding
Concurrent batching degrades model judgment significantly — restraint drops from 0.975 to 0.450 under load. Sequential results are the fair accuracy comparison; concurrent shows throughput capacity.

### Hardware
- Apple M3 Ultra, 192GB unified memory
- Backend: mlx_lm.server (MLX 0.31.0)
- Model: mlx-community/Qwen3.5-0.8B-MLX-8bit (~1 GB)
