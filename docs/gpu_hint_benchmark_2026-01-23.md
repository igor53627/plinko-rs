# GPU Hint Generation Benchmark

**Date:** 2026-01-23
**Author:** Benchmarked on Modal Labs infrastructure

## Summary

Benchmark of GPU-accelerated hint generation for Plinko PIR using ChaCha8-based SwapOrNot PRP.

**Key Results (Production Run 2026-01-23 with SHA-256 key derivation):**
- **50× H200:** 33.5M hints in **4.1 minutes** (wall clock 10.0 min)
- **Cost:** ~$20 (nearly constant regardless of GPU count)
- **Output:** 1.07 GB hints file
- **Throughput:** ~2,700 hints/sec per GPU

**ChaCha8 vs ChaCha20:** ChaCha20 is 1.9× slower (7.6 min vs 4.1 min max GPU time). ChaCha8 recommended for production.

## Production Parameters

| Parameter | Symbol | Value | Description |
|-----------|--------|-------|-------------|
| Security parameter | λ | 128 | Cryptographic security level |
| Chunk size | w | 131,072 | Entries per chunk |
| Set size | c | 16,404 | Number of chunks (blocks) |
| SwapOrNot rounds | t | 759 | PRP rounds per iPRF inverse |
| Cipher | - | ChaCha8 | 8-round ChaCha (ARX-based) |
| Total hints | - | 33,554,432 | 2 × λ × w |
| Blocks per hint | - | ~8,202 | c / 2 |
| Database size | - | ~103 GB | 2.15B entries × 48 bytes |

### SwapOrNot Round Calculation

```text
t = 7.23 × log₂(n) + 4.82 × log₂(p) + 4.82 × λ
  = 7.23 × log₂(131072) + 4.82 × log₂(2^64) + 4.82 × 128
  = 7.23 × 17 + 4.82 × 64 + 617
  ≈ 759 rounds
```

## Benchmark Configuration

### Hardware
- GPU: NVIDIA H200 (141 GB HBM3)
- Platform: Modal Labs serverless GPU

### Software
- CUDA 12.4
- Rust (release build)
- Custom CUDA kernel with:
  - SHA-256 key derivation (matches production: `prp_key = SHA256(block_key || "prp")`)
  - ChaCha8 ARX cipher (GPU-friendly, no S-box lookups)
  - Warp-level parallelism (`__shfl_sync`, `__ballot_sync`)
  - Batched SwapOrNot (7 rounds per ChaCha block)

## How to Run

### Prerequisites
```bash
# Install Modal CLI
pip install modal
modal setup

# Ensure mainnet data is uploaded to Modal volume
modal volume ls morphogenesis-data
# Should show: mainnet_optimized48.bin (~103 GB)
```

### Run Benchmark

```bash
# Basic: 2 H200 GPUs, 100% data replicated, 1% of hints
modal run scripts/modal_run_bench.py \
  --gpu h200 \
  --multi-gpu 2 \
  --replicate \
  --data-pct 100 \
  --hint-pct 1

# Scale test: 20 H200 GPUs
modal run scripts/modal_run_bench.py \
  --gpu h200 \
  --multi-gpu 20 \
  --replicate \
  --data-pct 100 \
  --hint-pct 1
```

### Production Hint Generation

```bash
# Generate all 33.5M hints with 50 H200 GPUs
# Saves to Modal volume: plinko-hints
modal run scripts/modal_run_bench.py --generate --num-gpus 50

# With custom run ID
modal run scripts/modal_run_bench.py --generate --num-gpus 50 --run-id my_run_001

# Re-run combine step only (if generation succeeded but combine failed)
modal run scripts/modal_run_bench.py --combine-only --run-id 20260123_164240 --num-gpus 50

# Download generated hints
modal volume get plinko-hints /hints/<run_id>/hints_combined.bin ./hints.bin
```

### Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--gpu` | h100 | GPU type: h100, h200, b200 |
| `--multi-gpu N` | 0 | Number of GPUs for parallel benchmark |
| `--replicate` | false | Replicate data to each worker (required for accurate benchmark) |
| `--data-pct` | 100.0 | Percentage of mainnet data to use |
| `--hint-pct` | 1.0 | Percentage of hints to generate (for faster testing) |
| `--generate` | false | Production mode: generate and save all hints to volume |
| `--num-gpus` | 50 | Number of H200 GPUs for production hint generation |
| `--run-id` | auto | Unique ID for hint generation run (auto-generated if empty) |
| `--combine-only` | false | Only run the combine step (for re-running after failure) |

## Results

### 2× H200 Benchmark (1% Hints Test)

**Configuration:**
- Data: 100% mainnet (103 GB) replicated to each GPU
- Hints: 1% = 335,544 (167,772 per GPU)
- set_size: 16,404 (production)

**Raw Results:**
```
Worker 0: 167,772 hints in 65.54s
Worker 1: 167,772 hints in 75.77s
Throughput: 4,429 hints/sec
```

### 50× H200 Production Run with SHA-256 Key Derivation

**Run ID:** `20260123_174356`

**Configuration:**
- GPUs: 50× H200
- Data: 100% mainnet (103 GB) - full database
- Hints: 100% = 33,554,432 (671,088 per GPU)
- set_size: 16,404 (production)
- Key derivation: SHA-256(block_key || "prp") → prp_key
- Output: 1.07 GB combined hints file

**Results:**
```
============================================================
RESULTS - 50× H200 Production Hint Generation (with SHA-256)
============================================================
Run ID:              20260123_174356

Database Parameters:
  n (entries):       2,150,000,000
  Entry size:        48 bytes
  Database size:     103 GB

Plinko Parameters:
  λ (lambda):        128
  w (chunk size):    131,072
  c (set size):      16,404
  t (SwapOrNot):     759 rounds
  Cipher:            ChaCha8
  Key derivation:    SHA-256 (matches production)

Hint Parameters:
  Total hints:       33,554,432 (= 2 × λ × w)
  Blocks per hint:   ~8,202 (= c / 2)
  Hint size:         32 bytes
  Output size:       1.07 GB

Timing:
  Wall clock time:   10.0 min (597.3s)
  Max GPU time:      4.1 min (246.7s)
  Avg GPU time:      3.7 min (220.8s)
  Min GPU time:      3.5 min (210.9s)

Per-Worker Stats:
  Workers:           50 × H200
  Hints per worker:  671,088
  Output per worker: ~21.5 MB

Cost Breakdown (H200 @ $0.001261/sec):
  GPU compute time:  ~11,000 GPU-seconds
  Container overhead: ~4,500 GPU-seconds (build, startup, I/O)
  Total billed:      ~16,000 GPU-seconds
  TOTAL COST:        ~$20
============================================================
```

**Throughput:** ~2,700 hints/sec per GPU

**Download hints:**
```bash
modal volume get plinko-hints /hints/20260123_174356/hints_combined.bin
```

### Comparison: With vs Without SHA-256

| Metric | Without SHA-256 | With SHA-256 | Difference |
|--------|----------------|--------------|------------|
| Run ID | 20260123_164240 | 20260123_174356 | - |
| Max GPU time | 259.1s | 246.7s | -4.8% |
| Avg GPU time | 233.4s | 220.8s | -5.4% |
| Min GPU time | 213.2s | 210.9s | -1.1% |
| Wall clock | 556.4s | 597.3s | +7.4% |

**Key observation:** Individual GPU compute times are slightly *faster* with SHA-256, likely due to variance in container scheduling. The SHA-256 overhead is minimal because:
1. Key derivation happens once per block per warp (amortized over 32 threads)
2. SHA-256 uses ARX operations that map efficiently to GPU
3. The cost is dominated by SwapOrNot rounds (759 ChaCha8 evaluations per block)

### ChaCha8 vs ChaCha20 Comparison

An experiment was run on 2026-01-23 comparing ChaCha8 (8 rounds) vs ChaCha20 (20 rounds) to measure the performance impact of increased cipher rounds.

**Experiment bookmark:** `chacha20-experiment` (jj)

| Metric | ChaCha8 | ChaCha20 | Slowdown |
|--------|---------|----------|----------|
| **2× H200 (1% hints)** |
| Worker 0 time | 77.5s | 154.1s | 2.0× |
| Worker 1 time | 75.5s | 133.6s | 1.8× |
| Throughput | 4,331 hints/sec | 2,178 hints/sec | 0.50× |
| **50× H200 (100% hints)** |
| Run ID | 20260123_174356 | 20260123_221733 | - |
| Wall clock | 597.3s (10.0 min) | 911.0s (15.2 min) | 1.5× |
| Max GPU time | 246.7s (4.1 min) | 457.9s (7.6 min) | 1.9× |
| Avg GPU time | ~220s | ~443s | 2.0× |
| Total hints | 33,554,432 | 33,554,432 | same |
| Output size | 1.07 GB | 1.07 GB | same |

**Analysis:**
- ChaCha20 is roughly **1.9× slower** than ChaCha8 on GPU
- This is less than the theoretical 2.5× (20/8 rounds) because:
  1. SHA-256 key derivation is fixed cost (same for both)
  2. Memory I/O overhead is constant
  3. SwapOrNot logic outside ChaCha is unchanged
- **Recommendation:** Use ChaCha8 for production. It provides adequate security for PRP usage while being ~2× faster.

**ChaCha20 Run Details:**
```
Run ID:           20260123_221733
GPUs:             50× H200
Total hints:      33,554,432
Max GPU time:     457.9s (7.6 min)
Wall clock:       911.0s (15.2 min)
Output:           1.07 GB
```

### Performance Scaling

| GPUs | Parallel Time | Est. Cost |
|------|---------------|-----------|
| 1× H200 | ~4 hr | ~$18 |
| 10× H200 | ~24 min | ~$18 |
| 20× H200 | ~12 min | ~$19 |
| 50× H200 | **4.3 min** | **~$20** |

**Key Insight:** Cost is nearly constant (~$18-20) regardless of GPU count. Multi-GPU only reduces wall-clock time. Linear scaling achieved.

### Throughput Analysis

| Metric | Value |
|--------|-------|
| Hints per second (1 H200) | ~2,200 |
| Block operations per hint | ~8,200 |
| ChaCha8 blocks per hint | ~8,200 × 759 × 2 ≈ 12.4M |
| Effective ChaCha8 throughput | ~27B blocks/sec |

## Comparison with Previous Estimates

| Test Config | set_size | Time for 33.5M hints | Notes |
|-------------|----------|---------------------|-------|
| 1% data | 132 | 5.9s | Misleading - tiny set_size |
| 10% data (split) | 44-84 | <1s per GPU | Invalid - split data |
| **100% data** | **16,404** | **4.2 hr** | **Production accurate** |

The 124× difference between 1% and 100% data benchmarks comes from set_size scaling (16,404 / 132 ≈ 124).

## Architecture Notes

### Why ChaCha8 over AES?
- AES requires S-box lookups (memory-bound on GPU)
- ChaCha uses only ARX operations (Add, Rotate, XOR)
- ChaCha8 is ~2× faster than ChaCha20 with acceptable security for PRP

### Parallelization Strategy
- Hints are embarrassingly parallel (independent)
- Each hint requires access to full database
- Split hints across GPUs, replicate data
- Cannot split data (each hint XORs entries across all blocks)

### CUDA Kernel Optimizations
1. **Warp-level key broadcast:** Block keys loaded once per warp, shared via `__shfl_sync`
2. **Batched ChaCha:** One ChaCha8 block provides randomness for 7 SwapOrNot rounds
3. **Early exit:** `__ballot_sync` skips blocks where no thread needs data

## Files

- `plinko/cuda/hint_kernel.cu` - CUDA kernel implementation
- `plinko/src/gpu.rs` - Rust GPU interface + CPU baseline
- `plinko/src/bin/bench_gpu_hints.rs` - Benchmark binary
- `scripts/modal_run_bench.py` - Modal orchestration script

## Modal Dashboard

View run history: <https://modal.com/apps/igor53627/main>
