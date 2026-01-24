# GPU Benchmark Commands & Reports

Quick reference for running GPU hint generation benchmarks on Modal Labs.

## Prerequisites

```bash
# Unset any stale tokens before running
unset MODAL_TOKEN_ID && unset MODAL_TOKEN_SECRET
```

## Commands

### 2x H200 Quick Test (1% hints, full data)

Fast benchmark to measure per-hint timing with production parameters.

```bash
unset MODAL_TOKEN_ID && unset MODAL_TOKEN_SECRET && \
modal run scripts/modal_run_bench.py \
  --gpu h200 \
  --multi-gpu 2 \
  --replicate \
  --data-pct 100 \
  --hint-pct 1
```

**Expected time:** ~75-80 seconds per worker (ChaCha8)
**Expected cost:** ~$0.20

### 50x H200 Production Run (all hints)

Full production hint generation with all 33.5M hints.

```bash
unset MODAL_TOKEN_ID && unset MODAL_TOKEN_SECRET && \
modal run scripts/modal_run_bench.py --generate --num-gpus 50
```

**Expected time:** ~4.1 min GPU time, ~10 min wall clock (ChaCha8)
**Expected cost:** ~$20

### Download Generated Hints

```bash
# List available runs
modal volume ls plinko-hints /hints/

# Download specific run
modal volume get plinko-hints /hints/<run_id>/hints_combined.bin ./hints.bin

# Example:
modal volume get plinko-hints /hints/20260123_174356/hints_combined.bin ./hints.bin
```

### Other Useful Commands

```bash
# Single GPU test
modal run scripts/modal_run_bench.py --gpu h200 --data-pct 100 --hint-pct 1

# 20x H200 scale test
modal run scripts/modal_run_bench.py \
  --gpu h200 \
  --multi-gpu 20 \
  --replicate \
  --data-pct 100 \
  --hint-pct 1

# Re-run combine step only (if generation succeeded but combine failed)
modal run scripts/modal_run_bench.py --combine-only --run-id <run_id> --num-gpus 50

# Custom run ID for production
modal run scripts/modal_run_bench.py --generate --num-gpus 50 --run-id my_custom_run
```

## Parameter Reference

| Flag | Default | Description |
|------|---------|-------------|
| `--gpu` | h100 | GPU type: h100, h200, b200 |
| `--multi-gpu N` | 0 | Number of GPUs for parallel benchmark |
| `--replicate` | false | Replicate data to each worker |
| `--data-pct` | 100.0 | Percentage of mainnet data |
| `--hint-pct` | 1.0 | Percentage of hints to generate |
| `--generate` | false | Production mode: save hints to volume |
| `--num-gpus` | 50 | Number of GPUs for production run |
| `--run-id` | auto | Custom run ID (auto-generated if empty) |
| `--combine-only` | false | Only run combine step |

---

## Benchmark Reports

### ChaCha8 + SHA-256 (Production Configuration)

#### 2x H200 (1% hints)

```text
============================================================
2× H200 Benchmark Results (with SHA-256 key derivation)
============================================================
Configuration:
  GPUs:            2× H200
  Data:            100% mainnet (103 GB) replicated
  Hints:           1% = 335,544 (167,772 per GPU)
  set_size:        16,404 (production)
  Key derivation:  SHA-256(block_key || "prp")
  Cipher:          ChaCha8

Results:
  Worker 0:        167,772 hints in 77.48s
  Worker 1:        167,772 hints in 75.49s
  Throughput:      4,331 hints/sec
============================================================
```

#### 50x H200 (100% hints) - Run ID: 20260123_174356

```text
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
  Container overhead: ~4,500 GPU-seconds
  Total billed:      ~16,000 GPU-seconds
  TOTAL COST:        ~$20
============================================================
```

---

### ChaCha20 + SHA-256 (Experiment)

#### 2x H200 (1% hints)

```text
============================================================
2× H200 Benchmark Results (ChaCha20)
============================================================
Configuration:
  GPUs:            2× H200
  Data:            100% mainnet (103 GB) replicated
  Hints:           1% = 335,544 (167,772 per GPU)
  set_size:        16,404 (production)
  Key derivation:  SHA-256(block_key || "prp")
  Cipher:          ChaCha20

Results:
  Worker 0:        167,772 hints in 154.09s
  Worker 1:        167,772 hints in 133.58s
  Throughput:      2,178 hints/sec
============================================================
```

#### 50x H200 (100% hints) - Run ID: 20260123_221733

```text
============================================================
RESULTS - 50× H200 Production Hint Generation (ChaCha20)
============================================================
Run ID:              20260123_221733

Database Parameters:
  n (entries):       2,150,000,000
  Entry size:        48 bytes
  Database size:     103 GB

Plinko Parameters:
  λ (lambda):        128
  w (chunk size):    131,072
  c (set size):      16,404
  t (SwapOrNot):     759 rounds
  Cipher:            ChaCha20
  Key derivation:    SHA-256

Hint Parameters:
  Total hints:       33,554,432
  Blocks per hint:   ~8,202
  Hint size:         32 bytes
  Output size:       1.07 GB

Timing:
  Wall clock time:   15.2 min (911.0s)
  Max GPU time:      7.6 min (457.9s)
  Avg GPU time:      ~7.4 min (~443s)
  Min GPU time:      ~7.2 min (~435s)

Per-Worker Stats:
  Workers:           50 × H200
  Hints per worker:  671,088
  Output per worker: ~21.5 MB

Cost Breakdown (H200 @ $0.001261/sec):
  TOTAL COST:        ~$40 (est. ~2× ChaCha8)
============================================================
```

---

## Comparison Summary

| Config | 2x H200 (1%) | 50x H200 (100%) | Throughput |
|--------|--------------|-----------------|------------|
| **ChaCha8** | ~77s | 4.1 min (max GPU) | 4,331 hints/sec |
| **ChaCha20** | ~154s | 7.6 min (max GPU) | 2,178 hints/sec |
| **Slowdown** | 2.0× | 1.9× | 0.50× |

**Recommendation:** Use ChaCha8 for production.
