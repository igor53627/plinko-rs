# GPU Hint Generation Benchmark (Optimized)

**Date:** 2026-01-24
**Author:** Benchmarked on Modal Labs infrastructure

## Summary

Benchmark of the optimized GPU-accelerated hint generation for Plinko PIR. This run utilizes **PRP Pre-derivation**, **Fast PRF bit extraction**, and **Vectorized 128-bit loads** to improve throughput and reduce costs compared to the 2026-01-23 baseline.

**Key Results (Production Run 2026-01-24):**
- **50× H200:** 33.5M hints in **4.2 minutes** (compute time) / **8.2 minutes** (wall clock)
- **Total Throughput:** **132,482 hints/sec**
- **Cost:** **$17.20**
- **Efficiency:** ~20% throughput increase over previous baseline

## Production Run Details

**Run ID:** `20260124_182042`

```
============================================================
RESULTS - 50× H200 Production Hint Generation (Optimized)
============================================================
Run ID:              20260124_182042

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
  Optimization:      PRP Pre-derivation + ulong2 loads + Fast PRF

Hint Parameters:
  Total hints:       33,554,432 (= 2 × λ × w)
  Blocks per hint:   ~8,202 (= c / 2)
  Hint size:         32 bytes
  Output size:       1.07 GB

Timing:
  Wall clock time:   8.2 min (492.0s)
  Max GPU time:      4.2 min (253.3s)
  Avg GPU time:      3.7 min (221.6s)
  Min GPU time:      3.1 min (189.2s)

Per-Worker Stats:
  Workers:           50 × H200
  Hints per worker:  671,088
  Output per worker: ~21.5 MB

Throughput:
  Per-GPU (Avg):     2,650 hints/sec
  Per-GPU (Max):     3,547 hints/sec
  Cluster Total:     132,482 hints/sec

Cost Breakdown:
  TOTAL COST:        ~$17.20
============================================================
```

## Comparison with Baseline

| Metric | Baseline (2026-01-23) | Optimized (2026-01-24) | Improvement |
| :--- | :--- | :--- | :--- |
| **Run ID** | `20260123_174356` | `20260124_182042` | - |
| **Architecture** | In-Kernel SHA-256 | **CPU Pre-derived Keys** | **Arch Change** |
| **Optimizations** | Basic Batched ChaCha8 | **Fast PRF Path + Vectorized `ulong2`** | **Code Change** |
| **Total Hints** | 33,554,432 | 33,554,432 | - |
| **Total Throughput** | ~110,000 hints/sec | **132,482 hints/sec** | **+20.4%** |
| **Avg GPU Throughput** | 2,200 hints/sec | **2,650 hints/sec** | **+20.5%** |
| **Max GPU Throughput** | ~2,700 hints/sec | **3,547 hints/sec** | **+31.4%** |
| **Total Cost** | ~$20.00 | **$17.20** | **-14.0%** |
| **Wall Clock Time** | 10.0 min | **8.2 min** | **-18.0%** |

## Implementation Notes

### 1. PRP Key Pre-derivation
The SHA-256 derivation `SHA256(block_key || "prp")` was moved from the GPU kernel to the CPU. Since this derivation only happens once per block per epoch, performing it on the CPU and uploading the results eliminates 16,404 hash operations from the GPU's inner loop per step.

### 2. Fast PRF Bit Extraction
Implemented `get_prf_bit_direct_fast` which computes the ChaCha rounds but skips the final addition and storage for the 15 unused words of the state. This reduces the instruction count in the critical path of the SwapOrNot inverse.

### 3. Vectorized Loads (`ulong2`)
Updated the parity XOR loop to use 128-bit `ulong2` loads. This reduces the number of memory transactions and improves throughput for the random-access database reads.

### 4. Register Pressure Management
Verified that `__forceinline__` remains superior to `__noinline__` for the fast path functions on H200, as the compiler is able to optimize the register usage effectively when inlined.
