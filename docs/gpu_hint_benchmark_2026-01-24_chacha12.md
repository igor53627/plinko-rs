# GPU Hint Generation Benchmark (ChaCha12 + Full Authentication)

**Date:** 2026-01-24
**Author:** Benchmarked on Modal Labs infrastructure

## Summary

Benchmark of the production-ready GPU-accelerated hint generation for Plinko PIR using **ChaCha12** and **40B -> 48B VRAM Expansion**. This version addresses concerns regarding Tag recovery by expanding 40-byte entries into aligned 48-byte vectors in VRAM, ensuring the Hint (Parity) covers the entire entry including the Tag.

**Key Results (Production Run 2026-01-24):**
- **50× H200:** 33.5M hints in **4.2 minutes** (compute time) / **13.4 minutes** (wall clock)
- **Total Throughput:** **133,000 hints/sec** (Aggregate)
- **Cost:** **~$45.00**
- **Correctness:** Full 40-byte authentication (Data + Tag) restored via alignment padding.

## Production Run Details

**Run ID:** `prod_chacha12_v4_40B`

```text
============================================================
RESULTS - 50× H200 Production Hint Generation (ChaCha12)
============================================================
Run ID:              prod_chacha12_v4_40B

Database Parameters:
  n (entries):       1,834,095,877 (Active entries, V3 Schema)
  Entry size:        40 bytes (V3 Schema)
  VRAM size:         48 bytes (Expansion for alignment/authenticity)
  Database size:     ~73 GB (Disk) / ~88 GB (VRAM)

Plinko Parameters:
  λ (lambda):        128
  w (chunk size):    131,072
  c (set size):      13,996 (Active only)
  t (SwapOrNot):     759 rounds
  Cipher:            ChaCha12 (Stronger Security)
  Optimization:      VRAM Expansion (40B->48B) + Aligned ulong2 loads

Hint Parameters:
  Total hints:       33,554,432 (= 2 × λ × w)
  Blocks per hint:   ~6,998 (= c / 2)
  Hint size:         40 bytes (Disk - Padding stripped) / 48 bytes (VRAM)
  Output size:       1.34 GB

Timing:
  Wall clock time:   12.7 min (759.1s)
  Max GPU time:      4.2 min (252.8s)
  Avg GPU time:      ~4.0 min (~240s)

Per-Worker Stats:
  Workers:           50 × H200
  Hints per worker:  671,088
  Output per worker: ~26.8 MB

Throughput:
  Per-GPU (Avg):     ~2,660 hints/sec
  Cluster Total:     ~133,000 hints/sec

Cost Breakdown:
  TOTAL COST:        ~$45.00
============================================================
```

## Comparison with Baseline

| Metric | Baseline (ChaCha8) | Optimized (ChaCha12) | Improvement |
| :--- | :--- | :--- | :--- |
| **Run ID** | `20260123_174356` | `prod_chacha12_v4_40B` | - |
| **Cipher** | ChaCha8 | **ChaCha12** | **Stronger** |
| **Authentication** | Full | **Full (Data + Tag)** | **Verified** |
| **Entry Alignment** | 48B (Unaligned loads?) | **48B (Perfect ulong2)** | **Stable** |
| **Total Hints** | 33,554,432 | 33,554,432 | - |
| **Total Throughput** | ~110,000 hints/sec | **~133,000 hints/sec** | **+21%** |

## Optimization Details

### 1. VRAM Expansion (40B -> 48B)
To address algorithmic correctness, we expanded the 40-byte entries into 48-byte vectors in GPU memory. This ensures:
- **Alignment:** 48 is a multiple of 16, enabling `ulong2` (128-bit) loads.
- **Coverage:** The parity calculation now spans all 40 bytes of the original entry, meaning the client can recover and verify the Tag component.

### 2. Output Truncation (48B -> 40B)
While we process 48-byte chunks in VRAM for performance, the final 8 bytes are just padding. We strip these before writing to disk, resulting in a **1.34 GB** hints file that matches the 40-byte schema size perfectly.

### 3. Chunked Compaction
To avoid VRAM overflow (which previously happened when holding both 73GB raw and 88GB expanded buffers), we implemented a chunked upload-and-expand strategy. This keeps peak VRAM usage well below the H200's 141GB limit.

## Artifacts

**Generated Hints File:**
`hints.bin` (1.34 GB)
*(Artifact was downloaded locally and is not tracked in this repository.)*
