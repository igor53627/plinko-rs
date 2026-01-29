# Plinko PIR: GPU Optimization & Benchmark Report
**Date:** 2026-01-26  
**Hardware:** NVIDIA H200 (141GB VRAM, ~100+ INT32 TOPS)  
**Cluster:** 50x H200 (Modal)

## Executive Summary
Through a series of architectural experiments, we identified that Plinko Hint Generation is **strictly compute-bound** by the sequential rounds of the Swap-Or-Not protocol. We successfully optimized the kernel to achieve **~3,000 hints/sec per GPU** on production Mainnet parameters.

**Result:** The entire Ethereum Mainnet hint set (33.5M hints) can be generated in **3.7 minutes** using 50 H200 GPUs at a cost of **~$15.00**.

---

## 1. Performance Evolution

| Phase | Architecture | Optimization | Throughput (per H200) | Bottleneck |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline** | Fused | Standard Scalar | 277k hints/s* | Compute |
| **Track A** | Fused | Collaborative (Warp) | 266k hints/s | Coalescing |
| **Track B** | Split | Buffered Aggregation | 270k hints/s | Compute |
| **Track C** | Split | **Interleaved 4x** | **315k hints/s*** | **Instruction Latency** |
| **Production** | Split | **Interleaved + High Occupancy** | **3,012 hints/s** (at c=16404) | **Fixed Compute** |

*\* Baseline and Track C measurements performed at c=160 (Synthetic). Production measured at c=16404 (Mainnet).*

---

## 2. Key Optimizations Implemented

### A. Split-Phase Architecture
We separated the kernel into two phases:
1.  **Phase 1 (Index Gen):** Runs the heavy `sn_inverse` (ChaCha12) and writes indices to a large VRAM buffer (100GB).
2.  **Phase 2 (Aggregation):** Reads the pre-calculated indices and performs high-bandwidth database XORs using `uint64_t` vectorized reads.
*   **Benefit:** Allows the aggregation phase to run at proven **73M hints/s**, ensuring memory access is never the bottleneck.

### B. Instruction Level Parallelism (Interleaving 4x)
The Swap-Or-Not rounds are sequential and dependency-heavy. By processing **4 hints per thread simultaneously**, we allowed the GPU scheduler to hide instruction latency.
*   **Benefit:** ~16% throughput increase even when compute-bound.

### C. VRAM Occupancy Tuning
By increasing the batch size to utilize **100GB of VRAM** for intermediate indices, we ensured that even with production set sizes ($c=16,404$), the GPU remains fully occupied ($>270,000$ active threads).
*   **Benefit:** 2x throughput increase compared to low-occupancy batches.

---

## 3. Production Requirements (Ethereum Mainnet)

To regenerate hints for the full Ethereum state ($N \approx 1.8B$, $\lambda=128$, $w=131072$):

*   **Total Hints:** 33,554,432
*   **Workload per Hint:** ~8,200 `sn_inverse` calls (50% density).
*   **Total Time (1x H200):** ~3.1 Hours
*   **Total Time (50x H200):** **3.7 Minutes**
*   **Estimated Cost:** $0.30 per GPU per run = **$15.00 total**

## 4. Final Recommendation
The current architecture is optimal for the given protocol constraints. Since the number of rounds is a hard security requirement, scaling is the primary lever for performance. The cluster-based approach is cost-effective and meets the latency requirements for periodic state updates.
