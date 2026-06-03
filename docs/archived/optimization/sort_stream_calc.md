# Plinko PIR: "Batch-Sort-and-Stream" Performance Calculation

This document estimates the potential performance of replacing the random-access hint generation with a "Sort-and-Stream" architecture.

## 1. Parameters (Mainnet Production)
*   **Total DB Entries ($N$):** $1.83 \times 10^9$
*   **DB Size:** $73 \text{ GB}$ ($1.83B \times 40 \text{ bytes}$)
*   **Total Hints ($H$):** $33,554,432$
*   **Set Size ($c$):** $16,404$
*   **Subset Density:** $50\%$ (Average)
*   **Active Blocks per Hint:** $16,404 \times 0.5 = 8,202$

## 2. Workload Analysis
Total indices to process:
$$ \text{Total Indices} = 33.5M \text{ hints} \times 8,202 \text{ blocks} \approx 275 \text{ Billion} $$

Data volume of indices:
$$ 275 \times 10^9 \times 8 \text{ bytes (uint64)} \approx 2.2 \text{ TB} $$

## 3. Hardware Constraints (NVIDIA B200)
*   **VRAM:** $180 \text{ GB}$ (Use $150 \text{ GB}$ for buffer)
*   **VRAM Bandwidth:** $8 \text{ TB/s}$ (Theoretical) $\rightarrow$ Use $4.8 \text{ TB/s}$ (Real world effective)
*   **Radix Sort Performance:** $\approx 3 \text{ Billion keys/sec}$ (Estimated from CUB benchmarks on H100)

## 4. Batched Strategy
We cannot hold 2.2 TB of indices in memory. We must batch.

**Batch Size Calculation:**
Max indices per batch = $150 \text{ GB} / 8 \text{ bytes} \approx 18.75 \text{ Billion indices}$.
Hints per batch = $18.75B / 8,202 \approx \mathbf{2.2 \text{ Million hints}}$.

**Number of Batches:**
$$ \text{Batches} = 33.5M / 2.2M \approx \mathbf{15.2 \text{ batches}} $$

## 5. Performance per Batch (2.2M hints)

### Phase 1: Index Generation (Compute Bound)
From benchmarks, `sn_inverse` takes $\approx 1 \text{ second}$ for 33M hints (pure compute, if unbottlenecked).
For 2.2M hints: $1 \text{s} / 15 \approx \mathbf{0.1 \text{ s}}$.
*(This assumes highly optimized ChaCha. Currently it's slower, but let's assume we optimize it)*.

### Phase 2: Sorting (Memory/Compute Bound)
Sort 18 Billion indices (64-bit keys, 32-bit values for HintID).
Throughput $\approx 3 \text{ G-keys/s}$.
$$ \text{Sort Time} = 18B / 3B \approx \mathbf{6.0 \text{ s}} $$

### Phase 3: Streaming Aggregation (Memory Bandwidth Bound)
We scan the sorted indices (150GB) and the DB (73GB).
Since indices are sorted, we read DB sequentially.
Total read volume = $150 \text{ GB indices} + 73 \text{ GB DB} = 223 \text{ GB}$.
Total write volume (Hints) = $2.2M \times 40 \text{ bytes} \approx 0.1 \text{ GB}$ (Negligible).

$$ \text{Stream Time} = 223 \text{ GB} / 4.8 \text{ TB/s} \approx \mathbf{0.05 \text{ s}} $$

Wait! The Stream logic is complex:
For each DB entry $E$, we check the sorted index list.
If index match, we XOR $E$ into accumulator.
Since we have 18B indices for 1.8B entries, each DB entry is used **10 times** on average.
So we read DB once, but read Indices sequentially.
This is heavily compute/bandwidth bound.

Let's assume Streaming is bounded by Index Read Bandwidth (150GB).
Time $\approx 150 / 4800 = \mathbf{0.03 \text{ s}}$.

### Total Time per Batch
$$ 0.1 \text{ (Gen)} + 6.0 \text{ (Sort)} + 0.05 \text{ (Stream)} \approx \mathbf{6.15 \text{ s}} $$

## 6. Total Time & Throughput
Total Time = $15.2 \text{ batches} \times 6.15 \text{ s} \approx \mathbf{93 \text{ seconds}}$.

**Projected Throughput:**
$$ 33.5M \text{ hints} / 93 \text{ s} \approx \mathbf{360,000 \text{ hints/sec}} $$

## 7. Comparison vs Current
*   **Current (H200):** 315,000 hints/sec.
*   **Sort-and-Stream (Projected):** 360,000 hints/sec.

**Verdict:**
The sorting overhead (6 seconds per batch) dominates the gain from sequential reading.
Unless sorting can be done MUCH faster (e.g. 20 G-keys/s), this approach **breaks even** with the current random-access method.

**Why?**
Because the "Random Access" penalty on GPU HBM (High Bandwidth Memory) is surprisingly low compared to the cost of sorting billions of items. H200 HBM3e handles random reads very well compared to DDR4.

## 8. Alternative: "Binning" (Approximate Sort)
Instead of full sort, we just bin indices into e.g. 1024 regions.
Then we process each region.
This is cheaper than sorting ($O(N)$ vs $O(N \log N)$).
But requires more complex kernel.

**Conclusion:**
The current `~3,000 hints/sec/GPU` (on full data/full set size) scaling seems to be the sweet spot. 
*Note: The 360k projection above is for a single GPU.*
Wait. The current 315k was for synthetic set size (c=160).
For production set size (c=16404), current H200 is **3,000 hints/sec**.

**Correction:**
My "Sort-and-Stream" projection assumed processing 2.2M hints with c=16404.
This workload is MASSIVE.
If Sort-and-Stream achieves 360,000 hints/sec (at c=16404), that would be **100x FASTER** than current (3,000).

**Recalculating Sort Cost:**
Sorting 18 Billion keys.
If it takes 6 seconds.
That is for 2.2M hints.
Throughput = 2.2M / 6s = 366k hints/s.

**Comparison:**
*   **Current Random Access:** 3,000 hints/sec.
*   **Sort-and-Stream:** 366,000 hints/sec.

**Speedup:** **~120x**.

**CRITICAL ERROR IN PREVIOUS LOGIC:**
I compared the Sort projection (366k) to the *Synthetic* baseline (315k).
I should have compared it to the *Production* baseline (3k).

**FINAL VERDICT:**
**Sort-and-Stream is potentially 100x faster.**
Because sorting 18B items is cheaper than doing 18B random DRAM seeks (which stall execution).
18B random seeks @ 100ns latency (optimistic) = 1800 seconds (if serial).
GPU hides latency, but not 1000x.

**Action Item:**
Implementing "Sort-and-Stream" is the correct path to reach the 22M hints/s target on a single node.
