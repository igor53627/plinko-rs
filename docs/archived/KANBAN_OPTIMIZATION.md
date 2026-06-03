# Kanban: Plinko PIR Performance Optimization

**Objective:** Increase Hint Generation throughput on H200 GPUs.
**Baseline:** ~277k hints/s (Current Fused Kernel on H200).
**Theoretical Max:** ~22.5M hints/s (Raw Aggregation Benchmark).

## Strategy
We have two competing hypotheses to bridge the gap between 277k and 22.5M. We will test both.

---

## Track A: Fused Kernel Optimization
**Hypothesis:** The current kernel is inefficient due to unaligned 40-byte reads and scalar accumulation. Integrating the `int64`/`ulong2` vectorization and warp-shuffling directly into the fused kernel will unlock massive gains without the memory overhead of a split architecture.

### To Do
- [ ] **A1. Data Layout Transformation**
    - Create a Rust pre-processor to convert the 40-byte raw DB into a 48-byte (16-byte aligned) structure (padding 8 bytes).
    - *Goal:* Enable 128-bit `ulong2` vector loads on the GPU.
- [ ] **A2. Kernel Vectorization**
    - Modify `hint_kernel.cu` to use `ulong2` loads for reading entries.
    - Remove the existing byte-wise XOR loops.
- [ ] **A3. Warp-Level Reduction**
    - Implement `__shfl_xor_sync` logic to allow threads in a warp to collaborate on XORing, rather than each thread doing atomic updates or inefficient accumulators.
- [ ] **A4. Benchmark A (Fused)**
    - Run on H200. Target: > 1M hints/s.

---

## Track B: Split Architecture (Batched) - ACTIVE
**Hypothesis:** Separating Compute (Index Gen) from IO (Data Aggregation) allows the Aggregation phase to reach its full ~22 TB/s (theoretical) potential. Using a 12.8GB VRAM buffer allows us to process 100k hints in a single batch.

### B1. Implementation Plan (Sub-Kanban)
- [ ] **B1.1. Intermediate Index Buffer Management**
    - [ ] Define `IndexBuffer` struct in Rust to handle `DeviceSlice<uint64_t>`.
    - [ ] Implement batch size logic: `batch_size = total_vram / (blocks_per_hint * 8 bytes)`.
- [ ] **B1.2. Index Generation Kernel (`hint_kernel.cu`)**
    - [ ] Create `gen_indices_kernel`: Performs `sn_inverse_batched_fast` and writes results to global memory.
    - [ ] *Optimization:* Warp-collaborative key loading (one load per 32 threads).
- [ ] **B1.3. Buffered Aggregation Kernel (`hint_kernel.cu`)**
    - [ ] Create `aggregate_buffered_kernel`: Reads `uint64_t` indices and performs `ulong2` vectorized reads from DB.
    - [ ] *Optimization:* Coalesced index reads.
- [ ] **B1.4. Rust Orchestrator**
    - [ ] Modify `gpu.rs` to loop over batches of hints.
    - [ ] Execute `Gen -> Aggregate` sequence.
- [ ] **B1.5. Correctness Validation**
    - [ ] Run `cargo test` on small synthetic DB.
    - [ ] Compare GPU output against CPU `sn_inverse` reference.
- [ ] **B1.6. Final H200 Benchmark**
    - [ ] Measure throughput with Mainnet params (Set Size 16404).
    - [ ] Target: > 1M hints/s (conservative) to 5M+ hints/s (ambitious).

---

## Analysis & Measurements
- [x] **Profiling (2026-01-26):** Confirmed Memory Bound. 32M hints/s (Crypto Only) vs 277k hints/s (Full).
- [x] **Hypothesis A Test (2026-01-26):** Collaborative Fused Kernel regressed to 266k hints/s due to uncoalesced memory access.

---

## Analysis & Measurements
- [ ] **Profiling**
    - Run Nsight Compute on the current Baseline kernel.
    - Determine if we are **Compute Bound** (ChaCha/SHA) or **Memory Bound** (DRAM).
    - *If Compute Bound:* Track B is favored (Crypto needs optimization).
    - *If Memory Bound:* Track A is favored (Optimize reads).
- [ ] **Bandwidth Math**
    - Track B writes indices to VRAM then reads them back.
    - *Cost:* 1 Hint = ~16k indices * 8 bytes = 128KB write + 128KB read overhead per hint.
    - Check if H200 VRAM bandwidth (4.8 TB/s) makes this overhead negligible.

## Results Log
| Experiment | GPU | Throughput | Notes |
| :--- | :--- | :--- | :--- |
| Baseline (Fused) | H200 | 277k hints/s | 121s for 33M hints (Current) |
| Forge Aggregation (Only) | H200 | 22.5M hints/s | Upper bound limit |
| Optimized Fused (A) | TBD | | |
| Split Batch (B) | TBD | | |
