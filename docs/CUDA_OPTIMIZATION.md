# CUDA Optimization Strategy: On-the-fly Compaction

## Problem
The 40-byte database schema ("Schema V3") introduces a performance penalty on GPUs due to misalignment.
- **Alignment:** 40 bytes is not a multiple of 16 bytes (128 bits).
- **Access Pattern:** Reading 40-byte strided data prevents the use of vectorized 128-bit loads (`ulong2` or `uint4`), forcing the use of slower 64-bit loads or causing alignment faults (XID 13 on H100/H200).
- **Bandwidth:** 20% of the memory bandwidth is wasted loading padding/tag bytes that are not used for the hint generation parity check.

## Solution: "Split-32/8" via VRAM Compaction

We implemented a "Compact-then-Process" strategy that maintains the 40-bit database structure on disk and host, but optimizes the layout in VRAM.

### 1. Data Loading (Host -> Device)
The host uploads the raw 40-byte entry database to the GPU as usual. This preserves compatibility with the existing storage format.

### 2. Compaction (Device -> Device)
A new CUDA kernel, `compact_entries_kernel`, runs immediately after upload. It reads the raw 40-byte entries and writes the first 32 bytes of each entry into a new, densely packed buffer.
- **Input:** `N * 40` bytes (Strided)
- **Output:** `N * 32` bytes (Packed)
- **Overhead:** Negligible (memory-bound, runs once per upload).

### 3. Optimized Processing
The main hint generation kernel (`hint_gen_kernel_opt`) now operates on the **packed 32-byte buffer**.
- **Alignment:** Every entry is at offset `i * 32`, which is always 16-byte aligned.
- **Vectorization:** We use `ulong2` loads to read the entire 32-byte payload in just 2 instruction pairs per thread, maximizing memory throughput.
- **Efficiency:** Global memory traffic is reduced by 20% (reading 32B instead of 40B).

## Benefits
- **Stability:** Eliminates alignment-related crashes (XID 13) on H100/H200.
- **Performance:** expected 20-30% speedup due to bandwidth reduction and improved load efficiency.
- **Compatibility:** No changes required to the database file format or CPU-side logic.
