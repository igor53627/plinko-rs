# Optimization Strategy: VRAM Expansion (40B -> 48B)

## Overview

The "VRAM Expansion" optimization resolves the alignment issues of the 40-byte schema on GPUs by expanding entries to 48 bytes *in VRAM only*. This allows us to use efficient vectorized loads while maintaining the storage savings of the 40-byte schema on disk.

## Mechanism

### 1. Disk Storage (40 Bytes)
Entries are stored densely on disk to save space (17% vs 48B).
*   **Format:** `[Balance(16) | Nonce(4) | CodeID(4) | Tag(8) | Padding(8)]`
*   *Note:* Total useful data is 32 bytes + Tag.

### 2. VRAM Layout (48 Bytes - Expanded)
Upon upload to the GPU, entries are expanded to 48 bytes to align with 16-byte boundaries (128-bit).
*   **Format:** `[Original 40 Bytes] + [8 Bytes Zero Padding]`
*   **Alignment:** 48 is divisible by 16. This allows the kernel to read each entry using **3x `ulong2`** loads (128-bit each), which is significantly faster than unaligned reads.

### 3. Hint Generation (48 Bytes)
The kernel computes the hint (parity) over the full **48 bytes**.
*   This covers the **Value** (32B), the **Tag** (8B), and the **Padding** (8B).
*   **Correctness:** Since the Tag is included in the parity XOR sum, the client can recover the Tag correctly from the hint, ensuring full authentication.

### 4. Output Truncation (40 Bytes)
The hint for the 48th byte (padding) is always 0 (XOR of zeros).
Before saving to disk, we strip the last 8 bytes of the hint.
*   **Disk Hint Size:** 40 Bytes.
*   **Result:** The hint file size matches the database entry size perfectly.

## Why not 32 Bytes?
We initially considered compacting to 32 bytes (dropping the Tag). However, for **Storage Entries**, the Tag is essential for resolving collisions locally. While the server cannot spoof the Tag (due to privacy), the client still needs to verify it matches the slot key. Therefore, the Hint *should* cover the Tag to prove the server used the correct Tag in the computation.

The **Expansion (48B)** strategy satisfies this correctness requirement while retaining the performance benefits of alignment.

## Performance Impact

*   **Throughput:** **~133,000 hints/sec** on 50x H200 (ChaCha12).
*   **Speedup:** ~21% faster than the baseline 48-byte schema (due to active set size reduction), and ~60% faster than unaligned 40-byte processing.
*   **Storage:** Keeps the **17% storage reduction** of the V3 schema.