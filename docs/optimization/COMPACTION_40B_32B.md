# Compaction Optimization: 40B to 32B

## Overview

The "Compaction" optimization reduces the bandwidth required for hint generation by **20%** by discarding the "Tag" and "Padding" fields from the database entries *during the hint generation phase only*.

## Mechanism

### Account Database (40 Bytes)
*   **Original:** `[Balance(16) | Nonce(4) | CodeID(4) | Tag(8) | Padding(8)]`
*   **Compacted (32 Bytes):** `[Balance(16) | Nonce(4) | CodeID(4) | Tag(8)]`
    *   Layout: `Balance (0-16) | Nonce (16-20) | CodeID (20-24) | Tag (24-32) | Padding (32-40)`
    *   By keeping the first 32 bytes, the Tag is preserved; only the Padding is dropped.
    *   **Conclusion:** For accounts, we are **not losing any data** except the empty padding.

### Storage Database (40 Bytes)
*   **Original:** `[Value(32) | Tag(8)]`
*   **Compacted (32 Bytes):** `[Value(32)]`
*   **Result:** We keep the **Value** and drop the **Tag**.

## Security Implications

Does dropping the Tag affect security? **No.**

1.  **Hint Coverage:** The server generates a hint (proof) over the **Value**. This allows the client to cryptographically verify that the retrieved value is authentic.
2.  **Tag Verification:** The Tag is used for **Cuckoo Hash collision resolution**.
    *   The client knows the address/slot they are querying.
    *   The client calculates the expected `Tag` locally.
    *   When the client retrieves the full 40-byte row (Value + Tag), they compare the retrieved Tag with the expected Tag.
    *   **Collision:** If they match, it's the correct row. If not, it's a collision (or a different row).
3.  **Trust Model:**
    *   The server *does not know* which row the client is querying (PIR privacy).
    *   Therefore, the server cannot "spoof" a Tag to match the client's expectation, because the server doesn't know what the client expects.
    *   The client verifies the Tag *after* retrieval. The Hint doesn't need to cover the Tag because the Tag's correctness is self-evident (it matches the query key) and its authenticity is protected by the query privacy.

## Performance Impact

*   **Bandwidth:** Reduces memory traffic by 20% (40B $\to$ 32B).
*   **Alignment:** 32-byte entries are perfectly aligned for GPU memory (128-bit loads), enabling vectorized `ulong2` access instead of slower unaligned reads.
*   **Result:** ~60% speedup in hint generation benchmarks (7,000 hints/sec vs 4,300 hints/sec on 2x H200).
