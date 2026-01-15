# Plinko RS: State Size & Performance Findings

This document provides exact data sizes for the full Ethereum state and estimates the client-side storage requirements for the Plinko PIR protocol.

## 1. Ethereum State Statistics (Exact)

**Source**: Reth Archive Node (`v1.8.2`) on `reth-onion-dev`
**Block Height**: **#23,237,684** (Mainnet)
**Date**: Nov 23, 2025

- **Total Unique Accounts ($N_{acc}$)**: **328,168,813**
- **Total Storage Slots ($N_{sto}$)**: **1,415,847,352**

### Processed Artifact Sizes

The extractor produces a flat, padded database to ensure O(1) access and alignment.

#### A. `database.bin` (The Raw Data)
*   **Accounts**: occupy 4 words (128 bytes) each.
    *   $328.1M \times 128 \text{ bytes} \approx 42.0 \text{ GB}$
*   **Storage Slots**: occupy 1 word (32 bytes) each.
    *   $1.415B \times 32 \text{ bytes} \approx 45.3 \text{ GB}$
*   **Total Database Size**: **~87.3 GB**

#### B. Mappings (The Index)
*   **Account Mapping**: $328.1M \times (20 \text{ addr} + 4 \text{ idx}) \approx 7.9 \text{ GB}$
*   **Storage Mapping**: $1.415B \times (20 \text{ addr} + 32 \text{ key} + 4 \text{ idx}) \approx 79.3 \text{ GB}$
*   **Total Mapping Size**: **~87.2 GB**

> **Note**: The mapping allows the server to locate data. The PIR client **does not** need the mappings; it only needs the `database.bin` stream to generate hints.

---

## 2. Plinko PIR Client Storage

The Plinko protocol allows a client to query the database privately by storing a small set of "hints" (parities) instead of the full data.

### Parameters
*   **Total DB Entries ($N$)**:
    *   Accounts contribute $328.1M \times 4 = 1.31B$ entries.
    *   Storage contributes $1.415B$ entries.
    *   $N \approx 2.73 \text{ Billion entries}$.
*   **Square Root ($\sqrt{N}$)**: $\approx 52,250$.

### Hint Size Estimation
According to the Plinko tutorial:
> "The client generates roughly $128 * \sqrt{N}$ **hints**... and store data equal to roughly $64 * \sqrt{N}$ cells"

Assuming we need to retrieve full 32-byte words (balances/storage), each "cell" in the hint is 32 bytes (256 bits).

*   **Primary Hint Count**: $128 \times 52,250 \approx 6.69 \text{ million hints}$.
*   **Storage Requirement**:
    *   $6.69M \text{ hints} \times 32 \text{ bytes} \approx 214 \text{ MB}$.

### Backup Hints
For each query we intend to make, we need a backup hint.
*   **Size per Backup Hint**: $\sqrt{N}$ cells.
*   $52,250 \times 32 \text{ bytes} \approx 1.67 \text{ MB}$ per query.

---

## 3. Optimizations & "Concrete Efficiency"

Vitalik's "Concrete Efficiency" (and the Plinko design) relies on treating the database as a square matrix to optimize bandwidth and storage.

### Optimization 1: Client Storage Reduction
*   **Naive Light Client**: Stores headers only, requests Merkle proofs. (High bandwidth per query, no privacy).
*   **Full Node**: Stores ~100GB+ state.
*   **Plinko Client**: Stores **~214 MB** of hints.
    *   **Reduction**: ~99.7% reduction compared to full state.
    *   **Privacy**: Information-theoretic (IT-PIR) for reads.

### Optimization 2: Update Efficiency
The tutorial mentions:
> "There is also an easy way for the client to update hints if the data set changes."

Since hints are linear combinations (XOR sums), updates are efficient. When a block changes $k$ accounts:
1.  Client receives the storage diff (e.g., 300 changes per block).
2.  Client updates relevant hints in $O(1)$ time per change.
3.  **Cost**: Negligible computation compared to re-downloading.

### Optimization 3: Hybrid / "Partially Stateless" (Vitalik's Vision)
If we apply the "Concrete Efficiency" idea of **partially stateless nodes**:
*   We can split the state (e.g., by address prefix).
*   A node/client only maintains hints for the "slice" of the state they care about (e.g., 1/16th of the state).
*   **Result**: Client storage drops from ~214 MB to **~13.4 MB**.

## 4. Update Strategy

The extractor produces a static snapshot at a specific block height. To handle new blocks efficiently without re-downloading the entire 175GB dataset:

1.  **State Diffs**: A separate service monitors the chain for state changes (balance updates, storage writes).
2.  **Incremental Updates**:
    *   The server publishes a compact list of `(DatabaseIndex, NewValue)` for each block.
    *   **Client Update**: The Plinko client updates its local hints using the XOR property:
        $$ H_{new} = H_{old} \oplus \text{Val}_{old} \oplus \text{Val}_{new} $$
    *   This operation is $O(1)$ per changed state entry and does not require access to the full database.

## Conclusion

| Metric | Raw Ethereum State | Plinko Optimized (Client) |
| :--- | :--- | :--- |
| **Data Volume** | ~175 GB (DB + Maps) | **~214 MB** (Hints) |
| **Privacy** | None (Public Read) | **Private** (PIR) |
| **Query Cost** | Local Read (Fast) | 1 Round Trip + XOR |
| **Setup Cost** | Download 175 GB | Stream 87 GB (Once) |

For a light client wallet (like Rabby), storing **~214 MB** of static hints to gain complete read privacy for the entire Ethereum state is a highly practical trade-off.
