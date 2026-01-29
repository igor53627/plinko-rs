# Plinko Protocol Overview

Plinko is a single-server PIR scheme that enables private lookup with efficient updates and strong asymptotic guarantees.

## Top-5 Features

1) **O(1) hint search via iPRF**: The client locates the relevant hint in constant time.
2) **Optimal query time trade-off**: Achieves O~(n/r) query time for any client storage r.
3) **O(1) update time**: Server publishes XOR deltas; client updates hints in constant time per changed entry.
4) **Ethereum-scale practicality**: Designed to operate on state sizes in the billions of entries.
5) **GPU-accelerated hint generation**: High-throughput preprocessing when CUDA is available.

## Phases

```text
Extract (Reth MDBX) -> database.bin + mappings + metadata
             -> HintGen (CPU/GPU) -> hints.bin
             -> Query/Update (client uses hints + deltas)
```

## References

- Paper: `docs/2024-318.pdf`
- Formalization: `docs/Plinko.v`
