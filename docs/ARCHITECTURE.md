# Architecture

This repository contains the Plinko PIR stack: data extraction from Ethereum state, hint generation, and formal verification.

## Components

- **Extractor** (`src/main.rs`): Reads Reth on-disk state via `reth-db` (MDBX-backed) and produces flat files.
  - Outputs: `database.bin`, `account-mapping.bin`, `storage-mapping.bin`, `code_store.bin`, `metadata.json`.
- **Schema** (`plinko/src/schema40.rs`): Defines the 40-byte entry layout (v3).
- **Hint generation** (`plinko/src/bin/plinko_hints.rs`, `plinko/src/bin/hint_gen/`): Streaming HintInit over mmap'd `database.bin` (CPU fast/CT; optional GPU).
- **GPU acceleration** (`plinko/cuda/`, `plinko/src/gpu.rs`): Optional CUDA backend for high-throughput hint generation.
- **Formal verification** (`plinko/formal/`, `docs/Plinko.v`): Rocq specs and proofs aligned with the paper.

## Data Flow

```text
Reth node DB (MDBX via reth-db)
   -> Extractor
      -> database.bin + mappings + metadata
         -> Hint generator (CPU or GPU; mmap flat database.bin)
            -> hints.bin (parities)
```

## Invariants

- `database.bin` is a flat array of 40-byte entries (v3 schema).
- Account entries are 1:1 with accounts; storage entries are 1:1 with storage slots.
- Mappings index into `database.bin` by entry number.
