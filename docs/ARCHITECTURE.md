# Architecture

This repository contains the Plinko PIR stack: data extraction from Ethereum state, hint generation, and formal verification.

## Components

- **Extractor** (`src/main.rs`): Reads Reth on-disk state via `reth-db` (MDBX-backed) and produces flat files.
  - Outputs: `database.bin`, `account-mapping.bin`, `storage-mapping.bin`, `code_store.bin`, `metadata.json`.
- **Schema** (`plinko/src/schema40.rs`): Defines the 40-byte entry layout (v3).
- **Hint generation** (`plinko/src/bin/plinko_hints.rs`, `plinko/src/bin/hint_gen/`): Streaming HintInit over mmap'd `database.bin` (standard path with worker threads; a constant-time path with the opt-in `tee` feature). `--output` writes the hint file.
- **Hint file** (`plinko/src/hints.rs`): Hint set of one client and its on-disk format, shared by the generator and the client.
- **Protocol client and server model** (`plinko/src/protocol.rs`): Loads a hint file (`Client::from_hints`), builds queries, reconstructs answers, and applies updates. In-memory model with 32-byte entries; see [`online_protocol.md`](online_protocol.md).
- **GPU benchmark** (`plinko/cuda/`, `plinko/src/gpu.rs`; opt-in features `gpu` and `cuda`): throughput benchmark. It does not compute Plinko hints and is not part of the [trust model](trust_model.md#gpu-path).
- **Formal models** (`plinko/formal/`; sketch `docs/Plinko.v`): Rocq models of the algorithms and proofs about the models. They do not verify the Rust code; see [`verification.md`](verification.md).

## Data Flow

```text
Reth node DB (MDBX via reth-db)
   -> Extractor
      -> database.bin + mappings + metadata
         -> Hint generator (CPU; mmap flat database.bin; master seed of the client)
            -> hints.bin (cutoffs + parities)
               -> Protocol client (hints.bin + master seed)
```

The client runs the hint generator and holds the master seed; see [`trust_model.md`](trust_model.md).

## Invariants

- `database.bin` is a flat array of 40-byte entries (v3 schema).
- Account entries are 1:1 with accounts; storage entries are 1:1 with storage slots.
- Mappings index into `database.bin` by entry number.
