# Architecture

This repository contains a Plinko PIR library for a database of fixed-size records: hint generation by the client, the online protocol, hint updates, and formal models. The extractor of Ethereum state is a source of large test data; the full Ethereum state is not a target ([`protocol_overview.md`](protocol_overview.md#scope)).

## Components

- **Extractor** (`src/main.rs`): Reads Reth on-disk state via `reth-db` (MDBX-backed) and produces flat files.
  - Outputs: `database.bin`, `account-mapping.bin`, `storage-mapping.bin`, `code_store.bin`, `metadata.json`.
- **Schema** (`plinko/src/schema40.rs`): Defines the 40-byte entry layout (v3).
- **HintInit** (`plinko/src/hint_init.rs`): Cutoffs and parities of a hint set on worker threads (standard streaming phase; a constant-time one with the opt-in `tee` feature). The one implementation for the binary and for the protocol client.
- **Hint generation binary** (`plinko/src/bin/plinko_hints.rs`, `plinko/src/bin/hint_gen/`): Runs HintInit over mmap'd `database.bin`. `--output` writes the hint file.
- **Hint file** (`plinko/src/hints.rs`): Hint set of one client and its on-disk format, shared by the generator and the client.
- **Delta** (`plinko/src/delta.rs`): Changed records between two database versions and their on-disk format; the client applies deltas in version order.
- **Protocol client and server model** (`plinko/src/protocol.rs`): Loads a hint file (`Client::from_hints`), builds queries, reconstructs answers, and applies updates. In-memory model; the record size is a parameter (`ProtocolParams::record_size`); see [`online_protocol.md`](online_protocol.md).
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
