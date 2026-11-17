---
id: decision-1
title: Target a PIR library for fixed-size records, not the full Ethereum state
date: '2026-11-17T17:22:00Z'
status: accepted
---
## Context

The repository was built to serve the full Ethereum state through Plinko PIR. A review of November 2026 measured the cost that each client pays in the model of the paper, where the client runs HintInit and holds the master seed (`docs/trust_model.md`). For the mainnet v3 snapshot (1,834,095,877 records):

- Download of `database.bin`: 73.4 GB.
- HintInit: about 300 core-hours (extrapolated from runs at the mainnet hint count; no mainnet run was done).
- A new HintInit when an insert moves the record indices (TASK-23) and after `Q` queries.
- No private lookup by address or slot without mapping files of 91 GB (TASK-22), and no check of an answer (TASK-24).

HintInit by a server, a cloud or a TEE gives the master seed to another party and is outside the trust model.

## Decision

The full Ethereum state is not the target. The target is a PIR library for a database that is an array of fixed-size records: HintInit by the client, the online protocol, and hint updates.

Decided by the owner (igor53627) on 2026-11-17.

## Consequences

- Documents do not describe the repository as a solution for Ethereum-scale state. Mainnet numbers stay as a scale reference.
- The Reth extractor, the v3 record format and `gen_synthetic` stay as sources of test data.
- The Ethereum data-layer tasks changed on 2026-11-17, with the agreement of the owner:
  - TASK-25 (consistent extractor snapshot) is closed without implementation. `docs/data_format.md` lists the limits of the extractor.
  - TASK-22 is now lookup by key for a general key-value database.
  - TASK-23 is now a database version and a delta feed.
  - TASK-24 is now a check of an answer against a commitment of the database, with no state root.
- The record size is fixed today (40-byte records, hints over the first 32 bytes). A library for other databases needs the record size as a parameter (TASK-31).
