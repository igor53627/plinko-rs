---
id: decision-2
title: The library gives no integrity of an answer or of a delta
date: '2026-11-19T00:00:00Z'
status: accepted
---
## Context

The client computes `record = server parity XOR hint parity`. A server that changes a parity gives the client a wrong record with no error, and the error spreads through the backup hint that the answer promotes. A delta holds `old value XOR new value`; a wrong change gives wrong parities with no error. The privacy games of the paper compute the answer with the honest `Answer` algorithm, so the paper gives no integrity. TASK-24 asked for a mechanism or for a statement that integrity is out of scope.

## Decision

The library gives no integrity of an answer or of a delta. The client must trust the server for each answer, and the server and the publisher for each delta. `docs/integrity.md` states what the client checks and what it must trust.

A hash of the database before and after the changes in the header of a delta is not implemented: it changes the delta format and the hint file, the publisher must hash the full database for each version, and it detects a delta of another database but not a wrong delta.

Proposed in TASK-24 on 2026-11-18. Confirmed by the owner (igor53627) on 2026-11-19.

## Consequences

- A record is opaque bytes for the library, and a parity covers all bytes of a record. An application that needs integrity puts an authenticator of the publisher in each record and checks it after `Recon`. It must not show the result of the check to the server.
- The hint file keeps the BLAKE3 hash of the database that HintInit read (`HintSet::init_database`). A caller can compare it with a value from the publisher. A delta does not update it.
- After a wrong record is found, the client must run HintInit again with a new master seed. It cannot repair its hints.
- `docs/integrity.md` lists three mechanisms that were not selected, with their cost. A later task can select one.
