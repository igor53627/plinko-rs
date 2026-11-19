---
id: TASK-28
title: Close the deviations of the online protocol from the paper
status: Done
assignee: []
created_date: '2026-11-17T11:36:00Z'
updated_date: '2026-11-17T14:53:00Z'
labels:
  - protocol
dependencies: []
references:
  - plinko/src/protocol.rs
  - plinko/src/iprf.rs
  - https://dl.acm.org/doi/pdf/10.1145/3658644.3690266
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The client in protocol.rs selects the decoy index in sequence (next_decoy); the paper requires a random index. prepare_query consumes the hint before the answer arrives, so a lost answer loses the hint. There is no amortized hint refresh: after q queries the client must run HintInit again. The query carries a mask and c/2 shared offsets; this is the RMS24 compact form, and the repository has no argument for it. Iprf::forward returns 0 for an input outside the domain, which is a valid bin. The client is an in-memory model with 32-byte entries.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The decoy index is uniform among the indices that are not in the cache
- [x] #2 A lost answer does not lose a hint, or the document states the rule for a retry and why it is private
- [x] #3 The document states what the client does after q queries; an amortized refresh is implemented or stated as out of scope
- [x] #4 The query format is checked against the RMS24 text (section 3 and the privacy proof) and the result is in the document
- [x] #5 Iprf::forward does not return a valid bin for an input outside the domain
- [x] #6 A test runs more than q queries and checks the behaviour at the limit
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
protocol.rs: a query for a cached index goes to a uniform index that is not cached and not in flight (rejection sampling, Fig. 7). prepare_query reserves the hint and one backup hint; the hint leaves the client in reconstruct_and_replenish or abandon_query; a retry sends the same Query; updates change a reserved hint. NoBackupHintsLeft and QueryInFlight come before a message is built. Hint candidates are tried in a random order. Iprf::forward panics outside the domain. Server::answer returns an error for a malformed mask. docs/online_protocol.md holds the rules and the open points. Result of the RMS24 check: the query with c/2 shared offsets is not the RMS24 encoding (RMS24 section 3.2 and Plinko Fig. 7 send one offset per block); it comes from keewoolee/rms24-plinko-spec. The document gives an argument for it, not a proof. Amortized refresh is out of scope; after q queries the client runs HintInit with a new master seed. Not done: database version in an answer, bias analysis for abandon_query, client identity in PreparedQuery (CodeRabbit, minor, documented). Checks: tests without features and with tee,gpu, e2e script, differential transcript script against the Python reference, CodeRabbit (1 minor finding, documented).
<!-- SECTION:FINAL_SUMMARY:END -->
