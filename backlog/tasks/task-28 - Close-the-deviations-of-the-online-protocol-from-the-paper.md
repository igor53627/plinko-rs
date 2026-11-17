---
id: TASK-28
title: Close the deviations of the online protocol from the paper
status: To Do
assignee: []
created_date: '2026-11-17T11:36:00Z'
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
- [ ] #1 The decoy index is uniform among the indices that are not in the cache
- [ ] #2 A lost answer does not lose a hint, or the document states the rule for a retry and why it is private
- [ ] #3 The document states what the client does after q queries; an amortized refresh is implemented or stated as out of scope
- [ ] #4 The query format is checked against the RMS24 text (section 3 and the privacy proof) and the result is in the document
- [ ] #5 Iprf::forward does not return a valid bin for an input outside the domain
- [ ] #6 A test runs more than q queries and checks the behaviour at the limit
<!-- AC:END -->
