---
id: TASK-27
title: Correct performance claims in the documents
status: Done
assignee: []
created_date: '2026-11-17T11:36:00Z'
updated_date: '2026-11-17T17:37:00Z'
labels:
  - docs
dependencies: []
references:
  - tee-test/SEV-SNP-BENCHMARK.md
  - tee-test/run-benchmark.sh
  - docs/BENCHMARK_RESULTS.md
  - docs/gpu_benchmark_commands.md
  - docs/protocol_overview.md
  - README.md
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
tee-test/SEV-SNP-BENCHMARK.md (mainnet in 20 minutes on 32 vCPU) was measured on a scheme with w = 245, no iPRF and no backup hints; that code is not in the repository. tee-test/run-benchmark.sh passes --threads, which the binary rejects. docs/BENCHMARK_RESULTS.md and docs/gpu_benchmark_commands.md describe kernels that are not in hint_kernel.cu and say ChaCha8 where the code uses ChaCha12. docs/protocol_overview.md promises O(1) hint search and O(1) update time. README.md names the wrong paper authors.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Each benchmark document is re-measured on the current code, or it is marked as historical with the code revision that it measured
- [x] #2 tee-test/run-benchmark.sh runs with the current plinko_hints options
- [x] #3 docs/protocol_overview.md gives the measured cost of a query and of an update in place of O(1)
- [x] #4 README.md names the paper authors Hoover, Patel, Persiano and Yeo
- [x] #5 CHANGELOG.md does not list documents or directories that do not exist
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Documents: scope section (the full Ethereum state is not a target, decision-1) and measured costs in docs/protocol_overview.md and README.md in place of the O(1) and Ethereum-scale claims; docs/update_strategy.md states what is implemented; docs/benchmarks.md lists current numbers first. Historical notes at the top of tee-test/SEV-SNP-BENCHMARK.md, docs/BENCHMARK_RESULTS.md, docs/gpu_benchmark_commands.md and plinko/docs/delta-format.md. Measured on 2026-11-17 (Apple M3 Ultra, hardware AES, 16,777,216 records, w = 4096): HintInit 412 s on 24 threads and 9,682 s CPU; query 56.0 ms client CPU median; update 0.51 to 0.53 ms per record; 1,000 queries and 200 queries after 2,000 updates were correct. Limits: the exact revision of the SEV-SNP binary and of each GPU run is not recorded, so the notes give the revision of the document and of the last kernel change; no SEV-SNP, CUDA or mainnet run was done; tee-test/run-benchmark.sh was run on macOS with a 4,096-record database, not in a guest.
<!-- SECTION:FINAL_SUMMARY:END -->
