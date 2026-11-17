---
id: TASK-27
title: Correct performance claims in the documents
status: To Do
assignee: []
created_date: '2026-11-17T11:36:00Z'
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
- [ ] #1 Each benchmark document is re-measured on the current code, or it is marked as historical with the code revision that it measured
- [ ] #2 tee-test/run-benchmark.sh runs with the current plinko_hints options
- [ ] #3 docs/protocol_overview.md gives the measured cost of a query and of an update in place of O(1)
- [ ] #4 README.md names the paper authors Hoover, Patel, Persiano and Yeo
- [ ] #5 CHANGELOG.md does not list documents or directories that do not exist
<!-- AC:END -->
