# Test Flakiness Analysis

## 1. Compilation Blocker (Fixed)

**Root cause**: The function `chacha8_block` was renamed to `chacha_block` in `plinko/src/gpu.rs`, but two test references were not updated:
- `gpu.rs:573` — `chacha8_block(&key, 1, 0)` (now `chacha_block`)
- `gpu.rs:577` — `chacha8_block(&key, 0, 0)` (now `chacha_block`)

This prevented **all** tests from compiling, and was observed to break CI in recent runs.

**Fix**: Renamed both references to `chacha_block` to match the current function name.

## 2. Formatting Issue (Fixed)

**Root cause**: Three files had formatting drift from `rustfmt` standards:
- `plinko/src/schema40.rs` — Long `assert_eq!` macro call needed line wrapping
- `plinko/src/bin/bench_gpu_hints.rs` — Import ordering, indentation of `if let` block, extra blank line, long `println!` call
- `plinko/src/gpu.rs` — Line length in `slice_mut` call (also fixed by `cargo fmt`)

**Fix**: Ran `cargo fmt`.

## 3. Test Suite Results (3 Runs)

| Run | Lib Tests | Bin Tests | Integration Tests | Failures | Time |
|-----|-----------|-----------|-------------------|----------|------|
| 1   | 86 pass, 5 ignored | 16 pass | 7 pass, 1 ignored | 0 | ~656s |
| 2   | 86 pass, 5 ignored | 16 pass | 7 pass, 1 ignored | 0 | ~660s |
| 3   | 86 pass, 5 ignored | 16 pass | 7 pass, 1 ignored | 0 | ~660s |

**No flaky tests detected across 3 consecutive runs.**

## 4. Flakiness Risk Analysis

### 4.1 Proptest-Based Tests (Low Risk)

The following modules use proptest with random inputs:

| Module | Test Count | Risk | Rationale |
|--------|-----------|------|-----------|
| `binomial.rs` | ~5 proptest | Low | Uses `proptest!` macro; regressions tracked |
| `iprf.rs` | ~6 proptest | Low | Uses `proptest!` macro; regressions tracked |
| `constant_time.rs` | ~9 proptest | Low | Simple property checks (equality, ordering) |
| `kani_proofs.rs` | ~5 proptest | Low | Bounded domain checks; regressions tracked |

**Mitigation already in place**: Proptest regression files exist at:
- `plinko/proptest-regressions/iprf.txt` — 1 saved regression case
- `plinko/proptest-regressions/kani_proofs.txt` — 1 saved regression case

These files ensure previously-discovered failure cases are replayed on every run, preventing regressions from being silently re-introduced.

### 4.2 Statistical Tests (Very Low Risk)

| Test | Tolerance | Risk | Rationale |
|------|-----------|------|-----------|
| `test_distribution_mean` | \|mean - 50.0\| < 2.0 | Very Low | Uses deterministic PRF sequence (`i * golden_ratio`), not random; 10,000 samples |
| `test_full_support` | count_seen >= 15/21 | Very Low | Deterministic PRF sequence; 100,000 iterations over domain [0,20] |
| `test_tee_distribution_mean` | \|mean - 50.0\| < 2.0 | Very Low | Same deterministic sequence as above |
| `test_tee_full_support` | count_seen >= 15/21 | Very Low | Same deterministic sequence as above |

These tests **appear** statistical but use a deterministic pseudo-random sequence (`i.wrapping_mul(0x9E3779B97F4A7C15)` — the golden ratio constant). This makes them fully reproducible and non-flaky.

### 4.3 Slow Tests (Timeout Risk in CI)

| Test | Approx Duration | Notes |
|------|----------------|-------|
| `test_tee_full_support` | ~180s | Marked as running >60s |
| `test_ct_and_fast_produce_same_results` | ~260s | Marked as running >60s |
| `test_ct_hintinit_*` (integration) | ~220s total | Multiple tests >60s each |

These are not flaky but could cause CI timeout failures if the CI runner has a per-test or per-job timeout configured too aggressively.

## 5. Summary

- **0 flaky tests** detected across 3 consecutive full test runs
- **1 compilation blocker** fixed (`chacha8_block` → `chacha_block`)
- **1 formatting issue** fixed (`cargo fmt` across 3 files)
- **Proptest regressions** are already tracked in source control
- **Statistical tests** use deterministic sequences — no flakiness risk
- **Slow tests** (~3-4 min each) are the main CI risk factor (timeouts, not flakiness)
