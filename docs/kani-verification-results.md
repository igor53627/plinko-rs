# Kani Formal Verification Results

Last run: 2025-12-06

## Summary

| Harness | Result | Time |
|---------|--------|------|
| `proof_binomial_sample_bounded` | ✅ SUCCESSFUL | 37.4s |
| `proof_binomial_sample_zero_denom` | ✅ SUCCESSFUL | 0.2s |
| `proof_binomial_sample_matches_coq` | ⏳ Long running | >5min |

## Verified Properties

### 1. `proof_binomial_sample_bounded`

**Property:** When `num < denom` and `count >= denom`, the output of `binomial_sample` is always `<= count`.

```
VERIFICATION RESULT:
 ** 0 of 10 failed

VERIFICATION:- SUCCESSFUL
Verification Time: 37.43s
```

**Constraints:**
- `count <= 1000`
- `denom > 1`, `denom <= 1000`
- `num < denom`
- `count >= denom`

This matches the PMNS invariant where balls (count) >= bins (denom).

### 2. `proof_binomial_sample_zero_denom`

**Property:** When `denom == 0`, `binomial_sample` returns `0` (safe division-by-zero handling).

```
VERIFICATION RESULT:
 ** 0 of 10 failed (5 unreachable)

VERIFICATION:- SUCCESSFUL
Verification Time: 0.2s
```

### 3. `proof_binomial_sample_matches_coq`

**Property:** The Rust implementation exactly matches the Coq definition:
```
binomial_sample(count, num, denom, prf_output) = 
  (count * num + (prf_output mod (denom + 1))) / denom
```

**Status:** Verification takes >5 minutes due to large symbolic search space. Recommend running locally or on self-hosted runner.

## Running Locally

```bash
# Via Docker (works on Apple Silicon)
docker run --platform linux/amd64 -v $(pwd):/work -w /work/plinko \
  rust:latest bash -c "
    cargo install --locked kani-verifier
    cargo kani setup
    cargo kani --harness proof_binomial_sample_bounded --output-format terse
  "

# Native (Linux x86_64 only)
cd plinko
cargo kani --harness proof_binomial_sample_bounded --output-format terse
```

## GitHub Actions

Kani runs manually via workflow dispatch:
```bash
gh workflow run kani.yml
```

Or: Actions → "Kani Formal Verification" → "Run workflow"
