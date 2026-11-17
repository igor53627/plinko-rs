# Constant-Time Mode for TEE Execution

## Security Goal

Eliminate timing side-channels that could leak the iPRF mapping (i.e., which hints contain which database entries).

This is critical because in Plinko, the hint structure encodes which database entries a client can privately retrieve. Leaking this structure would compromise PIR privacy.

## Implementation

Key techniques used:

- **Fixed-bound loops**: Always iterates `MAX_PREIMAGES` (512) times per entry, using masks to skip invalid indices. This prevents leaking preimage counts.

- **Branchless membership**: Computes the select value of the block with one AES-128 call (`SubsetPrf::select`) and compares it with the cutoff of the hint through `ct_lt_u64`. An invalid backup hint (cutoff 0) is masked with `ct_eq_u64`. See [hint_generation.md](hint_generation.md#block-subsets).

- **Masked XOR**: Uses `ct_xor_32_masked()` to conditionally XOR without branches, ensuring constant-time parity updates.

- **Branchless index clamping**: Uses `ct_select_usize` for safe array indexing.

## Security Model and Limitations

The CT mode protects against timing side-channels but does NOT provide full memory-access obliviousness. Array indexing patterns (e.g., `regular_hints[j].cutoff` and `regular_hints[j].parity`) may leak information to cache side-channel attackers (Prime+Probe, etc.).

This is acceptable for the paper's security model, which reasons in an idealized RAM model without microarchitectural side-channels. For stronger protection, an ORAM-based approach would be needed (O(n) overhead).

## MAX_PREIMAGES Bound

With default parameters (λ=128, q=λw), expected preimages per offset is approximately 2λ = 256.

The bound MAX_PREIMAGES=512 (approximately 2μ) is exceeded with probability < 2^{-140} per offset via Chernoff bounds. If it is exceeded, the run stops (see [PMNS Sampler In This Mode](#pmns-sampler-in-this-mode)).

A parameter guard enforces this at runtime:

```text
expected_preimages * 2 <= MAX_PREIMAGES
```

If exceeded, the program exits with an error instructing the user to reduce `total_hints` or increase `w`.

## Runtime Guards

The constant-time path requires:

1. `num_regular > 0` (lambda >= 1)
2. `num_backup > 0` (use `--backup-hints` to set q > 0)
3. Expected preimages * 2 <= MAX_PREIMAGES
4. `w` is a power of two (required in both modes)

## PMNS Sampler In This Mode

Both modes use the same exact sampler (`plinko/src/binomial.rs`): a PMNS split is the
number of set bits among the first `count` bits of an AES-128 counter keystream of the node.
This mode calls `coin_binomial_ct`, which always processes as many keystream blocks as a
public bound on the ball count needs and masks the unused bits, so its work does not depend
on the secret ball counts. The result is equal to the standard path bit for bit
(`test_pmns_tee_matches_standard`, `test_ct_variant_matches_for_every_bound`).

HintInit computes the tree of a database block once (`IprfTee::table_ct`), not once per
entry. A node that covers `b` bins is sampled with the bound
`min(total_hints, MAX_PREIMAGES * b)`. The bounds depend only on `total_hints` and `w`.
The table is read at the offset of the entry, which is public.

If a ball count exceeds its bound, or a bin holds more than `MAX_PREIMAGES` balls, the
table reports an overflow and `plinko_hints` stops with a panic. Before this check, a bin
above `MAX_PREIMAGES` was truncated silently and gave a wrong parity. The stop reveals only
that an event with the probability of the `MAX_PREIMAGES` bound occurred.

There is no limit on `total_hints` from the sampler.

## Performance

Measured at w = 65536 and 16,777,216 hints on one thread (Apple M3 Ultra, hardware AES): 68 ms per database entry in this mode and 7 ms in the standard path, a factor of about 10. The cost is the 512 fixed inverse-PRP iterations per entry; the standard path does about 256 and its PRP skips the levels that an input does not reach.
