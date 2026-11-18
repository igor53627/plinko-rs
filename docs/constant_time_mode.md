# Constant-Time Mode for TEE Execution

## Status

Experimental. This mode is not part of the [trust model](trust_model.md#tee-mode): the client runs HintInit, and the paper has no TEE. The repository has no attestation, no seed channel into an enclave, and no hint export. The attacker that this mode must resist is not defined yet (TASK-20).

The default build does not contain this mode. Build with the `tee` feature: `cargo build --release -p plinko --bin plinko_hints --features tee`. The code is in `plinko/src/iprf/tee.rs`, `plinko/src/constant_time.rs` and `plinko/src/hint_init/tee.rs` (`fill_parities_ct`).

## Security Goal

Eliminate timing side-channels that could leak the iPRF mapping (i.e., which hints contain which database entries).

This is critical because in Plinko, the hint structure encodes which database entries a client can privately retrieve. Leaking this structure would compromise PIR privacy.

## Implementation

Key techniques used:

- **Fixed-bound loops**: Always iterates `MAX_PREIMAGES` (512) times per entry, using masks to skip invalid indices. This prevents leaking preimage counts.

- **Branchless membership**: Computes the select value of the block with one AES-128 call (`SubsetPrf::select`) and compares it with the cutoff of the hint through `ct_lt_u64`. An invalid backup hint (cutoff 0) is masked with `ct_eq_u64`. See [hint_generation.md](hint_generation.md#block-subsets).

- **Masked XOR**: Uses `ct_xor_masked()` to conditionally XOR without branches, ensuring constant-time parity updates.

- **Branchless index clamping**: Uses `ct_select_usize` for safe array indexing.

### Fixed PRP Work

The inverse PRP of the 512 slots of an entry is one call of `SwapOrNotSrTee::inverse_many_ct`. Every slot runs every round of every Sometimes-Recurse stage. A mask keeps a value that a stage does not hold. The number of AES blocks and the memory accesses of the call depend only on `total_hints`.

Before this change, the path called `SwapOrNotSrTee::inverse` for each slot. That function runs only the stages that hold its input: 15 us at z = 65535 and 213 us at z = 0 for a domain of 65536. The time of an entry thus depended on the position of its balls, which the block key decides. `SwapOrNotSrTee::inverse` still has this property; the constant-time path does not call it.

Not covered: the round key of a round is `AES(...) mod n`, a division with a key-derived dividend, and the division time depends on the operand values on some CPUs.

## Security Model and Limitations

The CT mode protects against timing side-channels but does NOT provide full memory-access obliviousness. Array indexing patterns (the cutoff and the parity accumulators of hint `j`) may leak information to cache side-channel attackers (Prime+Probe, etc.).

The cutoffs are computed before the streaming phase with `hint_init::hint_cutoffs`, the same function as in the standard path. It is not constant-time: it runs a selection (`select_nth_unstable`) over the secret select values of a hint (TASK-20).

The paper does not cover this case: its client runs HintInit, and no other party sees the execution. Against an attacker that sees memory accesses, such as the host of a confidential VM, this mode is not sufficient.

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
5. One thread by default; `--threads` starts more (see [Worker Threads](#worker-threads)).

## Worker Threads

The path runs on one thread unless `--threads n` is given. With `n` workers it processes the database blocks as the standard path does ([hint_generation.md](hint_generation.md#worker-threads)), and the hint file does not change (`constant_time_path_gives_the_same_parities_for_every_thread_count` with 1, 2, 3 and 16 threads; `constant_time_path_writes_the_same_file_for_each_record_size` runs the binary with 3 threads for one record size).

Each worker does the fixed work of its entries. The workers share the parity accumulators: atomic 64-bit words that a worker updates with a masked XOR at the hint index. The XOR value is zero for a slot that does not count.

- The addresses that the path reads and writes are the same for one thread and for many threads: the cutoffs and the accumulators at the hint index. This is the leak of [Security Model and Limitations](#security-model-and-limitations); threads do not add an address.
- With more than one worker, two workers that update the same word at the same time delay each other. Which words those are depends on the hint indices of both workers, so the time of a run then depends on secret values. The size of this effect is not measured. It is the reason for the default of one thread.
- The dummy accesses of all workers go to hint 0 of each accumulator, as on one thread.

Measured on 24 blocks with w = 1024 (24,576 entries of 40 bytes, 262,144 hints; Apple M3 Ultra, hardware AES, 2026-11-18): 294.6 s on one thread (12.0 ms per entry) and 14.3 s on 24 threads (341 s of CPU time). Both hint files are byte-identical to the file of the standard path.

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

Measured at w = 65536 and 16,777,216 hints on one thread (Apple M3 Ultra, hardware AES): 16.7 ms per database entry in this mode and 0.52 ms in the standard path, a factor of about 32. The cost is the inverse PRP: this mode runs 512 slots through all 23 stages (about 8.3 million AES blocks per entry); the standard path runs about 256 balls through the stages that hold them, about 2 stages on average. Before the batched PRP ([hint_generation.md](hint_generation.md#batched-inverse-prp)) the numbers were 68 ms and 7 ms.
