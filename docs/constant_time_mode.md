# Constant-Time Mode for TEE Execution

## Status

Experimental. This mode is not part of the [trust model](trust_model.md#tee-mode): the client runs HintInit, and the paper has no TEE. The repository has no attestation, no seed channel into an enclave, and no hint export.

The default build does not contain this mode. Build with the `tee` feature: `cargo build --release -p plinko --bin plinko_hints --features tee`. The code is in `plinko/src/iprf/tee.rs`, `plinko/src/constant_time.rs`, `plinko/src/hint_init/tee.rs` (`hint_cutoffs_ct`, `fill_parities_ct`) and `plinko/src/subsets.rs` (`find_cutoff_ct`).

## Claim

The mode makes one claim and no other:

> On one thread, the sequence of operations of HintInit depends only on public values: the number of entries, the record size, `w`, `c` and the hint counts. The memory addresses that it reads and writes are not hidden, and neither is the time that follows from them.

The secret values are the master seed and all values that follow from it: the block keys, the iPRF mapping (which hints hold which entry) and the block subsets (the cutoffs).

## Attacker

**Partly covered: an attacker that measures time.** Another party runs HintInit with the seed of the client, and the attacker measures the duration of the run, of a phase or of an entry. It does not see which memory the process touches.

- Removed: the time differences from the number and the kind of the operations. Before TASK-20 the PRP ran a number of rounds that depends on its input, the round key was a division with a key-derived dividend, and the cutoff was a selection over secret values.
- Not removed: the time of a memory access depends on the caches, and the accesses of an entry go to the cutoffs and the parities at the secret hint indices. Two entries with the same number of operations can thus take different times. This channel is present on one thread too. Its size is not measured, and no attack through it was built.

**Not covered: an attacker that sees memory accesses.** The host of a confidential VM sees the accessed pages (4 KiB) through page faults. A process on the same core sees the accessed cache lines (64 bytes) through cache probing. This mode does not resist that attacker, and a TEE deployment has that attacker.

The paper does not cover either case: its client runs HintInit, and no other party sees the execution.

### What The Hint Index Access Reveals

For entry `i` in block `alpha` at offset `beta` (all public), the path reads the cutoff and updates the parity words of each hint `j` with `iPRF_alpha(j) = beta`. For each unused slot of the 512 it accesses hint 0.

- An observer of the addresses learns the set of these hints for each entry: the iPRF mapping of each block. It also learns the number of preimages of the entry, from the number of accesses to hint 0.
- The precision is the number of hints in one observed unit. A page holds 1,024 cutoffs, or 102 parities of 40 bytes. A cache line holds 16 cutoffs, or 1 to 2 parities.
- The observer does not learn the block subsets from the addresses: the parity update is a masked XOR that always writes.

Consequence for the queries (an analysis; no experiment was made): a query sends `c/2` offsets, and those are `iPRF_k(j)` for the blocks `k` of hint `j`. With the iPRF mapping, the server can find `j` and the half of the bit mask that belongs to the hint. The queried index is then one of `c/2` known candidates, not one of `n`, and the candidates of more queries for the same index intersect. The privacy of the client against a server that has this trace is lost.

## Implementation

Key techniques used:

- **Fixed-bound loops**: Always iterates `MAX_PREIMAGES` (512) times per entry, using masks to skip invalid indices. The number of iterations does not depend on the preimage count.

- **Branchless membership**: Computes the select value of the block with one AES-128 call (`SubsetPrf::select`) and compares it with the cutoff of the hint through `ct_lt_u64`. An invalid backup hint (cutoff 0) is masked with `ct_eq_u64`. See [hint_generation.md](hint_generation.md#block-subsets).

- **Masked XOR**: Each parity update XORs the entry under a mask (`SharedParities::xor_masked`), so an update that does not count writes the same words with zero.

- **Branchless index clamping**: Uses `ct_select_usize` for safe array indexing.

### Fixed PRP Work

The inverse PRP of the 512 slots of an entry is one call of `SwapOrNotSrTee::inverse_many_ct`. Every slot runs every round of every Sometimes-Recurse stage. A mask keeps a value that a stage does not hold. The number of AES blocks and the memory accesses of the call depend only on `total_hints`.

`SwapOrNotSrTee::inverse` is the same call with one value, and `SwapOrNotSrTee::forward` runs every round of every stage under a mask. Before TASK-20 both ran only the stages that hold their input: 15 us at z = 65535 and 213 us at z = 0 for a domain of 65536. The test `test_sr_tee_time_does_not_depend_on_the_input` (release build only) compares the median times of `inverse` and of `forward` on a domain of 4096 for four values, from one that leaves the recursion at the first stage to one that passes all 12 stages. The limit is a ratio of 1.25 between the slowest and the fastest median. The test fails with the old `inverse` (medians of 2.76 ms and 0.26 ms).

The round key of a round has no division. It is one AES block mapped to `0..n` with two widening multiplications ([hint_generation.md](hint_generation.md#round-keys)).

### Fixed Cutoff Work

The cutoff of a hint is the select value below which exactly `c/2 + 1` (regular) or `c/2` (backup) blocks lie. The standard path finds it with a selection (`select_nth_unstable`), whose comparisons and swaps depend on the secret select values.

This mode uses `hint_init::hint_cutoffs_ct`. `find_cutoff_ct` builds the cutoff bit by bit from the most significant bit: 32 steps, and each step counts the values below a candidate over all `c` values with a branchless compare (`ct_lt_u32`). One more pass checks for a tie. The work depends only on `c`. The result equals the standard cutoff (`find_cutoff_ct_equals_find_cutoff`, `constant_time_cutoffs_equal_the_standard_cutoffs`).

A worker of this phase writes only to its own part of the result, so more workers add no secret-dependent timing here.

## What Remains

Criterion 4 of TASK-20 asks for this list. A line is either a property that was not verified or an operation that stays.

| Item                                    | State                                                                                                               |
|-----------------------------------------|---------------------------------------------------------------------------------------------------------------------|
| Memory access at the hint index         | Stays. See [What The Hint Index Access Reveals](#what-the-hint-index-access-reveals).                               |
| Cache-dependent time of these accesses  | Stays, on one thread too. A timing attacker sees it. Not measured.                                                  |
| More than one worker in the entry phase | Stays, opt-in. See [Worker Threads](#worker-threads).                                                               |
| Machine code                            | Not verified. The source has no branch on a secret value; no tool checked the compiler output.                      |
| Timing evidence                         | One statistical test of `SwapOrNotSrTee::inverse`, on one machine. No test of a full run.                           |
| Multiplication in the round key         | Stays. The time of a multiplication depends on the operands on some CPUs. Not measured.                             |
| AES                                     | Hardware AES, or the software code of the `aes` crate, which that crate states is constant-time. Not verified here. |
| Divisions                               | Only by public values: stage sizes (`n / 2`), bin ranges, the block index.                                          |
| Stop on a PMNS overflow                 | Stays. See [PMNS Sampler In This Mode](#pmns-sampler-in-this-mode).                                                 |
| `SwapOrNotTee` (one-level PRP)          | HintInit does not use it. Its round key is `AES(...) mod domain`: one division whose dividend depends on the key.   |

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

- The addresses that the path reads and writes are the same for one thread and for many threads: the cutoffs and the accumulators at the hint index. This is the leak of [What The Hint Index Access Reveals](#what-the-hint-index-access-reveals); threads do not add an address.
- With more than one worker, two workers that update the same word at the same time delay each other. Which words those are depends on the hint indices of both workers, so the time of a run then depends on secret values. The size of this effect is not measured. It is the reason for the default of one thread.
- The dummy accesses of all workers go to hint 0 of each accumulator, as on one thread.

Measured on 24 blocks with w = 1024 (24,576 entries of 40 bytes, 262,144 hints; Apple M3 Ultra, hardware AES, 2026-11-18, with the constant-time cutoffs): 291.3 s on one thread (11.9 ms per entry) and 14.1 s on 24 threads (337 s of CPU time). Both hint files are byte-identical to the file of the standard path. Before TASK-20 the same runs took 294.6 s and 14.3 s.

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

Measured at w = 65536 and 16,777,216 hints on one thread (Apple M3 Ultra, hardware AES): 16.7 ms per database entry in this mode and 0.52 ms in the standard path, a factor of about 32. The cost is the inverse PRP: this mode runs 512 slots through all 23 stages (about 8.3 million AES blocks per entry); the standard path runs about 256 balls through the stages that hold them, about 2 stages on average. Before the batched PRP ([hint_generation.md](hint_generation.md#batched-inverse-prp)) the numbers were 68 ms and 7 ms. The changes of TASK-20 (round keys without a division, all stages in `SwapOrNotSrTee::inverse`) did not change these two numbers measurably: 16.5 ms and 0.50 ms in the same scratch benchmark on 2026-11-18.

The constant-time cutoffs cost 101.7 us for each hint at c = 27,988 on one thread, against 39.1 us for the standard selection (262,144 hints: 26.7 s against 10.2 s, and 1.26 s on 24 threads, with equal cutoffs). For the 16,777,216 hints of mainnet that is about 28 minutes on one thread (computed), which is small against the entry phase.

`SwapOrNotSrTee::forward` and the single-value `SwapOrNotSrTee::inverse` are slower than before, because they run every stage and not only the stages that hold the value. HintInit does not call them; their new cost was not measured.
