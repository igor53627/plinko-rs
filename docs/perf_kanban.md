# Plinko Hint Gen Performance Kanban (0.01% Runs)

## Backlog
- [ ] Baseline: run `plinko_hints` on 0.01% sample without profiling; capture timings + throughput.
- [ ] Baseline: run `plinko_hints` on 0.01% sample with profiling; capture flamegraph + timings.
- [ ] Record HW/flags: CPU model, VAES/AVX-512, build flags, feature flags used.
- [ ] Capture dataset metadata: sample size, N, w, c, padding, seed.

## Planned Experiments
- [ ] Enable `btpe` feature; rerun 0.01% (no profiling); compare total + streaming time.
- [ ] Enable `btpe` + profiling; validate hotspot shifts (AES/PRP vs binomial).
- [ ] Add round-key precompute in `SwapOrNotSr`; rerun 0.01%.
- [ ] Add batched `SwapOrNotSr::inverse_batch` using `encrypt_blocks` (VAES); rerun 0.01%.
- [ ] Replace `block_in_subset` in fast path with bitset lookup; rerun 0.01%.
- [ ] Avoid `trace_ball_inverse` Vec allocation (return range); rerun 0.01%.
- [ ] Build with `-C target-cpu=native -C lto -C codegen-units=1`; rerun 0.01%.

## In Progress

## Done
- [x] Add SHA-256 key derivation to GPU CUDA kernel (match production code)
- [x] Re-run 50x H200 benchmark with SHA-256 key derivation → Run ID: 20260123_174356, 4.1 min max GPU time
- [x] Baseline 0.01% prod w/lambda (w=49177, lambda=127) → 4184.51s total, streaming 4180.31s (journald run)
- [x] BTPE 0.01% prod w/lambda → 4199.02s total, streaming 4194.80s (no improvement)
- [x] Native + LTO (CARGO_PROFILE_RELEASE_LTO=fat, target-cpu=native, codegen-units=1) 0.01% prod w/lambda → 2823.21s total, streaming 2819.79s
- [x] Native + LTO + SR round-key precompute 0.01% prod w/lambda → 2783.70s total, streaming 2780.35s (~1.4% faster vs native baseline)
- [x] Native + LTO + batch_iPRF (VAES batch) 0.01% prod w/lambda → 873.11s total, streaming 869.72s (~3.2x faster vs native baseline)
- [x] Native + LTO + batch_iPRF + profiling 0.01% prod w/lambda → 922.58s total, streaming 919.20s (profile overhead ~5.7%)

## Result Template (copy per run)
```
Run:
  Date:
  Host:
  Command:
  Features:
  Build flags:
  Dataset: sample_0p01pct.db (size: )
  N / w / c / padding:
Timings:
  Key derivation:
  Hint init:
  iPRF setup:
  Streaming:
  Total:
  Throughput:
Profile:
  Flamegraph:
  Top hotspots:
Notes:
  Correctness checks (parity non-zero counts):
```

```
Run:
  Date: 2026-01-18
  Host: aya (root@aya)
  Command: systemd-run --unit plinko_hints_0p01_prodW_btpe --working-directory=/mnt/mainnet/plinko bash -lc 'RUST_BACKTRACE=1 ./target/release/plinko_hints --db-path /mnt/mainnet/plinko/tmp/sample_0p01pct.db --entries-per-block 49177 --lambda 127 --seed 000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f'
  Features: btpe
  Build flags: default release
  Dataset: sample_0p01pct.db (size: 7.4M)
  N / w / c / padding: N=241,752; w=49,177; c=6 (bumped from 5); padding=4,133 + 53,310 (to even c)
Timings:
  Key derivation: 6.01µs
  Hint init: 4.22s
  iPRF setup: 21.96µs
  Streaming: 4194.80s
  Total: 4199.02s
  Throughput: 0.00 MB/s (tiny file; rounding)
Profile:
  Flamegraph: n/a
  Top hotspots: n/a
Notes:
  Correctness checks (parity non-zero counts): regular 0/6,245,479; backup_in 0/6,245,479; backup_out 0/6,245,479
  Journald: plinko_hints_0p01_prodW_btpe (finished at 18:02:52 UTC)
```

## Results

```

```
Run:
  Date: 2026-01-19
  Host: aya (root@aya)
  Command: systemd-run --unit plinko_hints_0p01_prodW_native_precompute --collect --property=RemainAfterExit=yes --property=StandardOutput=journal --property=StandardError=journal --property=SyslogIdentifier=plinko_hints_native_precompute --working-directory=/mnt/mainnet/plinko bash -lc 'RUST_BACKTRACE=1 ./target/release/plinko_hints --db-path /mnt/mainnet/plinko/tmp/sample_0p01pct.db --entries-per-block 49177 --lambda 127 --seed 000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f'
  Features: default (round-key precompute change)
  Build flags: CARGO_PROFILE_RELEASE_LTO=fat; RUSTFLAGS='-C target-cpu=native -C codegen-units=1'
  Dataset: sample_0p01pct.db (size: 7.4M)
  N / w / c / padding: N=241,752; w=49,177; c=6 (bumped from 5); padding=4,133 + 53,310 (to even c)
Timings:
  Key derivation: 900ns
  Hint init: 3.35s
  iPRF setup: 15.78µs
  Streaming: 2780.35s
  Total: 2783.70s
  Throughput: 0.00 MB/s (tiny file; rounding)
Profile:
  Flamegraph: n/a
  Top hotspots: n/a
Notes:
  Correctness checks (parity non-zero counts): regular 0/6,245,479; backup_in 0/6,245,479; backup_out 0/6,245,479
  Journald tag: plinko_hints_native_precompute (finished at 08:51:45 UTC)
  Speedup vs native baseline: ~1.4%
```

```
Run:
  Date: 2026-01-19
  Host: aya (root@aya)
  Command: systemd-run --unit plinko_hints_0p01_prodW_native_batch_profile --collect --property=RemainAfterExit=yes --property=StandardOutput=journal --property=StandardError=journal --property=SyslogIdentifier=plinko_hints_native_batch_profile --working-directory=/mnt/mainnet/plinko bash -lc 'RUST_BACKTRACE=1 ./target/release/plinko_hints --db-path /mnt/mainnet/plinko/tmp/sample_0p01pct.db --entries-per-block 49177 --lambda 127 --seed 000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f --profile --profile-out /mnt/mainnet/plinko/tmp/profile_0p01pct_native_batch.svg --profile-freq 100'
  Features: batch_iprf, profiling
  Build flags: CARGO_PROFILE_RELEASE_LTO=fat; RUSTFLAGS='-C target-cpu=native -C codegen-units=1'
  Dataset: sample_0p01pct.db (size: 7.4M)
  N / w / c / padding: N=241,752; w=49,177; c=6 (bumped from 5); padding=4,133 + 53,310 (to even c)
Timings:
  Key derivation: 710ns
  Hint init: 3.38s
  iPRF setup: 371.76µs
  Streaming: 919.20s
  Total: 922.58s
  Throughput: 0.01 MB/s (tiny file; rounding)
Profile:
  Flamegraph: /mnt/mainnet/plinko/tmp/profile_0p01pct_native_batch.svg
  Top hotspots: process_entries_fast; puruspe::beta::betai/betacf; exp/log in libc
Notes:
  Correctness checks (parity non-zero counts): regular 0/6,245,479; backup_in 0/6,245,479; backup_out 0/6,245,479
  Journald tag: plinko_hints_native_batch_profile (finished at 15:58:22 UTC)
  Profile overhead: ~5.7% vs non-profile batch run (873s → 923s)
```

```
Run:
  Date: 2026-01-19
  Host: aya (root@aya)
  Command: systemd-run --unit plinko_hints_0p01_prodW_native_batch --collect --property=RemainAfterExit=yes --property=StandardOutput=journal --property=StandardError=journal --property=SyslogIdentifier=plinko_hints_native_batch --working-directory=/mnt/mainnet/plinko bash -lc 'RUST_BACKTRACE=1 ./target/release/plinko_hints --db-path /mnt/mainnet/plinko/tmp/sample_0p01pct.db --entries-per-block 49177 --lambda 127 --seed 000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f'
  Features: batch_iprf
  Build flags: CARGO_PROFILE_RELEASE_LTO=fat; RUSTFLAGS='-C target-cpu=native -C codegen-units=1'
  Dataset: sample_0p01pct.db (size: 7.4M)
  N / w / c / padding: N=241,752; w=49,177; c=6 (bumped from 5); padding=4,133 + 53,310 (to even c)
Timings:
  Key derivation: 680ns
  Hint init: 3.39s
  iPRF setup: 405.34µs
  Streaming: 869.72s
  Total: 873.11s
  Throughput: 0.01 MB/s (tiny file; rounding)
Profile:
  Flamegraph: n/a
  Top hotspots: n/a
Notes:
  Correctness checks (parity non-zero counts): regular 0/6,245,479; backup_in 0/6,245,479; backup_out 0/6,245,479
  Journald tag: plinko_hints_native_batch (finished at 10:30:05 UTC)
  Speedup vs native baseline: ~3.2x
```
Run:
  Date: 2026-01-18
  Host: aya (root@aya)
  Command: systemd-run --unit plinko_hints_0p01_prodW --working-directory=/mnt/mainnet/plinko bash -lc 'RUST_BACKTRACE=1 ./target/release/plinko_hints --db-path /mnt/mainnet/plinko/tmp/sample_0p01pct.db --entries-per-block 49177 --lambda 127 --seed 000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f'
  Features: default
  Build flags: default release
  Dataset: sample_0p01pct.db (size: 7.4M)
  N / w / c / padding: N=241,752; w=49,177; c=6 (bumped from 5); padding=4,133 + 53,310 (to even c)
Timings:
  Key derivation: 1.38µs
  Hint init: 4.20s
  iPRF setup: 14.99µs
  Streaming: 4180.31s
  Total: 4184.51s
  Throughput: 0.00 MB/s (tiny file; rounding)
Profile:
  Flamegraph: n/a
  Top hotspots: n/a
Notes:
  Correctness checks (parity non-zero counts): regular 0/6,245,479; backup_in 0/6,245,479; backup_out 0/6,245,479
  Journald: plinko_hints_0p01_prodW (finished at 15:28:48 UTC)
```

```
Run:
  Date: 2026-01-19
  Host: aya (root@aya)
  Command: systemd-run --unit plinko_hints_0p01_prodW_native2 --collect --property=RemainAfterExit=yes --property=StandardOutput=journal --property=StandardError=journal --property=SyslogIdentifier=plinko_hints_native2 --working-directory=/mnt/mainnet/plinko bash -lc 'RUST_BACKTRACE=1 ./target/release/plinko_hints --db-path /mnt/mainnet/plinko/tmp/sample_0p01pct.db --entries-per-block 49177 --lambda 127 --seed 000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f'
  Features: default
  Build flags: CARGO_PROFILE_RELEASE_LTO=fat; RUSTFLAGS='-C target-cpu=native -C codegen-units=1'
  Dataset: sample_0p01pct.db (size: 7.4M)
  N / w / c / padding: N=241,752; w=49,177; c=6 (bumped from 5); padding=4,133 + 53,310 (to even c)
Timings:
  Key derivation: 980ns
  Hint init: 3.43s
  iPRF setup: 16.14µs
  Streaming: 2819.79s
  Total: 2823.21s
  Throughput: 0.00 MB/s (tiny file; rounding)
Profile:
  Flamegraph: n/a
  Top hotspots: n/a
Notes:
  Correctness checks (parity non-zero counts): regular 0/6,245,479; backup_in 0/6,245,479; backup_out 0/6,245,479
  Journald tag: plinko_hints_native2 (finished at 05:28:13 UTC)
  Speedup vs baseline: ~32.5%
```
