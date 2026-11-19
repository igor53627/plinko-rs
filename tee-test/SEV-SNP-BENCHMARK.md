# Plinko PIR Hint Generation: SEV-SNP TEE Benchmark Results

> **Historical (December 2025). These numbers do not apply to the current code.**
>
> - Measured code: `state-syncer/src/bin/plinko_hints.rs` at revision `3e3fe92`
>   (2025-12-05) or earlier; the exact revision of the binary is not recorded.
> - That generator made regular hints only. Its file header says "Backup hints:
>   Not implemented". It had no iPRF and no inverse. A BLAKE3 hash per
>   (block, hint) pair, or one BLAKE3-XOF or AES-CTR stream per block, decided
>   which hints a record goes to.
> - The first table uses w = 245 and 31,360 hints. The current default for the
>   same database is w = 65,536 and 16,777,216 hints.
> - The database is an older dataset of 2,418,358,495 entries. The v3 snapshot
>   has 1,834,095,877 records of 40 bytes (`docs/data_format.md`).
> - The current `plinko_hints` needs about 300 core-hours for the v3 snapshot
>   (extrapolated, `docs/hint_generation.md`), not 20 minutes.
> - The conclusions "production-viable", "no code changes required" and
>   "attestation ready" are not claims of the repository today. A TEE is not
>   part of the trust model (`docs/trust_model.md`).
>
> What the document still shows: the relative overhead of SEV-SNP over KVM for
> a compute-bound workload with AES-NI on this host.

## Summary

We benchmarked Plinko PIR hint generation inside an AMD SEV-SNP Trusted Execution Environment to measure the overhead of confidential computing.

**Key Finding: SEV-SNP adds only ~1.8% overhead** compared to a regular KVM VM, making TEE deployment practical for privacy-preserving PIR hint generation.

## Test Environment

- **Host**: AMD EPYC 9375F (64 cores, 1.1TB RAM)
- **Guest VM**: Ubuntu 24.04, 32 vCPUs, 128GB RAM
- **Database**: Ethereum mainnet state (2.4B entries, 73GB)
- **Workload**: Plinko PIR HintInit with lambda=128

### Software Stack

| Component | Version | Source |
|-----------|---------|--------|
| Host Kernel | 6.14.0-36-generic | Ubuntu |
| QEMU | 10.0.0 | [AMDESE/qemu snp-latest](https://github.com/AMDESE/qemu) |
| OVMF | edk2-stable202502 | [AMDESE/ovmf snp-latest](https://github.com/AMDESE/ovmf) |
| Guest Kernel | 6.8.0-49-generic | Ubuntu 24.04 cloud image |

## Benchmark Results

| Environment | vCPUs | Wall Time | Throughput | PRF calls/s | XOR ops/s |
|-------------|-------|-----------|------------|-------------|-----------|
| Bare metal | 64 | ~19.5 min | ~62 MB/s | - | - |
| KVM (no SEV) | 32 | 19m 35s | 62.90 MB/s | 263.83M/s | 131.91M/s |
| **SEV-SNP** | 32 | **19m 57s** | **61.80 MB/s** | **259.21M/s** | **129.61M/s** |

### Overhead Analysis

| Metric | KVM (baseline) | SEV-SNP | Overhead |
|--------|----------------|---------|----------|
| Wall clock time | 1173.29s | 1194.19s | **+1.8%** |
| Throughput | 62.90 MB/s | 61.80 MB/s | -1.7% |
| User CPU time | 37505s | 37951s | +1.2% |
| System CPU time | 36.5s | 255.5s | +600% |
| Page faults (major) | 0 | 666 | N/A |

The significant increase in system time is expected due to encrypted memory operations, but the overall wall-clock overhead remains minimal because the workload is compute-bound.

## SEV-SNP Verification

Guest kernel confirms SEV-SNP is active:

```
$ sudo dmesg | grep -i sev
[    7.926133] Memory Encryption Features active: AMD SEV SEV-ES SEV-SNP
[    8.139008] SEV: APIC: wakeup_secondary_cpu() replaced with wakeup_cpu_via_vmgexit()
[    9.145313] SEV: Using SNP CPUID table, 28 entries present.
[    9.701889] SEV: SNP guest platform device initialized.
```

## QEMU Configuration

```bash
qemu-system-x86_64 \
    -enable-kvm \
    -cpu EPYC-v4 \
    -smp 32 \
    -m 131072 \
    -machine q35,confidential-guest-support=sev0,memory-backend=ram1 \
    -object memory-backend-memfd,id=ram1,size=131072M,share=true,prealloc=false \
    -object sev-snp-guest,id=sev0,policy=0x30000,cbitpos=51,reduced-phys-bits=1 \
    -bios OVMF.fd \
    ...
```

## Workload Details

Plinko PIR hint generation parameters:
- **N** = 2,418,358,495 entries (Ethereum mainnet accounts + storage)
- **w** = 245 entries per block
- **c** = 9,870,851 blocks
- **lambda** = 128 (security parameter)
- **hints** = 31,360 (lambda * w)
- **PRF calls** = 3.10e11 (c * hints)
- **XOR operations** = 1.55e11 (expected, Bernoulli 1/2)

## AES-CTR Benchmark with w=49177 (Dec 4, 2025)

We tested AES-NI accelerated PRF with optimal Plinko parameters (w=√N, square layout):

| Environment | vCPUs | Time | Throughput | XOR ops/s |
|-------------|-------|------|------------|-----------|
| Bare metal | 64 | 19.7 min | 62 MB/s | 131M/s |
| SEV-SNP | 32 | 57 min | 21.6 MB/s | 45M/s |

**SEV-SNP overhead: ~2.9x** (but half the vCPUs). Normalized for vCPUs, overhead is ~1.45x.

With w=√N parameters, PRF mode (BLAKE3 vs AES-CTR) doesn't affect performance - the bottleneck is memory bandwidth at ~130M XOR/s on bare metal.

## Conclusions

1. **SEV-SNP is production-viable** for Plinko PIR hint generation with minimal overhead
2. **Compute-bound workloads** benefit most from TEE - memory encryption overhead is amortized
3. **No code changes required** - the same binary runs in TEE with full memory encryption
4. **Attestation ready** - SEV-SNP provides `/dev/sev-guest` for remote attestation
5. **AES-NI works in SEV-SNP** - hardware acceleration is available inside the TEE

## Future Work

- Integrate [AMD SEV-SNP Attestation SDK](https://github.com/automata-network/amd-sev-snp-attestation-sdk) for remote attestation
- Test with larger lambda values
- Measure cold-start overhead (first hint generation after boot)

## References

- [Plinko Paper](https://eprint.iacr.org/2024/318.pdf)
- [AMD SEV-SNP Documentation](https://www.amd.com/en/developer/sev.html)
- [AMDESE/AMDSEV GitHub](https://github.com/AMDESE/AMDSEV)
