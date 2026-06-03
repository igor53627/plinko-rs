# Plinko Performance Kanban

## Done

- [x] Implement 40-byte schema (v3) - 17% storage reduction
  - AccountEntry40: Balance(16) + Nonce(4) + CodeID(4) + TAG(8) + Padding(8) = 40B
  - StorageEntry40: Value(32) + TAG(8) = 40B
- [x] Extract mainnet v3 database on hsiao (69 GB, 1.83B entries)
- [x] Upload to Modal volume `plinko-data/mainnet-v3/`
- [x] Fix GPU alignment for 40-byte entries (use uint64_t instead of ulong2)
- [x] Update modal scripts for v3 schema

## In Progress

- [ ] Investigate GPU performance variance (Worker 0: 31s vs Worker 1: 64s)
- [ ] Benchmark comparison: 40B vs 48B schema on same hardware

## Backlog

- [ ] Multi-GPU scaling test (10-50 GPUs)
- [ ] Production hint generation with v3 schema
- [ ] Optimize kernel for 40-byte entries (explore alternatives to uint64_t)

## Blocked

- [ ] Full hint generation (~6 hours with 2 GPUs) - waiting for performance investigation

---

## Benchmark Results (2026-01-24)

### Schema v3 (40-byte entries)

| Config | Hints | Time | Throughput | Notes |
|--------|-------|------|------------|-------|
| 2x H200, uint64_t | 50K/GPU | 31-64s | 1,565/sec | High variance |
| 2x H200, CPU padding | 50K/GPU | 129-136s | 741/sec | CPU bottleneck |

### Extrapolation (33.5M hints)

| GPUs | Est. Time | Est. Cost |
|------|-----------|-----------|
| 2 | ~6 hours | $58 |
| 50 | ~15 min | ~$60 |

### Storage Comparison

| Schema | Entry Size | Mainnet DB | Savings |
|--------|------------|------------|---------|
| v2 | 48 bytes | 88 GB | baseline |
| v3 | 40 bytes | 69 GB | 22% |
