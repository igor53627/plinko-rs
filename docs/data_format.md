# Data Format

This document defines the Plinko v3 on-disk format and dataset layout.

## Files

- `database.bin`: flat array of 40-byte entries.
- `account-mapping.bin`: `Address(20) || Index(4)`.
- `storage-mapping.bin`: `Address(20) || SlotKey(32) || Index(4)`.
- `code_store.bin`: `[count: u32][hash0: 32B][hash1: 32B]...`.
- `metadata.json`: snapshot metadata and schema version.

## Entry Layout (v3, 40B)

### Account Entry (40B)
- Balance: 16B (lower 128 bits, LE)
- Nonce: 4B (u32, LE)
- CodeID: 4B (LE). Index into `code_store.bin` (0 = EOA)
- TAG: 8B (cuckoo fingerprint `keccak256(address)[0:8]`)
- Padding: 8B (zeroed)

### Storage Entry (40B)
- Value: 32B (LE)
- TAG: 8B (cuckoo fingerprint `keccak256(address || slot_key)[0:8]`)

## Mainnet v3 Snapshot

As of block 23,876,768 (2026-01-24):
- Accounts: 351,681,953
- Storage slots: 1,482,413,924
- Total entries: 1,834,095,877
- Database size: 73,363,835,080 bytes

## PIR Bucket Links

Base URL: `https://pir.53627.org/mainnet-pir-data-v3/`

| File | Size (bytes) | Link |
|------|--------------|------|
| `database.bin` | 73,363,835,080 | https://pir.53627.org/mainnet-pir-data-v3/database.bin |
| `account-mapping.bin` | 8,440,366,872 | https://pir.53627.org/mainnet-pir-data-v3/account-mapping.bin |
| `storage-mapping.bin` | 83,015,179,744 | https://pir.53627.org/mainnet-pir-data-v3/storage-mapping.bin |
| `code_store.bin` | 66,064,516 | https://pir.53627.org/mainnet-pir-data-v3/code_store.bin |
| `manifest.json` | 1,124 | https://pir.53627.org/mainnet-pir-data-v3/manifest.json |
| `metadata.json` | 237 | https://pir.53627.org/mainnet-pir-data-v3/metadata.json |

## Regression Dataset (Legacy)

Legacy v2 snapshot (see `README.md` for URLs). Check `metadata.json` for `schema_version` and `entry_size_bytes`.
