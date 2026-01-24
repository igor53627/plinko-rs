# Developer Optimization Guide: plinko-rs

**Status:** `DRAFT-V4` (Strict PMNS Alignment)
**Context:** This guide outlines optimization strategies for the `plinko-rs` codebase using hardware acceleration (AES-NI/AVX) and zero-copy memory management.

## ⚠️ Critical Architecture Warnings

1.  **Do NOT Replace iPRF Offsets:** The AES-NI primitive below is intended *only* for internal randomness expansion (e.g., generating hint vectors from seeds). It **must not** replace the canonical iPRF offset generation, which is defined as `iPRF = PRP ∘ PMNS` in `docs/Plinko.v`. Replacing structural offsets with a raw PRF stream will break correctness.
2.  **PMNS Node Encoding:** When using the PRF for node-dependent values, the input `node_id` **must** be the canonical PMNS node encoding (i.e., `encode_node(low, high, n)`), matching `docs/Plinko.v:70-90`. Using a sequential loop counter or index will destroy the PMNS binomial split semantics.

---

## 1. Hardware-Accelerated PRF (AES-NI)

**Objective:** Use hardware AES-NI instructions to generate pseudorandom bytes for seed expansion, achieving ~2GB/s+ throughput.

**Canonical constraint (docs/Plinko.v):**
- `node_id` must be the PMNS node encoding (`encode_node(low, high, n)`), so each PMNS node gets its own PRF output.
- Do **not** use a sequential counter/stream across nodes or replace iPRF offsets with a raw PRF stream.

### Implementation
*Add to `Cargo.toml`: `aes`, `ctr`*

```rust
// src/crypto.rs
use aes::cipher::{KeyIvInit, StreamCipher};
use aes::Aes128;
use ctr::Ctr128BE;

type Aes128Ctr = Ctr128BE<Aes128>;

/// Generates pseudorandom bytes using hardware AES-NI.
/// 
/// # Security & Correctness
/// - `seed`: The subset seed or block key (16 bytes).
/// - `domain_sep`: Context separation (e.g., 0x1 for hints, 0x2 for internal nodes).
/// - `node_id`: CRITICAL. This must be the canonical PMNS node encoding
///    (e.g., `encode_node(low, high, n)`). Do NOT use a sequential loop counter.
///    Using a non-canonical ID will break the PMNS dependency structure.
/// 
/// # Performance
/// Uses AES-NI instructions if available. Throughput ~2GB/s+.
#[inline(always)]
pub fn expand_node_randomness_aes(
    seed: &[u8; 16], 
    domain_sep: u64, 
    node_id: u64, 
    output: &mut [u8]
) {
    // Construct IV: [Domain Separator (8B) | Node ID (8B)]
    // This guarantees distinct streams per PMNS node.
    let mut iv = [0u8; 16];
    iv[0..8].copy_from_slice(&domain_sep.to_be_bytes());
    iv[8..16].copy_from_slice(&node_id.to_be_bytes());

    // Initialize AES-CTR.
    let mut cipher = Aes128Ctr::new(seed.into(), &iv.into());
    
    // Encrypting zeros produces the pure keystream.
    cipher.apply_keystream(output);
}
```

---

## 2. SIMD Vectorization (AVX2/AVX-512)

**Objective:** Maximize throughput for the "hot loop" (XORing massive hint vectors) by bypassing conservative compiler auto-vectorization.

### Implementation

```rust
// src/simd.rs
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// XORs 32 bytes using AVX2 instructions.
/// 
/// # Safety
/// Caller must verify `is_x86_feature_detected!("avx2")`.
#[target_feature(enable = "avx2")]
pub unsafe fn xor_block_avx2(dst: &mut [u8; 32], src: &[u8; 32]) {
    // Use unaligned loads (loadu) to ensure safety regardless of memory alignment.
    let v_dst = _mm256_loadu_si256(dst.as_ptr() as *const __m256i);
    let v_src = _mm256_loadu_si256(src.as_ptr() as *const __m256i);
    
    // Perform 256-bit XOR (32 bytes at once)
    let res = _mm256_xor_si256(v_dst, v_src);
    
    // Store result back
    _mm256_storeu_si256(dst.as_mut_ptr() as *mut __m256i, res);
}
```

---

## 3. Memory-Safe Zero-Copy Database

**Objective:** Instantaneous loading of 100GB+ datasets via `mmap`.
**Correctness:** Strictly enforces 32-byte alignment to prevent index drift and owns the memory mapping to prevent use-after-free bugs.

### Implementation
*Add to `Cargo.toml`: `memmap2`*

```rust
// src/storage.rs
use memmap2::Mmap;
use std::fs::File;
use std::slice;
use std::io::{self, Error, ErrorKind};

/// A wrapper around a memory-mapped file that ensures memory safety and alignment.
pub struct Database {
    // The Mmap acts as the owner of the memory.
    _mmap: Mmap, 
    entries_ptr: *const u8,
    count: usize,
}

// Canonical Plinko Entry size (32 bytes).
const ENTRY_SIZE: usize = 32;

impl Database {
    pub fn open(path: &str) -> io::Result<Self> {
        let file = File::open(path)?;
        
        // SAFETY: We assume the file is immutable while mapped.
        let mmap = unsafe { Mmap::map(&file)? };
        
        // VALIDATION: Database length must be a multiple of 32 bytes (Plinko word size).
        // Divisibility by w is handled at the geometry layer; silent truncation here would
        // shift indices and break retrieval.
        if mmap.len() % ENTRY_SIZE != 0 {
            return Err(Error::new(
                ErrorKind::InvalidData, 
                "Database file size must be a multiple of 32 bytes (Plinko Word Size)."
            ));
        }

        let count = mmap.len() / ENTRY_SIZE;
        let entries_ptr = mmap.as_ptr();

        Ok(Self {
            _mmap: mmap,
            entries_ptr,
            count,
        })
    }

    /// Returns a slice of the data.
    /// The lifetime 'a is tied to &self, ensuring the map is valid during access.
    pub fn get_entries<'a>(&'a self) -> &'a [[u8; ENTRY_SIZE]] {
        unsafe {
            slice::from_raw_parts(
                self.entries_ptr as *const [u8; ENTRY_SIZE],
                self.count
            )
        }
    }
}
```

---

## 4. CPU Cache Prefetching

**Objective:** Reduce L1/L2 cache misses during linear scans of hint vectors.

### Implementation

```rust
// src/processing.rs
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[inline(always)]
pub unsafe fn prefetch_ahead(ptr: *const u8, offset_bytes: usize) {
    // _MM_HINT_T0: Prefetch into all cache levels (temporal data).
    _mm_prefetch(ptr.add(offset_bytes) as *const i8, _MM_HINT_T0);
}
```

---

## 5. Build Configuration

To enable these intrinsics, compile with native CPU features:

```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release
```
