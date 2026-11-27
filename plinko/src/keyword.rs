//! Keyword PIR table core: 2-segment Key-Value Segmented Cuckoo Filter (`KV-SCF`).
//!
//! Reduces private key-value lookup to 2 index PIR queries (one per segment) as
//! specified in `docs/keyword_lookup.md` (adapted from RisePIR, `ePrint 2026/2082`).
//!
//! Layout on disk (`database.bin`):
//! - `database.bin = segment_0 | segment_1` (`2 * s * record_size` bytes)
//! - `segment_j = bucket[0] | ... | bucket[s - 1]` (`s >= 1` buckets)
//! - `bucket = slot[0] | ... | slot[b - 1]` (`record_size = b * (tag_size + value_size)`)
//! - `slot = tag (tag_size bytes) | value (value_size bytes)`
//!
//! An empty slot is all bytes zero. Keyed BLAKE3 derives `(a0, tag)` from
//! `(table_seed, kind, key)` and the alternate bucket `a1 = (a0 + h1(tag)) mod s`
//! from `tag` alone, enabling partial-key cuckoo eviction during build and live
//! inserts without storing original keys or an `other_bucket` side array.

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::io::{self, Write};
#[cfg(not(target_arch = "wasm32"))]
use std::path::Path;

/// Number of disjoint cuckoo segments (`d = 2`).
pub const NUM_SEGMENTS: usize = 2;

/// Default number of slots per bucket (`b = 4`).
pub const DEFAULT_BUCKET_SIZE: usize = 4;

/// Default tag width in bytes (`8` bytes = 64 bits).
pub const DEFAULT_TAG_SIZE: usize = 8;

/// Minimum supported tag width in bytes (`4` bytes = 32 bits).
pub const MIN_TAG_SIZE: usize = 4;

/// Maximum supported tag width in bytes (`32` bytes = 256 bits).
pub const MAX_TAG_SIZE: usize = 32;

/// Default random-walk eviction budget per insert.
pub const DEFAULT_MAX_KICKS: usize = 500;

const HASH_DOMAIN_KEY: &[u8] = b"plinko-kvscf-v1";
const HASH_DOMAIN_ALT: &[u8] = b"plinko-kvscf-alt-v1";
const HASH_DOMAIN_RNG: &[u8] = b"plinko-kvscf-rng-v1";

/// Errors returned by [`KeywordTableSpec`] and [`KvScfTable`] operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KeywordError {
    /// Table parameters are invalid or overflow `usize`.
    InvalidParams(&'static str),
    /// Supplied key length does not match `KeywordTableSpec::key_size`.
    KeySizeMismatch {
        /// Configured fixed key size.
        expected: usize,
        /// Provided key length.
        actual: usize,
    },
    /// Supplied key exceeds `u32::MAX` bytes.
    KeyTooLong {
        /// Provided key length.
        actual: usize,
    },
    /// Supplied value length does not match `KeywordTableSpec::value_size`.
    ValueSizeMismatch {
        /// Configured fixed value size.
        expected: usize,
        /// Provided value length.
        actual: usize,
    },
    /// Supplied bucket buffer does not match `KeywordTableSpec::record_size()`.
    BucketSizeMismatch {
        /// Configured bucket record size.
        expected: usize,
        /// Provided buffer length.
        actual: usize,
    },
    /// Supplied tag does not match `KeywordTableSpec::tag_size` or is all-zero.
    InvalidTag,
    /// Supplied raw `database.bin` buffer length does not match `database_bytes()`.
    DatabaseSizeMismatch {
        /// Expected byte length.
        expected: usize,
        /// Provided byte length.
        actual: usize,
    },
    /// Loaded `database.bin` contains malformed or non-canonical slot data.
    CorruptDatabase(&'static str),
    /// A slot or stash entry with the same `(a0, tag)` already exists.
    DuplicateKey,
    /// Target key was not found during `update` or `delete`.
    KeyNotFound,
    /// Lookup observed more than one matching slot across the two buckets and stash.
    DuplicateTagMatch,
    /// Eviction random walk exhausted `max_kicks` and stash capacity is full or disabled.
    TableFull,
    /// Build exhausted `max_seed_attempts` without placing all keys.
    SeedAttemptsExhausted {
        /// Number of seeds tried.
        attempts: usize,
    },
}

impl fmt::Display for KeywordError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidParams(msg) => write!(f, "invalid keyword table parameters: {msg}"),
            Self::KeySizeMismatch { expected, actual } => {
                write!(f, "key length mismatch: expected {expected}, got {actual}")
            }
            Self::KeyTooLong { actual } => {
                write!(f, "key length {actual} exceeds u32::MAX")
            }
            Self::ValueSizeMismatch { expected, actual } => {
                write!(
                    f,
                    "value length mismatch: expected {expected}, got {actual}"
                )
            }
            Self::BucketSizeMismatch { expected, actual } => {
                write!(
                    f,
                    "bucket length mismatch: expected {expected}, got {actual}"
                )
            }
            Self::InvalidTag => write!(f, "invalid or all-zero tag"),
            Self::DatabaseSizeMismatch { expected, actual } => {
                write!(
                    f,
                    "database byte length mismatch: expected {expected}, got {actual}"
                )
            }
            Self::CorruptDatabase(msg) => write!(f, "corrupt keyword database: {msg}"),
            Self::DuplicateKey => write!(f, "key or (bucket_seg0, tag) already exists"),
            Self::KeyNotFound => write!(f, "key not found in table or runtime stash"),
            Self::DuplicateTagMatch => {
                write!(f, "multiple matching tags found across candidate buckets")
            }
            Self::TableFull => write!(f, "table full: eviction walk exhausted max_kicks"),
            Self::SeedAttemptsExhausted { attempts } => {
                write!(
                    f,
                    "failed to build table without stash after {attempts} seeds"
                )
            }
        }
    }
}

impl std::error::Error for KeywordError {}

/// Failure returned by [`KvScfTable::build`] naming the first unplaceable key.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BuildError {
    /// Zero-based index of the failing input entry.
    pub entry_index: usize,
    /// Namespace byte of the failing key.
    pub kind: u8,
    /// Raw bytes of the key that could not be placed.
    pub key: Vec<u8>,
    /// Underlying insertion failure (`TableFull`, `DuplicateKey`, size mismatch, etc.).
    pub error: KeywordError,
}

impl fmt::Display for BuildError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "failed to place entry #{} (kind=0x{:02x}, key_len={}): {}",
            self.entry_index,
            self.kind,
            self.key.len(),
            self.error
        )
    }
}

impl std::error::Error for BuildError {}

/// Public geometry and metadata for a 2-segment `KV-SCF` table.
///
/// All geometry invariants (`buckets_per_segment >= 1`, `bucket_size >= 1`,
/// `4 <= tag_size <= 32`, `value_size >= 1`, no byte-length overflow) are
/// validated by [`Self::new`] and enforced on deserialization.
///
/// `state_epoch` is owned by the batch publisher: individual mutations on
/// [`KvScfTable`] do not auto-increment it; call [`KvScfTable::advance_state_epoch`]
/// once per published delta/stash batch.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(
    feature = "cli",
    derive(serde::Serialize, serde::Deserialize),
    serde(try_from = "KeywordTableSpecRaw")
)]
pub struct KeywordTableSpec {
    table_seed: [u8; 32],
    table_epoch: u64,
    state_epoch: u64,
    buckets_per_segment: usize,
    bucket_size: usize,
    tag_size: usize,
    value_size: usize,
    key_size: Option<usize>,
}

#[cfg(feature = "cli")]
#[derive(serde::Deserialize)]
struct KeywordTableSpecRaw {
    table_seed: [u8; 32],
    table_epoch: u64,
    state_epoch: u64,
    buckets_per_segment: usize,
    bucket_size: usize,
    tag_size: usize,
    value_size: usize,
    key_size: Option<usize>,
}

#[cfg(feature = "cli")]
impl TryFrom<KeywordTableSpecRaw> for KeywordTableSpec {
    type Error = KeywordError;

    fn try_from(raw: KeywordTableSpecRaw) -> Result<Self, Self::Error> {
        Self::new(
            raw.buckets_per_segment,
            raw.bucket_size,
            raw.tag_size,
            raw.value_size,
            raw.key_size,
            raw.table_seed,
            raw.table_epoch,
            raw.state_epoch,
        )
    }
}

impl KeywordTableSpec {
    /// Creates and validates a [`KeywordTableSpec`] from explicit parameters.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        buckets_per_segment: usize,
        bucket_size: usize,
        tag_size: usize,
        value_size: usize,
        key_size: Option<usize>,
        table_seed: [u8; 32],
        table_epoch: u64,
        state_epoch: u64,
    ) -> Result<Self, KeywordError> {
        if buckets_per_segment == 0 {
            return Err(KeywordError::InvalidParams(
                "buckets_per_segment must be >= 1",
            ));
        }
        if bucket_size == 0 {
            return Err(KeywordError::InvalidParams("bucket_size must be >= 1"));
        }
        if !(MIN_TAG_SIZE..=MAX_TAG_SIZE).contains(&tag_size) {
            return Err(KeywordError::InvalidParams(
                "tag_size must be between 4 and 32 bytes",
            ));
        }
        if value_size == 0 {
            return Err(KeywordError::InvalidParams("value_size must be >= 1"));
        }
        if let Some(ks) = key_size {
            if u32::try_from(ks).is_err() {
                return Err(KeywordError::InvalidParams("key_size must fit in u32"));
            }
        }
        let slot_size: usize = tag_size
            .checked_add(value_size)
            .ok_or(KeywordError::InvalidParams("slot_size overflow"))?;
        let record_size: usize = bucket_size
            .checked_mul(slot_size)
            .ok_or(KeywordError::InvalidParams("record_size overflow"))?;
        let total_buckets: usize = buckets_per_segment
            .checked_mul(NUM_SEGMENTS)
            .ok_or(KeywordError::InvalidParams("total_buckets overflow"))?;
        let _total_slots: usize = total_buckets
            .checked_mul(bucket_size)
            .ok_or(KeywordError::InvalidParams("total_slots overflow"))?;
        let db_bytes: usize = total_buckets
            .checked_mul(record_size)
            .ok_or(KeywordError::InvalidParams("database_bytes overflow"))?;
        if db_bytes > isize::MAX as usize {
            return Err(KeywordError::InvalidParams(
                "database_bytes exceeds isize::MAX",
            ));
        }

        Ok(Self {
            table_seed,
            table_epoch,
            state_epoch,
            buckets_per_segment,
            bucket_size,
            tag_size,
            value_size,
            key_size,
        })
    }

    /// Derives `buckets_per_segment = max(1, ceil(ceil(num_keys / (b * load)) / 2))`
    /// using default layout (`b = 4`, `tag_size = 8`, arbitrary `key_size`).
    pub fn for_capacity(
        num_keys: usize,
        value_size: usize,
        load: f64,
        table_seed: [u8; 32],
    ) -> Result<Self, KeywordError> {
        Self::with_layout(
            num_keys,
            None,
            value_size,
            DEFAULT_TAG_SIZE,
            DEFAULT_BUCKET_SIZE,
            load,
            table_seed,
        )
    }

    /// Derives `buckets_per_segment` from `(num_keys, bucket_size, load)` with
    /// explicit key, value, tag, and bucket parameters.
    pub fn with_layout(
        num_keys: usize,
        key_size: Option<usize>,
        value_size: usize,
        tag_size: usize,
        bucket_size: usize,
        load: f64,
        table_seed: [u8; 32],
    ) -> Result<Self, KeywordError> {
        if !load.is_finite() || load <= 0.0 || load >= 1.0 {
            return Err(KeywordError::InvalidParams(
                "load must be finite and strictly in (0.0, 1.0)",
            ));
        }
        if bucket_size == 0 {
            return Err(KeywordError::InvalidParams("bucket_size must be >= 1"));
        }

        let buckets_per_segment: usize = if num_keys == 0 {
            1
        } else {
            let denom: f64 = (bucket_size as f64) * load;
            let total_buckets_f: f64 = (num_keys as f64 / denom).ceil();
            if !total_buckets_f.is_finite() || total_buckets_f > (usize::MAX as f64) {
                return Err(KeywordError::InvalidParams(
                    "derived total_buckets overflow",
                ));
            }
            let total_buckets: usize = (total_buckets_f as usize).max(NUM_SEGMENTS);
            total_buckets.div_ceil(NUM_SEGMENTS).max(1)
        };

        Self::new(
            buckets_per_segment,
            bucket_size,
            tag_size,
            value_size,
            key_size,
            table_seed,
            0,
            0,
        )
    }

    /// Public 32-byte BLAKE3 key shared by builder, server, and clients.
    #[inline]
    pub fn table_seed(&self) -> &[u8; 32] {
        &self.table_seed
    }

    /// Major epoch: increments on seed or geometry change (requires fresh `HintInit`).
    #[inline]
    pub fn table_epoch(&self) -> u64 {
        self.table_epoch
    }

    /// Minor epoch: incremented once per published delta/stash batch.
    #[inline]
    pub fn state_epoch(&self) -> u64 {
        self.state_epoch
    }

    /// Number of buckets in each of the 2 segments (`s >= 1`).
    #[inline]
    pub fn buckets_per_segment(&self) -> usize {
        self.buckets_per_segment
    }

    /// Number of slots packed into each bucket (`b >= 1`).
    #[inline]
    pub fn bucket_size(&self) -> usize {
        self.bucket_size
    }

    /// Tag width in bytes (`4..=32`).
    #[inline]
    pub fn tag_size(&self) -> usize {
        self.tag_size
    }

    /// Fixed value width in bytes (`>= 1`).
    #[inline]
    pub fn value_size(&self) -> usize {
        self.value_size
    }

    /// Optional fixed key length in bytes (`None` accepts arbitrary-length keys).
    #[inline]
    pub fn key_size(&self) -> Option<usize> {
        self.key_size
    }

    /// Bytes per slot (`tag_size + value_size`).
    #[inline]
    pub fn slot_size(&self) -> usize {
        self.tag_size + self.value_size
    }

    /// Bytes per bucket / PIR record (`bucket_size * slot_size`).
    #[inline]
    pub fn record_size(&self) -> usize {
        self.bucket_size * self.slot_size()
    }

    /// Total number of buckets across both segments (`2 * buckets_per_segment`).
    #[inline]
    pub fn total_buckets(&self) -> usize {
        NUM_SEGMENTS * self.buckets_per_segment
    }

    /// Total number of slots across both segments (`2 * buckets_per_segment * bucket_size`).
    #[inline]
    pub fn total_slots(&self) -> usize {
        self.total_buckets() * self.bucket_size
    }

    /// Byte length of one segment (`buckets_per_segment * record_size`).
    #[inline]
    pub fn segment_bytes(&self) -> usize {
        self.buckets_per_segment * self.record_size()
    }

    /// Total byte length of `database.bin` (`2 * segment_bytes`).
    #[inline]
    pub fn database_bytes(&self) -> usize {
        NUM_SEGMENTS * self.segment_bytes()
    }

    /// Number of index PIR queries per keyword lookup (`2`).
    #[inline]
    pub const fn queries_per_lookup(&self) -> usize {
        NUM_SEGMENTS
    }

    /// Conservative union-bound non-member false-match probability over the
    /// `d * b` bucket slots (excluding any public runtime-stash), accounting for
    /// the zero-tag remap (`[0; tag_size] -> [1, 0, ...]`, which adds at most 1
    /// extra raw digest preimage): `(d * b + 1) / 2^(8 * tag_size)`.
    pub fn false_match_upper_bound(&self) -> f64 {
        let bits: i32 = (self.tag_size * 8) as i32;
        ((NUM_SEGMENTS * self.bucket_size) + 1) as f64 / 2.0_f64.powi(bits)
    }

    /// Computes candidate bucket offsets `(bucket_seg0, bucket_seg1)` and `tag`
    /// for `(kind, key)` strictly from public table parameters.
    pub fn locate(&self, kind: u8, key: &[u8]) -> Result<KeyLocation, KeywordError> {
        let mut tag_buf: [u8; MAX_TAG_SIZE] = [0u8; MAX_TAG_SIZE];
        let (bucket_seg0, bucket_seg1) = self.locate_into_buf(kind, key, &mut tag_buf)?;
        Ok(KeyLocation {
            bucket_seg0,
            bucket_seg1,
            tag: tag_buf[..self.tag_size].to_vec(),
        })
    }

    /// Stack-only version of [`Self::locate`] writing the tag into `tag_buf[..self.tag_size]`.
    fn locate_into_buf(
        &self,
        kind: u8,
        key: &[u8],
        tag_buf: &mut [u8; MAX_TAG_SIZE],
    ) -> Result<(usize, usize), KeywordError> {
        let key_hasher_base: blake3::Hasher = make_key_hasher_base(&self.table_seed);
        let alt_hasher_base: blake3::Hasher = make_alt_hasher_base(&self.table_seed);
        self.locate_with_bases(kind, key, &key_hasher_base, &alt_hasher_base, tag_buf)
    }

    fn locate_with_bases(
        &self,
        kind: u8,
        key: &[u8],
        key_hasher_base: &blake3::Hasher,
        alt_hasher_base: &blake3::Hasher,
        tag_buf: &mut [u8; MAX_TAG_SIZE],
    ) -> Result<(usize, usize), KeywordError> {
        if let Some(expected) = self.key_size {
            if key.len() != expected {
                return Err(KeywordError::KeySizeMismatch {
                    expected,
                    actual: key.len(),
                });
            }
        }
        let key_len_u32: u32 =
            u32::try_from(key.len()).map_err(|_| KeywordError::KeyTooLong { actual: key.len() })?;

        let mut h: blake3::Hasher = key_hasher_base.clone();
        h.update(&[kind]);
        h.update(&key_len_u32.to_le_bytes());
        h.update(key);

        let mut out: [u8; 8 + MAX_TAG_SIZE] = [0u8; 8 + MAX_TAG_SIZE];
        h.finalize_xof().fill(&mut out[..8 + self.tag_size]);

        let raw_a0: u64 = u64::from_le_bytes(out[0..8].try_into().unwrap());
        let s: u64 = self.buckets_per_segment as u64;
        let bucket_seg0: usize = (raw_a0 % s) as usize;

        tag_buf[..self.tag_size].copy_from_slice(&out[8..8 + self.tag_size]);
        if is_zero_slice(&tag_buf[..self.tag_size]) {
            tag_buf[0] = 1;
        }

        let bucket_seg1: usize =
            self.alt_bucket_with_base(0, bucket_seg0, &tag_buf[..self.tag_size], alt_hasher_base);
        Ok((bucket_seg0, bucket_seg1))
    }

    /// Computes the alternate bucket offset in segment `1 - segment` strictly from
    /// `(segment, bucket, tag)` using modular partial-key cuckoo hashing.
    pub fn alt_bucket(&self, segment: usize, bucket: usize, tag: &[u8]) -> usize {
        let alt_hasher_base: blake3::Hasher = make_alt_hasher_base(&self.table_seed);
        self.alt_bucket_with_base(segment, bucket, tag, &alt_hasher_base)
    }

    #[inline]
    fn alt_bucket_with_base(
        &self,
        segment: usize,
        bucket: usize,
        tag: &[u8],
        alt_hasher_base: &blake3::Hasher,
    ) -> usize {
        debug_assert!(segment < NUM_SEGMENTS);
        debug_assert!(bucket < self.buckets_per_segment);
        debug_assert_eq!(tag.len(), self.tag_size);

        let mut h: blake3::Hasher = alt_hasher_base.clone();
        h.update(tag);
        let digest: blake3::Hash = h.finalize();
        let raw_delta: u64 = u64::from_le_bytes(digest.as_bytes()[0..8].try_into().unwrap());
        let s: u64 = self.buckets_per_segment as u64;
        let delta: u64 = raw_delta % s;
        let b: u64 = bucket as u64;

        if segment == 0 {
            let sum: u128 = b as u128 + delta as u128;
            (sum % s as u128) as usize
        } else if b >= delta {
            (b - delta) as usize
        } else {
            (b + (s - delta)) as usize
        }
    }

    /// Scans the two recovered candidate buckets and an optional runtime-stash match,
    /// combining all matches before deciding `None`, `Some(value)`, or `DuplicateTagMatch`.
    pub fn scan_buckets<'a>(
        &self,
        bucket0: &'a [u8],
        bucket1: &'a [u8],
        tag: &[u8],
        stash_match: Option<&'a [u8]>,
    ) -> Result<Option<&'a [u8]>, KeywordError> {
        let rec: usize = self.record_size();
        if bucket0.len() != rec {
            return Err(KeywordError::BucketSizeMismatch {
                expected: rec,
                actual: bucket0.len(),
            });
        }
        if bucket1.len() != rec {
            return Err(KeywordError::BucketSizeMismatch {
                expected: rec,
                actual: bucket1.len(),
            });
        }
        if tag.len() != self.tag_size || is_zero_slice(tag) {
            return Err(KeywordError::InvalidTag);
        }

        let slot_size: usize = self.slot_size();
        let mut found: Option<&'a [u8]> = None;

        for bucket in [bucket0, bucket1] {
            for slot_idx in 0..self.bucket_size {
                let start: usize = slot_idx * slot_size;
                let slot_tag: &[u8] = &bucket[start..start + self.tag_size];
                if slot_tag == tag {
                    if found.is_some() {
                        return Err(KeywordError::DuplicateTagMatch);
                    }
                    found = Some(&bucket[start + self.tag_size..start + slot_size]);
                }
            }
        }

        if let Some(sv) = stash_match {
            if sv.len() != self.value_size {
                return Err(KeywordError::ValueSizeMismatch {
                    expected: self.value_size,
                    actual: sv.len(),
                });
            }
            if found.is_some() {
                return Err(KeywordError::DuplicateTagMatch);
            }
            found = Some(sv);
        }

        Ok(found)
    }
}

/// Candidate bucket coordinates and tag for a single key.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KeyLocation {
    /// Candidate bucket offset in segment 0 (`0..buckets_per_segment`).
    pub bucket_seg0: usize,
    /// Candidate bucket offset in segment 1 (`0..buckets_per_segment`).
    pub bucket_seg1: usize,
    /// Non-zero tag bytes (`tag_size` bytes).
    pub tag: Vec<u8>,
}

impl KeyLocation {
    /// Returns the two distinct record indices in a concatenated
    /// `database.bin = segment_0 | segment_1` file: `[bucket_seg0, s + bucket_seg1]`.
    #[inline]
    pub fn global_indices(&self, spec: &KeywordTableSpec) -> [usize; NUM_SEGMENTS] {
        [
            self.bucket_seg0,
            spec.buckets_per_segment + self.bucket_seg1,
        ]
    }
}

/// Net XOR delta for one touched bucket in segment `0` or `1`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BucketDelta {
    /// Segment index (`0` or `1`).
    pub segment: usize,
    /// Bucket index within `segment` (`0..buckets_per_segment`).
    pub bucket_idx: usize,
    /// Bytewise XOR difference (`old_bucket XOR new_bucket`, length `record_size`).
    pub xor_delta: Vec<u8>,
}

impl BucketDelta {
    /// Returns the global bucket index inside a concatenated `database.bin`.
    #[inline]
    pub fn global_index(&self, spec: &KeywordTableSpec) -> usize {
        self.segment * spec.buckets_per_segment + self.bucket_idx
    }
}

/// Entry in the optional bounded public runtime-stash, keyed by `(bucket_seg0, tag)`.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "cli", derive(serde::Serialize, serde::Deserialize))]
pub struct StashEntry {
    /// Primary segment-0 bucket offset `a0` of the stashed key.
    pub bucket_seg0: usize,
    /// Non-zero tag of the stashed key.
    pub tag: Vec<u8>,
    /// Fixed-size value bytes.
    pub value: Vec<u8>,
}

/// Mutation applied to the optional public runtime-stash.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StashOp {
    /// Appends a new entry to the runtime-stash after `TableFull` fallback.
    Insert(StashEntry),
    /// Replaces the value of an existing `(bucket_seg0, tag)` stash entry.
    Update {
        /// Primary segment-0 bucket offset `a0`.
        bucket_seg0: usize,
        /// Matching tag bytes.
        tag: Vec<u8>,
        /// Replacement value bytes.
        value: Vec<u8>,
    },
    /// Removes an existing `(bucket_seg0, tag)` stash entry.
    Delete {
        /// Primary segment-0 bucket offset `a0`.
        bucket_seg0: usize,
        /// Matching tag bytes.
        tag: Vec<u8>,
    },
}

/// Result of a dynamic `insert`, `update`, `delete`, or `upsert` on [`KvScfTable`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MutationOutcome {
    /// Net XOR bucket deltas sorted by `(segment, bucket_idx)`.
    pub deltas: Vec<BucketDelta>,
    /// Optional runtime-stash mutation (only when runtime-stash is enabled).
    pub stash_op: Option<StashOp>,
    /// Number of cuckoo eviction steps performed (0 for direct placement / update / delete).
    pub kicks: usize,
}

/// Summary statistics from [`KvScfTable::build`].
#[derive(Debug, Clone, PartialEq)]
pub struct BuildStats {
    /// Number of keys inserted into buckets.
    pub num_keys: usize,
    /// Achieved bucket load factor `num_keys / total_slots`.
    pub load_factor: f64,
    /// Total eviction kicks across the whole build.
    pub total_kicks: u64,
    /// Maximum eviction kicks observed for any single key.
    pub max_kicks_observed: usize,
    /// Mean kicks over the last 1% of inserted keys (at near-final load).
    pub tail_mean_kicks: f64,
    /// 99th-percentile kicks over the last 1% of inserted keys.
    pub tail_p99_kicks: usize,
    /// Maximum kicks over the last 1% of inserted keys.
    pub tail_max_kicks: usize,
}

/// In-memory 2-segment `KV-SCF` table supporting build, live mutations, and
/// `database.bin` export.
///
/// The backing byte array stores `segment_0` immediately followed by `segment_1`,
/// so [`Self::as_bytes`] is directly a valid `database.bin` and
/// [`Self::segment_bytes`] returns zero-copy slices for each segment PIR instance.
pub struct KvScfTable {
    spec: KeywordTableSpec,
    data: Vec<u8>,
    bucket_entries: usize,
    stash: Vec<StashEntry>,
    max_stash: usize,
    max_kicks: usize,
    key_hasher_base: blake3::Hasher,
    alt_hasher_base: blake3::Hasher,
    eviction_rng: ChaCha20Rng,
}

impl fmt::Debug for KvScfTable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("KvScfTable")
            .field("spec", &self.spec)
            .field("bucket_entries", &self.bucket_entries)
            .field("stash_len", &self.stash.len())
            .field("max_stash", &self.max_stash)
            .field("max_kicks", &self.max_kicks)
            .finish()
    }
}

impl KvScfTable {
    /// Allocates an empty table for `spec` with `max_kicks = 500` and runtime-stash disabled.
    pub fn new(spec: KeywordTableSpec) -> Self {
        let key_hasher_base: blake3::Hasher = make_key_hasher_base(&spec.table_seed);
        let alt_hasher_base: blake3::Hasher = make_alt_hasher_base(&spec.table_seed);
        let eviction_rng: ChaCha20Rng = derive_eviction_rng(&spec.table_seed);
        let data: Vec<u8> = vec![0u8; spec.database_bytes()];
        Self {
            spec,
            data,
            bucket_entries: 0,
            stash: Vec::new(),
            max_stash: 0,
            max_kicks: DEFAULT_MAX_KICKS,
            key_hasher_base,
            alt_hasher_base,
            eviction_rng,
        }
    }

    /// Reconstructs a [`KvScfTable`] from an existing `database.bin` buffer.
    ///
    /// Rejects non-canonical empty slots (all-zero tag with non-zero value bytes)
    /// and duplicate `(bucket_seg0, tag)` identities across segments or slots.
    pub fn from_database_bytes(
        spec: KeywordTableSpec,
        data: Vec<u8>,
    ) -> Result<Self, KeywordError> {
        let expected: usize = spec.database_bytes();
        if data.len() != expected {
            return Err(KeywordError::DatabaseSizeMismatch {
                expected,
                actual: data.len(),
            });
        }
        let key_hasher_base: blake3::Hasher = make_key_hasher_base(&spec.table_seed);
        let alt_hasher_base: blake3::Hasher = make_alt_hasher_base(&spec.table_seed);
        let slot_size: usize = spec.slot_size();
        let tag_size: usize = spec.tag_size;
        let rec: usize = spec.record_size();
        let s: usize = spec.buckets_per_segment;
        let mut bucket_entries: usize = 0;
        let mut seen: HashSet<(usize, Vec<u8>)> = HashSet::new();

        for seg in 0..NUM_SEGMENTS {
            for bidx in 0..s {
                let b_off: usize = (seg * s + bidx) * rec;
                let bucket: &[u8] = &data[b_off..b_off + rec];
                for sidx in 0..spec.bucket_size {
                    let start: usize = sidx * slot_size;
                    let slot: &[u8] = &bucket[start..start + slot_size];
                    let tag: &[u8] = &slot[..tag_size];
                    if is_zero_slice(tag) {
                        if !is_zero_slice(&slot[tag_size..]) {
                            return Err(KeywordError::CorruptDatabase(
                                "empty slot (all-zero tag) has non-zero value bytes",
                            ));
                        }
                    } else {
                        let a0: usize = if seg == 0 {
                            bidx
                        } else {
                            spec.alt_bucket_with_base(1, bidx, tag, &alt_hasher_base)
                        };
                        if !seen.insert((a0, tag.to_vec())) {
                            return Err(KeywordError::DuplicateTagMatch);
                        }
                        bucket_entries += 1;
                    }
                }
            }
        }

        let eviction_rng: ChaCha20Rng = derive_eviction_rng(&spec.table_seed);
        Ok(Self {
            spec,
            data,
            bucket_entries,
            stash: Vec::new(),
            max_stash: 0,
            max_kicks: DEFAULT_MAX_KICKS,
            key_hasher_base,
            alt_hasher_base,
            eviction_rng,
        })
    }

    /// Sets the random-walk eviction budget per insert (must be >= 1; clamped to 1 if 0).
    pub fn with_max_kicks(mut self, max_kicks: usize) -> Self {
        self.max_kicks = max_kicks.max(1);
        self
    }

    /// Enables a bounded runtime-stash of at most `max_stash` entries for post-build inserts.
    ///
    /// Build-time construction via [`Self::build`] always keeps the stash disabled (`0`).
    pub fn with_max_stash(mut self, max_stash: usize) -> Self {
        self.max_stash = max_stash;
        self
    }

    /// Builds a stash-free table from an iterator of `(kind, key, value)` entries.
    ///
    /// Returns [`BuildError`] naming the first key that cannot be placed or collides.
    pub fn build<'a, I>(
        spec: KeywordTableSpec,
        entries: I,
    ) -> Result<(Self, BuildStats), BuildError>
    where
        I: IntoIterator<Item = (u8, &'a [u8], &'a [u8])>,
    {
        Self::build_with_max_kicks(spec, DEFAULT_MAX_KICKS, entries)
    }

    /// Builds a stash-free table with a custom `max_kicks` budget.
    pub fn build_with_max_kicks<'a, I>(
        spec: KeywordTableSpec,
        max_kicks: usize,
        entries: I,
    ) -> Result<(Self, BuildStats), BuildError>
    where
        I: IntoIterator<Item = (u8, &'a [u8], &'a [u8])>,
    {
        let mut table: Self = Self::new(spec).with_max_kicks(max_kicks);
        let iter = entries.into_iter();
        let (lower_hint, _) = iter.size_hint();
        let mut kick_samples: Vec<usize> = Vec::with_capacity(lower_hint);
        let mut total_kicks: u64 = 0;
        let mut max_kicks_observed: usize = 0;

        for (entry_index, (kind, key, value)) in iter.enumerate() {
            match table.insert_bulk(kind, key, value) {
                Ok(kicks) => {
                    total_kicks += kicks as u64;
                    if kicks > max_kicks_observed {
                        max_kicks_observed = kicks;
                    }
                    kick_samples.push(kicks);
                }
                Err(error) => {
                    return Err(BuildError {
                        entry_index,
                        kind,
                        key: key.to_vec(),
                        error,
                    });
                }
            }
        }

        let num_keys: usize = kick_samples.len();
        let (tail_mean_kicks, tail_p99_kicks, tail_max_kicks) = if num_keys == 0 {
            (0.0, 0, 0)
        } else {
            let tail_len: usize = (num_keys / 100).max(1);
            let mut tail: Vec<usize> = kick_samples[num_keys - tail_len..].to_vec();
            let sum: u64 = tail.iter().map(|&k| k as u64).sum();
            let mean: f64 = sum as f64 / tail.len() as f64;
            tail.sort_unstable();
            let p99_idx: usize = ((tail.len() * 99) / 100).min(tail.len() - 1);
            let p99: usize = tail[p99_idx];
            let max_tail: usize = *tail.last().unwrap_or(&0);
            (mean, p99, max_tail)
        };

        let stats = BuildStats {
            num_keys,
            load_factor: table.load_factor(),
            total_kicks,
            max_kicks_observed,
            tail_mean_kicks,
            tail_p99_kicks,
            tail_max_kicks,
        };
        Ok((table, stats))
    }

    /// Builds a stash-free table from a reusable slice of `(kind, key, value)` entries,
    /// rotating `table_seed` deterministically up to `max_seed_attempts` times if a seed
    /// fails to place all keys.
    #[allow(clippy::too_many_arguments)]
    pub fn build_with_seed_retry<'a>(
        entries: &[(u8, &'a [u8], &'a [u8])],
        key_size: Option<usize>,
        value_size: usize,
        tag_size: usize,
        bucket_size: usize,
        load: f64,
        initial_seed: [u8; 32],
        max_seed_attempts: usize,
    ) -> Result<(Self, BuildStats), KeywordError> {
        if max_seed_attempts == 0 {
            return Err(KeywordError::InvalidParams(
                "max_seed_attempts must be >= 1",
            ));
        }
        let mut seed: [u8; 32] = initial_seed;
        for attempt in 0..max_seed_attempts {
            if attempt > 0 {
                seed =
                    *blake3::keyed_hash(&initial_seed, &(attempt as u64).to_le_bytes()).as_bytes();
            }
            let spec: KeywordTableSpec = KeywordTableSpec::with_layout(
                entries.len(),
                key_size,
                value_size,
                tag_size,
                bucket_size,
                load,
                seed,
            )?;
            match Self::build(spec, entries.iter().copied()) {
                Ok(built) => return Ok(built),
                Err(build_err) => match build_err.error {
                    KeywordError::TableFull => continue,
                    other => return Err(other),
                },
            }
        }
        Err(KeywordError::SeedAttemptsExhausted {
            attempts: max_seed_attempts,
        })
    }

    /// Returns the table's public specification.
    #[inline]
    pub fn spec(&self) -> &KeywordTableSpec {
        &self.spec
    }

    /// Increments `spec.state_epoch` by 1 using checked arithmetic, returning
    /// `None` on `u64::MAX` overflow. Call once per published delta/stash batch.
    pub fn advance_state_epoch(&mut self) -> Option<u64> {
        let next: u64 = self.spec.state_epoch.checked_add(1)?;
        self.spec.state_epoch = next;
        Some(next)
    }

    /// Sets `spec.state_epoch` explicitly (for external batch publishers).
    #[inline]
    pub fn set_state_epoch(&mut self, epoch: u64) {
        self.spec.state_epoch = epoch;
    }

    /// Total number of stored keys (buckets + runtime-stash).
    #[inline]
    pub fn len(&self) -> usize {
        self.bucket_entries + self.stash.len()
    }

    /// Returns `true` if neither the buckets nor the runtime-stash hold any keys.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Number of keys stored in the two cuckoo segments (excluding runtime-stash).
    #[inline]
    pub fn bucket_len(&self) -> usize {
        self.bucket_entries
    }

    /// Returns the current runtime-stash entries.
    #[inline]
    pub fn stash(&self) -> &[StashEntry] {
        &self.stash
    }

    /// Current bucket load factor `bucket_len / total_slots`.
    #[inline]
    pub fn load_factor(&self) -> f64 {
        self.bucket_entries as f64 / self.spec.total_slots() as f64
    }

    /// Contiguous raw bytes of `database.bin` (`segment_0 | segment_1`).
    #[inline]
    pub fn as_bytes(&self) -> &[u8] {
        &self.data
    }

    /// Zero-copy slice of segment `0` or `1` (`buckets_per_segment * record_size` bytes).
    pub fn segment_bytes(&self, segment: usize) -> Option<&[u8]> {
        if segment >= NUM_SEGMENTS {
            return None;
        }
        let seg_len: usize = self.spec.segment_bytes();
        let start: usize = segment * seg_len;
        Some(&self.data[start..start + seg_len])
    }

    /// Zero-copy slice of one bucket (`record_size` bytes) in `segment` (`0` or `1`).
    pub fn bucket_bytes(&self, segment: usize, bucket_idx: usize) -> Option<&[u8]> {
        if segment >= NUM_SEGMENTS || bucket_idx >= self.spec.buckets_per_segment {
            return None;
        }
        let off: usize = self.bucket_byte_offset(segment, bucket_idx);
        Some(&self.data[off..off + self.spec.record_size()])
    }

    /// Plaintext lookup across both candidate buckets and the optional runtime-stash.
    pub fn get(&self, kind: u8, key: &[u8]) -> Result<Option<&[u8]>, KeywordError> {
        let mut tag_buf: [u8; MAX_TAG_SIZE] = [0u8; MAX_TAG_SIZE];
        let (a0, a1) = self.spec.locate_with_bases(
            kind,
            key,
            &self.key_hasher_base,
            &self.alt_hasher_base,
            &mut tag_buf,
        )?;
        let tag: &[u8] = &tag_buf[..self.spec.tag_size];
        let bucket0: &[u8] = self.bucket_slice(0, a0);
        let bucket1: &[u8] = self.bucket_slice(1, a1);
        let stash_match: Option<&[u8]> = self
            .stash
            .iter()
            .find(|e| e.bucket_seg0 == a0 && e.tag.as_slice() == tag)
            .map(|e| e.value.as_slice());
        self.spec.scan_buckets(bucket0, bucket1, tag, stash_match)
    }

    /// Inserts a new `(kind, key, value)` entry and returns net XOR bucket deltas
    /// (or a runtime-stash insert if enabled and the eviction walk hits `max_kicks`).
    pub fn insert(
        &mut self,
        kind: u8,
        key: &[u8],
        value: &[u8],
    ) -> Result<MutationOutcome, KeywordError> {
        self.check_value_len(value)?;
        let mut tag_buf: [u8; MAX_TAG_SIZE] = [0u8; MAX_TAG_SIZE];
        let (a0, a1) = self.spec.locate_with_bases(
            kind,
            key,
            &self.key_hasher_base,
            &self.alt_hasher_base,
            &mut tag_buf,
        )?;
        let tag: &[u8] = &tag_buf[..self.spec.tag_size];

        // Duplicate check across both candidate buckets and runtime-stash.
        if self.find_slot_in_bucket(0, a0, tag).is_some()
            || self.find_slot_in_bucket(1, a1, tag).is_some()
            || self.find_stash_index(a0, tag).is_some()
        {
            return Err(KeywordError::DuplicateKey);
        }

        // Fast path: free slot in seg 0 or seg 1 (pick less-loaded bucket first).
        if let Some((seg, bidx, sidx)) = self.find_free_candidate_slot(a0, a1) {
            let rec: usize = self.spec.record_size();
            let mut xor_delta: Vec<u8> = vec![0u8; rec];
            let slot_start: usize = sidx * self.spec.slot_size();
            xor_delta[slot_start..slot_start + self.spec.tag_size].copy_from_slice(tag);
            xor_delta[slot_start + self.spec.tag_size..slot_start + self.spec.slot_size()]
                .copy_from_slice(value);

            self.write_slot(seg, bidx, sidx, tag, value);
            self.bucket_entries += 1;
            return Ok(MutationOutcome {
                deltas: vec![BucketDelta {
                    segment: seg,
                    bucket_idx: bidx,
                    xor_delta,
                }],
                stash_op: None,
                kicks: 0,
            });
        }

        // Eviction random walk with full rollback snapshot.
        let mut originals: HashMap<(usize, usize), Vec<u8>> = HashMap::new();
        match self.eviction_walk(a0, a1, tag, value, Some(&mut originals)) {
            Ok(kicks) => {
                self.bucket_entries += 1;
                let deltas: Vec<BucketDelta> = self.finalize_deltas(originals);
                Ok(MutationOutcome {
                    deltas,
                    stash_op: None,
                    kicks,
                })
            }
            Err(KeywordError::TableFull) => {
                self.rollback_originals(originals);
                if self.stash.len() < self.max_stash {
                    let entry = StashEntry {
                        bucket_seg0: a0,
                        tag: tag.to_vec(),
                        value: value.to_vec(),
                    };
                    self.stash.push(entry.clone());
                    Ok(MutationOutcome {
                        deltas: Vec::new(),
                        stash_op: Some(StashOp::Insert(entry)),
                        kicks: self.max_kicks,
                    })
                } else {
                    Err(KeywordError::TableFull)
                }
            }
            Err(other) => {
                self.rollback_originals(originals);
                Err(other)
            }
        }
    }

    /// Overwrites the value of an existing key in-place (1 bucket delta, or a stash update).
    pub fn update(
        &mut self,
        kind: u8,
        key: &[u8],
        new_value: &[u8],
    ) -> Result<MutationOutcome, KeywordError> {
        self.check_value_len(new_value)?;
        let mut tag_buf: [u8; MAX_TAG_SIZE] = [0u8; MAX_TAG_SIZE];
        let (a0, a1) = self.spec.locate_with_bases(
            kind,
            key,
            &self.key_hasher_base,
            &self.alt_hasher_base,
            &mut tag_buf,
        )?;
        let tag: &[u8] = &tag_buf[..self.spec.tag_size];

        match self.find_unique_target(a0, a1, tag)? {
            Some(SlotTarget::Bucket(seg, bidx, sidx)) => {
                Ok(self.update_bucket_slot(seg, bidx, sidx, new_value))
            }
            Some(SlotTarget::Stash(stash_idx)) => {
                self.stash[stash_idx].value.copy_from_slice(new_value);
                Ok(MutationOutcome {
                    deltas: Vec::new(),
                    stash_op: Some(StashOp::Update {
                        bucket_seg0: a0,
                        tag: tag.to_vec(),
                        value: new_value.to_vec(),
                    }),
                    kicks: 0,
                })
            }
            None => Err(KeywordError::KeyNotFound),
        }
    }

    /// Updates an existing key or inserts it if absent.
    pub fn upsert(
        &mut self,
        kind: u8,
        key: &[u8],
        value: &[u8],
    ) -> Result<MutationOutcome, KeywordError> {
        match self.update(kind, key, value) {
            Ok(outcome) => Ok(outcome),
            Err(KeywordError::KeyNotFound) => self.insert(kind, key, value),
            Err(other) => Err(other),
        }
    }

    /// Deletes an existing key by zeroing its slot (1 bucket delta) or removing it from
    /// the runtime-stash.
    pub fn delete(&mut self, kind: u8, key: &[u8]) -> Result<MutationOutcome, KeywordError> {
        let mut tag_buf: [u8; MAX_TAG_SIZE] = [0u8; MAX_TAG_SIZE];
        let (a0, a1) = self.spec.locate_with_bases(
            kind,
            key,
            &self.key_hasher_base,
            &self.alt_hasher_base,
            &mut tag_buf,
        )?;
        let tag: &[u8] = &tag_buf[..self.spec.tag_size];

        match self.find_unique_target(a0, a1, tag)? {
            Some(SlotTarget::Bucket(seg, bidx, sidx)) => {
                Ok(self.clear_bucket_slot(seg, bidx, sidx))
            }
            Some(SlotTarget::Stash(stash_idx)) => {
                self.stash.swap_remove(stash_idx);
                Ok(MutationOutcome {
                    deltas: Vec::new(),
                    stash_op: Some(StashOp::Delete {
                        bucket_seg0: a0,
                        tag: tag.to_vec(),
                    }),
                    kicks: 0,
                })
            }
            None => Err(KeywordError::KeyNotFound),
        }
    }

    /// Writes the contiguous `database.bin` (`segment_0 | segment_1`) to `writer`.
    pub fn write_database<W: Write>(&self, mut writer: W) -> io::Result<()> {
        writer.write_all(&self.data)
    }

    /// Writes `database.bin` to `path`.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn write_database_file<P: AsRef<Path>>(&self, path: P) -> io::Result<()> {
        let file = std::fs::File::create(path)?;
        let mut writer = io::BufWriter::new(file);
        self.write_database(&mut writer)?;
        writer.flush()
    }

    /// Writes segment 0 and segment 1 to separate writers.
    pub fn write_segments<W0: Write, W1: Write>(
        &self,
        mut seg0_writer: W0,
        mut seg1_writer: W1,
    ) -> io::Result<()> {
        let seg_len: usize = self.spec.segment_bytes();
        seg0_writer.write_all(&self.data[..seg_len])?;
        seg1_writer.write_all(&self.data[seg_len..])
    }

    /// Fast bulk insert used by [`Self::build`] (avoids delta allocations when no eviction
    /// occurs; still rolls back atomically on `TableFull`).
    fn insert_bulk(&mut self, kind: u8, key: &[u8], value: &[u8]) -> Result<usize, KeywordError> {
        self.check_value_len(value)?;
        let mut tag_buf: [u8; MAX_TAG_SIZE] = [0u8; MAX_TAG_SIZE];
        let (a0, a1) = self.spec.locate_with_bases(
            kind,
            key,
            &self.key_hasher_base,
            &self.alt_hasher_base,
            &mut tag_buf,
        )?;
        let tag: &[u8] = &tag_buf[..self.spec.tag_size];

        if self.find_slot_in_bucket(0, a0, tag).is_some()
            || self.find_slot_in_bucket(1, a1, tag).is_some()
        {
            return Err(KeywordError::DuplicateKey);
        }

        if let Some((seg, bidx, sidx)) = self.find_free_candidate_slot(a0, a1) {
            self.write_slot(seg, bidx, sidx, tag, value);
            self.bucket_entries += 1;
            return Ok(0);
        }

        let mut originals: HashMap<(usize, usize), Vec<u8>> = HashMap::new();
        match self.eviction_walk(a0, a1, tag, value, Some(&mut originals)) {
            Ok(kicks) => {
                self.bucket_entries += 1;
                Ok(kicks)
            }
            Err(err) => {
                self.rollback_originals(originals);
                Err(err)
            }
        }
    }

    fn eviction_walk(
        &mut self,
        a0: usize,
        a1: usize,
        tag: &[u8],
        value: &[u8],
        mut originals: Option<&mut HashMap<(usize, usize), Vec<u8>>>,
    ) -> Result<usize, KeywordError> {
        let mut seg: usize = self.eviction_rng.gen_range(0..NUM_SEGMENTS);
        let mut bidx: usize = if seg == 0 { a0 } else { a1 };

        let slot_size: usize = self.spec.slot_size();
        let tag_size: usize = self.spec.tag_size;
        let mut cur_slot: Vec<u8> = Vec::with_capacity(slot_size);
        cur_slot.extend_from_slice(tag);
        cur_slot.extend_from_slice(value);
        let mut displaced: Vec<u8> = vec![0u8; slot_size];

        for kick in 1..=self.max_kicks {
            if let Some(ref mut orig) = originals {
                orig.entry((seg, bidx))
                    .or_insert_with(|| self.bucket_slice(seg, bidx).to_vec());
            }

            let victim_slot: usize = self.eviction_rng.gen_range(0..self.spec.bucket_size);
            let victim_off: usize = self.slot_byte_offset(seg, bidx, victim_slot);
            displaced.copy_from_slice(&self.data[victim_off..victim_off + slot_size]);
            self.data[victim_off..victim_off + slot_size].copy_from_slice(&cur_slot);
            std::mem::swap(&mut cur_slot, &mut displaced);

            // Compute the displaced slot's alternate bucket in the opposite segment
            // purely from its stored tag (partial-key cuckoo hashing).
            let next_bidx: usize = self.spec.alt_bucket_with_base(
                seg,
                bidx,
                &cur_slot[..tag_size],
                &self.alt_hasher_base,
            );
            let next_seg: usize = 1 - seg;

            if let Some(free_slot) = self.find_free_slot_in_bucket(next_seg, next_bidx) {
                if let Some(ref mut orig) = originals {
                    orig.entry((next_seg, next_bidx))
                        .or_insert_with(|| self.bucket_slice(next_seg, next_bidx).to_vec());
                }
                let free_off: usize = self.slot_byte_offset(next_seg, next_bidx, free_slot);
                self.data[free_off..free_off + slot_size].copy_from_slice(&cur_slot);
                return Ok(kick);
            }

            seg = next_seg;
            bidx = next_bidx;
        }

        Err(KeywordError::TableFull)
    }

    fn update_bucket_slot(
        &mut self,
        segment: usize,
        bucket_idx: usize,
        slot_idx: usize,
        new_value: &[u8],
    ) -> MutationOutcome {
        let rec: usize = self.spec.record_size();
        let val_in_bucket: usize = slot_idx * self.spec.slot_size() + self.spec.tag_size;
        let abs_val_off: usize = self.bucket_byte_offset(segment, bucket_idx) + val_in_bucket;

        let mut xor_delta: Vec<u8> = vec![0u8; rec];
        let mut changed: bool = false;
        for i in 0..self.spec.value_size {
            let diff: u8 = self.data[abs_val_off + i] ^ new_value[i];
            if diff != 0 {
                changed = true;
                xor_delta[val_in_bucket + i] = diff;
                self.data[abs_val_off + i] = new_value[i];
            }
        }

        let deltas: Vec<BucketDelta> = if changed {
            vec![BucketDelta {
                segment,
                bucket_idx,
                xor_delta,
            }]
        } else {
            Vec::new()
        };
        MutationOutcome {
            deltas,
            stash_op: None,
            kicks: 0,
        }
    }

    fn clear_bucket_slot(
        &mut self,
        segment: usize,
        bucket_idx: usize,
        slot_idx: usize,
    ) -> MutationOutcome {
        let rec: usize = self.spec.record_size();
        let slot_size: usize = self.spec.slot_size();
        let slot_in_bucket: usize = slot_idx * slot_size;
        let abs_off: usize = self.bucket_byte_offset(segment, bucket_idx) + slot_in_bucket;

        let mut xor_delta: Vec<u8> = vec![0u8; rec];
        xor_delta[slot_in_bucket..slot_in_bucket + slot_size]
            .copy_from_slice(&self.data[abs_off..abs_off + slot_size]);
        self.data[abs_off..abs_off + slot_size].fill(0);
        self.bucket_entries -= 1;

        MutationOutcome {
            deltas: vec![BucketDelta {
                segment,
                bucket_idx,
                xor_delta,
            }],
            stash_op: None,
            kicks: 0,
        }
    }

    fn finalize_deltas(&self, originals: HashMap<(usize, usize), Vec<u8>>) -> Vec<BucketDelta> {
        let mut entries: Vec<((usize, usize), Vec<u8>)> = originals.into_iter().collect();
        entries.sort_unstable_by_key(|&(k, _)| k);

        let rec: usize = self.spec.record_size();
        let mut deltas: Vec<BucketDelta> = Vec::with_capacity(entries.len());
        for ((segment, bucket_idx), mut orig) in entries {
            let cur: &[u8] = self.bucket_slice(segment, bucket_idx);
            let mut non_zero: bool = false;
            for i in 0..rec {
                orig[i] ^= cur[i];
                if orig[i] != 0 {
                    non_zero = true;
                }
            }
            if non_zero {
                deltas.push(BucketDelta {
                    segment,
                    bucket_idx,
                    xor_delta: orig,
                });
            }
        }
        deltas
    }

    fn rollback_originals(&mut self, originals: HashMap<(usize, usize), Vec<u8>>) {
        let rec: usize = self.spec.record_size();
        for ((segment, bucket_idx), orig) in originals {
            let off: usize = self.bucket_byte_offset(segment, bucket_idx);
            self.data[off..off + rec].copy_from_slice(&orig);
        }
    }

    #[inline]
    fn check_value_len(&self, value: &[u8]) -> Result<(), KeywordError> {
        if value.len() != self.spec.value_size {
            return Err(KeywordError::ValueSizeMismatch {
                expected: self.spec.value_size,
                actual: value.len(),
            });
        }
        Ok(())
    }

    #[inline]
    fn bucket_byte_offset(&self, segment: usize, bucket_idx: usize) -> usize {
        (segment * self.spec.buckets_per_segment + bucket_idx) * self.spec.record_size()
    }

    #[inline]
    fn slot_byte_offset(&self, segment: usize, bucket_idx: usize, slot_idx: usize) -> usize {
        self.bucket_byte_offset(segment, bucket_idx) + slot_idx * self.spec.slot_size()
    }

    #[inline]
    fn bucket_slice(&self, segment: usize, bucket_idx: usize) -> &[u8] {
        let off: usize = self.bucket_byte_offset(segment, bucket_idx);
        &self.data[off..off + self.spec.record_size()]
    }

    #[inline]
    fn find_slot_in_bucket(&self, segment: usize, bucket_idx: usize, tag: &[u8]) -> Option<usize> {
        let bucket: &[u8] = self.bucket_slice(segment, bucket_idx);
        let slot_size: usize = self.spec.slot_size();
        let tag_size: usize = self.spec.tag_size;
        for sidx in 0..self.spec.bucket_size {
            let start: usize = sidx * slot_size;
            if &bucket[start..start + tag_size] == tag {
                return Some(sidx);
            }
        }
        None
    }

    #[inline]
    fn find_free_slot_in_bucket(&self, segment: usize, bucket_idx: usize) -> Option<usize> {
        let bucket: &[u8] = self.bucket_slice(segment, bucket_idx);
        let slot_size: usize = self.spec.slot_size();
        let tag_size: usize = self.spec.tag_size;
        for sidx in 0..self.spec.bucket_size {
            let start: usize = sidx * slot_size;
            if is_zero_slice(&bucket[start..start + tag_size]) {
                return Some(sidx);
            }
        }
        None
    }

    /// Picks a free slot between `(seg 0, a0)` and `(seg 1, a1)`, preferring the
    /// bucket with fewer occupied slots (2-choice load balancing).
    #[inline]
    fn find_free_candidate_slot(&self, a0: usize, a1: usize) -> Option<(usize, usize, usize)> {
        let (free0, occ0) = self.bucket_free_and_occupancy(0, a0);
        let (free1, occ1) = self.bucket_free_and_occupancy(1, a1);
        match (free0, free1) {
            (Some(s0), Some(s1)) => {
                if occ0 <= occ1 {
                    Some((0, a0, s0))
                } else {
                    Some((1, a1, s1))
                }
            }
            (Some(s0), None) => Some((0, a0, s0)),
            (None, Some(s1)) => Some((1, a1, s1)),
            (None, None) => None,
        }
    }

    #[inline]
    fn bucket_free_and_occupancy(
        &self,
        segment: usize,
        bucket_idx: usize,
    ) -> (Option<usize>, usize) {
        let bucket: &[u8] = self.bucket_slice(segment, bucket_idx);
        let slot_size: usize = self.spec.slot_size();
        let tag_size: usize = self.spec.tag_size;
        let mut first_free: Option<usize> = None;
        let mut occupied: usize = 0;
        for sidx in 0..self.spec.bucket_size {
            let start: usize = sidx * slot_size;
            if is_zero_slice(&bucket[start..start + tag_size]) {
                if first_free.is_none() {
                    first_free = Some(sidx);
                }
            } else {
                occupied += 1;
            }
        }
        (first_free, occupied)
    }

    #[inline]
    fn write_slot(
        &mut self,
        segment: usize,
        bucket_idx: usize,
        slot_idx: usize,
        tag: &[u8],
        value: &[u8],
    ) {
        let off: usize = self.slot_byte_offset(segment, bucket_idx, slot_idx);
        self.data[off..off + self.spec.tag_size].copy_from_slice(tag);
        self.data[off + self.spec.tag_size..off + self.spec.slot_size()].copy_from_slice(value);
    }

    #[inline]
    fn find_stash_index(&self, bucket_seg0: usize, tag: &[u8]) -> Option<usize> {
        self.stash
            .iter()
            .position(|e| e.bucket_seg0 == bucket_seg0 && e.tag.as_slice() == tag)
    }

    fn find_unique_target(
        &self,
        a0: usize,
        a1: usize,
        tag: &[u8],
    ) -> Result<Option<SlotTarget>, KeywordError> {
        let mut found: Option<SlotTarget> = None;
        let slot_size: usize = self.spec.slot_size();
        let tag_size: usize = self.spec.tag_size;

        for (seg, bidx) in [(0usize, a0), (1usize, a1)] {
            let bucket: &[u8] = self.bucket_slice(seg, bidx);
            for sidx in 0..self.spec.bucket_size {
                let start: usize = sidx * slot_size;
                if &bucket[start..start + tag_size] == tag {
                    if found.is_some() {
                        return Err(KeywordError::DuplicateTagMatch);
                    }
                    found = Some(SlotTarget::Bucket(seg, bidx, sidx));
                }
            }
        }

        for (idx, entry) in self.stash.iter().enumerate() {
            if entry.bucket_seg0 == a0 && entry.tag.as_slice() == tag {
                if found.is_some() {
                    return Err(KeywordError::DuplicateTagMatch);
                }
                found = Some(SlotTarget::Stash(idx));
            }
        }

        Ok(found)
    }
}

enum SlotTarget {
    Bucket(usize, usize, usize),
    Stash(usize),
}

#[inline]
fn is_zero_slice(bytes: &[u8]) -> bool {
    bytes.iter().all(|&b| b == 0)
}

#[inline]
fn make_key_hasher_base(seed: &[u8; 32]) -> blake3::Hasher {
    let mut h: blake3::Hasher = blake3::Hasher::new_keyed(seed);
    h.update(HASH_DOMAIN_KEY);
    h
}

#[inline]
fn make_alt_hasher_base(seed: &[u8; 32]) -> blake3::Hasher {
    let mut h: blake3::Hasher = blake3::Hasher::new_keyed(seed);
    h.update(HASH_DOMAIN_ALT);
    h
}

fn derive_eviction_rng(seed: &[u8; 32]) -> ChaCha20Rng {
    let rng_seed: [u8; 32] = *blake3::keyed_hash(seed, HASH_DOMAIN_RNG).as_bytes();
    ChaCha20Rng::from_seed(rng_seed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::multilevel::{BackendVariant, MultiLevelClient, MultiLevelParams, MultiLevelServer};

    fn make_key(i: u64) -> [u8; 20] {
        let h = blake3::hash(&i.to_le_bytes());
        let mut out = [0u8; 20];
        out.copy_from_slice(&h.as_bytes()[..20]);
        out
    }

    fn make_val(i: u64, value_size: usize) -> Vec<u8> {
        let mut out = vec![0u8; value_size];
        let mut h = blake3::Hasher::new();
        h.update(b"val");
        h.update(&i.to_le_bytes());
        h.finalize_xof().fill(&mut out);
        out
    }

    #[test]
    fn empty_table_spec_has_nonzero_segments_and_locates_safely() {
        let spec = KeywordTableSpec::for_capacity(0, 32, 0.90, [0x11; 32]).unwrap();
        assert_eq!(spec.buckets_per_segment, 1);
        assert_eq!(spec.total_buckets(), 2);
        assert_eq!(spec.record_size(), 4 * (8 + 32));
        let loc = spec.locate(0x01, b"any-key").unwrap();
        assert_eq!(loc.bucket_seg0, 0);
        assert_eq!(loc.bucket_seg1, 0);
        assert_eq!(loc.global_indices(&spec), [0, 1]);
        assert!(!is_zero_slice(&loc.tag));
    }

    #[test]
    fn partial_key_alt_bucket_is_exact_involution_for_non_power_of_two_s() {
        for &s in &[1usize, 3, 7, 17, 100, 13_337, 1_000_003] {
            let spec = KeywordTableSpec::new(s, 4, 8, 24, None, [0x42; 32], 0, 0).unwrap();
            for i in 0u64..200 {
                let k = make_key(i);
                let loc = spec.locate((i % 3) as u8, &k).unwrap();
                assert!(loc.bucket_seg0 < s);
                assert!(loc.bucket_seg1 < s);
                assert_eq!(
                    spec.alt_bucket(0, loc.bucket_seg0, &loc.tag),
                    loc.bucket_seg1
                );
                assert_eq!(
                    spec.alt_bucket(1, loc.bucket_seg1, &loc.tag),
                    loc.bucket_seg0
                );
                let [g0, g1] = loc.global_indices(&spec);
                assert_ne!(g0, g1);
                assert!(g0 < s && g1 >= s && g1 < 2 * s);
            }
        }
    }

    #[test]
    fn build_and_lookup_at_90_percent_load_arbitrary_sizes() {
        for &(value_size, tag_size, bucket_size) in
            &[(1usize, 8, 4), (24, 8, 4), (32, 16, 4), (13, 8, 2)]
        {
            let n: usize = 4_000;
            let load: f64 = if bucket_size == 2 { 0.80 } else { 0.90 };
            let keys: Vec<[u8; 20]> = (0..n as u64).map(make_key).collect();
            let vals: Vec<Vec<u8>> = (0..n as u64).map(|i| make_val(i, value_size)).collect();
            let entries: Vec<(u8, &[u8], &[u8])> = (0..n)
                .map(|i| ((i % 2) as u8, keys[i].as_slice(), vals[i].as_slice()))
                .collect();

            let (table, stats) = KvScfTable::build_with_seed_retry(
                &entries,
                Some(20),
                value_size,
                tag_size,
                bucket_size,
                load,
                [0xAB; 32],
                8,
            )
            .unwrap();

            assert_eq!(table.len(), n);
            assert_eq!(stats.num_keys, n);
            assert!(table.stash().is_empty());

            for i in 0..n {
                let got = table.get((i % 2) as u8, &keys[i]).unwrap();
                assert_eq!(got, Some(vals[i].as_slice()));
                // Wrong kind namespace must not match.
                let wrong_kind = table.get(((i + 1) % 2) as u8, &keys[i]).unwrap();
                assert_eq!(wrong_kind, None);
            }
            for i in n..(n + 500) {
                let missing = make_key(i as u64);
                assert_eq!(table.get(0, &missing).unwrap(), None);
            }

            // Round-trip via database.bin bytes.
            let mut buf: Vec<u8> = Vec::new();
            table.write_database(&mut buf).unwrap();
            assert_eq!(buf.len(), table.spec().database_bytes());
            let reloaded = KvScfTable::from_database_bytes(table.spec().clone(), buf).unwrap();
            assert_eq!(reloaded.bucket_len(), n);
            assert_eq!(reloaded.get(0, &keys[0]).unwrap(), Some(vals[0].as_slice()));
        }
    }

    #[test]
    fn build_reports_unplaceable_key_when_table_is_oversubscribed() {
        // Total slots = 2 * 2 * 2 = 8; inserting 9 distinct keys must fail and report the key.
        let spec = KeywordTableSpec::new(2, 2, 8, 8, None, [0x55; 32], 0, 0).unwrap();
        let keys: Vec<Vec<u8>> = (0u64..12)
            .map(|i| format!("key-{i}").into_bytes())
            .collect();
        let val = [0x99u8; 8];
        let entries: Vec<(u8, &[u8], &[u8])> = keys
            .iter()
            .map(|k| (0u8, k.as_slice(), val.as_slice()))
            .collect();

        let err = KvScfTable::build(spec, entries).unwrap_err();
        assert_eq!(err.error, KeywordError::TableFull);
        assert!(err.entry_index >= 8);
        assert_eq!(err.key, keys[err.entry_index]);
    }

    #[test]
    fn live_mutations_emit_exact_xor_deltas_and_support_runtime_stash() {
        let spec = KeywordTableSpec::new(16, 4, 8, 16, None, [0x77; 32], 0, 0).unwrap();
        let mut table = KvScfTable::new(spec.clone()).with_max_stash(16);
        let mut mirror: Vec<u8> = table.as_bytes().to_vec();

        let apply_deltas = |mirror: &mut [u8], spec: &KeywordTableSpec, deltas: &[BucketDelta]| {
            let rec: usize = spec.record_size();
            for d in deltas {
                assert_eq!(d.xor_delta.len(), rec);
                let gidx: usize = d.global_index(spec);
                let off: usize = gidx * rec;
                for i in 0..rec {
                    mirror[off + i] ^= d.xor_delta[i];
                }
            }
        };

        // Insert until table fills and spills into runtime-stash (128 bucket slots + 10 stash).
        let total: usize = 138;
        let mut stashed_keys: Vec<Vec<u8>> = Vec::new();
        for i in 0..total as u64 {
            let k = format!("k-{i}").into_bytes();
            let v = make_val(i, 16);
            let out = table.insert(0, &k, &v).unwrap();
            apply_deltas(&mut mirror, &spec, &out.deltas);
            assert_eq!(mirror.as_slice(), table.as_bytes());
            if out.stash_op.is_some() {
                stashed_keys.push(k);
            }
        }
        assert!(
            !stashed_keys.is_empty(),
            "expected overflow into runtime-stash"
        );
        assert_eq!(table.len(), total);

        // Duplicate insert must be rejected for both bucketed and stashed keys.
        assert_eq!(
            table.insert(0, b"k-0", &[0xAA; 16]).unwrap_err(),
            KeywordError::DuplicateKey
        );
        assert_eq!(
            table.insert(0, &stashed_keys[0], &[0xAA; 16]).unwrap_err(),
            KeywordError::DuplicateKey
        );

        // Update both a bucketed key and a stashed key.
        let new_v0 = [0xCC; 16];
        let u0 = table.update(0, b"k-0", &new_v0).unwrap();
        assert!(u0.stash_op.is_none());
        assert_eq!(u0.deltas.len(), 1);
        apply_deltas(&mut mirror, &spec, &u0.deltas);
        assert_eq!(mirror.as_slice(), table.as_bytes());
        assert_eq!(table.get(0, b"k-0").unwrap(), Some(new_v0.as_slice()));

        let new_vs = [0xDD; 16];
        let us = table.update(0, &stashed_keys[0], &new_vs).unwrap();
        assert!(us.deltas.is_empty());
        assert!(matches!(us.stash_op, Some(StashOp::Update { .. })));
        assert_eq!(
            table.get(0, &stashed_keys[0]).unwrap(),
            Some(new_vs.as_slice())
        );

        // Delete both a bucketed key and a stashed key.
        let d0 = table.delete(0, b"k-0").unwrap();
        assert_eq!(d0.deltas.len(), 1);
        apply_deltas(&mut mirror, &spec, &d0.deltas);
        assert_eq!(mirror.as_slice(), table.as_bytes());
        assert_eq!(table.get(0, b"k-0").unwrap(), None);

        let ds = table.delete(0, &stashed_keys[0]).unwrap();
        assert!(ds.deltas.is_empty());
        assert!(matches!(ds.stash_op, Some(StashOp::Delete { .. })));
        assert_eq!(table.get(0, &stashed_keys[0]).unwrap(), None);
        assert_eq!(table.len(), total - 2);
    }

    #[test]
    fn failed_insert_rolls_back_table_bytes_cleanly() {
        let spec = KeywordTableSpec::new(2, 2, 8, 8, None, [0x33; 32], 0, 0).unwrap();
        let mut table = KvScfTable::new(spec); // max_stash = 0
        for i in 0u64..8 {
            let k = format!("fill-{i}").into_bytes();
            let v = make_val(i, 8);
            table.insert(0, &k, &v).unwrap();
        }
        let snapshot = table.as_bytes().to_vec();
        let err = table.insert(0, b"overflow-key", &[0xEE; 8]).unwrap_err();
        assert_eq!(err, KeywordError::TableFull);
        assert_eq!(table.as_bytes(), snapshot.as_slice());
        assert_eq!(table.len(), 8);
    }

    #[test]
    fn scan_buckets_combines_buckets_and_stash_before_deciding() {
        let spec = KeywordTableSpec::new(4, 2, 8, 4, None, [0x12; 32], 0, 0).unwrap();
        let empty = vec![0u8; spec.record_size()];
        let tag = [1u8; 8];
        let val = [9u8; 4];

        // Stash-only hit must succeed (not return None from empty buckets).
        assert_eq!(
            spec.scan_buckets(&empty, &empty, &tag, Some(&val)).unwrap(),
            Some(val.as_slice())
        );

        // Match in both bucket and stash must return DuplicateTagMatch.
        let mut b0 = empty.clone();
        b0[..8].copy_from_slice(&tag);
        b0[8..12].copy_from_slice(&val);
        assert_eq!(
            spec.scan_buckets(&b0, &empty, &tag, Some(&val))
                .unwrap_err(),
            KeywordError::DuplicateTagMatch
        );
    }

    #[test]
    fn kv_scf_end_to_end_with_two_segment_multilevel_pir_and_deltas() {
        let n: usize = 4_096;
        let value_size: usize = 32;
        let keys: Vec<[u8; 20]> = (0..n as u64).map(make_key).collect();
        let vals: Vec<Vec<u8>> = (0..n as u64).map(|i| make_val(i, value_size)).collect();
        let entries: Vec<(u8, &[u8], &[u8])> = (0..n)
            .map(|i| (0x01, keys[i].as_slice(), vals[i].as_slice()))
            .collect();

        let (mut table, _stats) = KvScfTable::build_with_seed_retry(
            &entries,
            Some(20),
            value_size,
            8,
            4,
            0.80,
            [0x5A; 32],
            4,
        )
        .unwrap();

        let spec = table.spec().clone();
        let s: usize = spec.buckets_per_segment;
        let rec: usize = spec.record_size();
        let ml_params = MultiLevelParams::two_level(s, rec, 8, 64, BackendVariant::FlatB).unwrap();

        let mut c0 = MultiLevelClient::hint_init(
            ml_params.clone(),
            &[0x10; 32],
            table.segment_bytes(0).unwrap(),
            1,
        )
        .unwrap();
        let mut c1 = MultiLevelClient::hint_init(
            ml_params.clone(),
            &[0x20; 32],
            table.segment_bytes(1).unwrap(),
            1,
        )
        .unwrap();

        let mut rng = ChaCha20Rng::seed_from_u64(99);
        let lookup_pir = |key: &[u8],
                          c0: &mut MultiLevelClient,
                          c1: &mut MultiLevelClient,
                          table: &KvScfTable,
                          rng: &mut ChaCha20Rng|
         -> Option<Vec<u8>> {
            let loc = spec.locate(0x01, key).unwrap();
            let p0 = c0.prepare_query(loc.bucket_seg0, rng).unwrap();
            let p1 = c1.prepare_query(loc.bucket_seg1, rng).unwrap();
            let a0 = MultiLevelServer::answer(
                table.segment_bytes(0).unwrap(),
                &ml_params,
                &p0.query,
                c0.epoch(),
            )
            .unwrap();
            let a1 = MultiLevelServer::answer(
                table.segment_bytes(1).unwrap(),
                &ml_params,
                &p1.query,
                c1.epoch(),
            )
            .unwrap();
            let b0 = c0.finish_query(&p0, &a0, rng).unwrap();
            let b1 = c1.finish_query(&p1, &a1, rng).unwrap();
            spec.scan_buckets(&b0, &b1, &loc.tag, None)
                .unwrap()
                .map(|v| v.to_vec())
        };

        // Check existing and absent keys before mutations.
        for i in [0usize, 17, 1024, n - 1] {
            assert_eq!(
                lookup_pir(&keys[i], &mut c0, &mut c1, &table, &mut rng),
                Some(vals[i].clone())
            );
        }
        let absent = make_key(999_999);
        assert_eq!(
            lookup_pir(&absent, &mut c0, &mut c1, &table, &mut rng),
            None
        );

        // Apply live update + live insert + live delete and propagate deltas to both segments.
        let new_val = make_val(123_456, value_size);
        let ins_key = make_key(777_777);
        let ins_val = make_val(777_777, value_size);

        for outcome in [
            table.update(0x01, &keys[17], &new_val).unwrap(),
            table.insert(0x01, &ins_key, &ins_val).unwrap(),
            table.delete(0x01, &keys[1024]).unwrap(),
        ] {
            let d0: Vec<(usize, Vec<u8>)> = outcome
                .deltas
                .iter()
                .filter(|d| d.segment == 0)
                .map(|d| (d.bucket_idx, d.xor_delta.clone()))
                .collect();
            let d1: Vec<(usize, Vec<u8>)> = outcome
                .deltas
                .iter()
                .filter(|d| d.segment == 1)
                .map(|d| (d.bucket_idx, d.xor_delta.clone()))
                .collect();
            if !d0.is_empty() {
                c0.apply_deltas_batch(&d0, 1).unwrap();
            }
            if !d1.is_empty() {
                c1.apply_deltas_batch(&d1, 1).unwrap();
            }
        }

        assert_eq!(
            lookup_pir(&keys[17], &mut c0, &mut c1, &table, &mut rng),
            Some(new_val)
        );
        assert_eq!(
            lookup_pir(&ins_key, &mut c0, &mut c1, &table, &mut rng),
            Some(ins_val)
        );
        assert_eq!(
            lookup_pir(&keys[1024], &mut c0, &mut c1, &table, &mut rng),
            None
        );
    }

    #[test]
    fn from_database_bytes_rejects_non_canonical_empty_slot_and_duplicates() {
        let spec = KeywordTableSpec::new(4, 2, 8, 8, None, [0x44; 32], 0, 0).unwrap();

        // 1) Non-canonical empty slot: zero tag with non-zero value bytes.
        let mut corrupt = vec![0u8; spec.database_bytes()];
        corrupt[8] = 0xFF;
        assert_eq!(
            KvScfTable::from_database_bytes(spec.clone(), corrupt).unwrap_err(),
            KeywordError::CorruptDatabase("empty slot (all-zero tag) has non-zero value bytes")
        );

        // 2) Duplicate identity placed in both candidate buckets of the same key.
        let loc = spec.locate(0, b"dup-key").unwrap();
        let mut dup = vec![0u8; spec.database_bytes()];
        let rec = spec.record_size();
        let off0 = loc.bucket_seg0 * rec;
        let off1 = (spec.buckets_per_segment() + loc.bucket_seg1) * rec;
        dup[off0..off0 + 8].copy_from_slice(&loc.tag);
        dup[off0 + 8..off0 + 16].copy_from_slice(&[1u8; 8]);
        dup[off1..off1 + 8].copy_from_slice(&loc.tag);
        dup[off1 + 8..off1 + 16].copy_from_slice(&[2u8; 8]);
        assert_eq!(
            KvScfTable::from_database_bytes(spec, dup).unwrap_err(),
            KeywordError::DuplicateTagMatch
        );
    }

    #[cfg(feature = "cli")]
    #[test]
    fn spec_deserialization_rejects_invalid_geometry() {
        let valid = KeywordTableSpec::for_capacity(100, 32, 0.90, [0x22; 32]).unwrap();
        let json = serde_json::to_string(&valid).unwrap();
        let roundtrip: KeywordTableSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(roundtrip, valid);

        let bad_s = json.replace(
            &format!("\"buckets_per_segment\":{}", valid.buckets_per_segment()),
            "\"buckets_per_segment\":0",
        );
        assert!(serde_json::from_str::<KeywordTableSpec>(&bad_s).is_err());

        let bad_tag = json.replace("\"tag_size\":8", "\"tag_size\":64");
        assert!(serde_json::from_str::<KeywordTableSpec>(&bad_tag).is_err());
    }
}
