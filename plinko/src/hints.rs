//! Hint set of a Plinko client and its on-disk format.
//!
//! The hint generator (`plinko_hints`) writes a [`HintSet`], and the protocol
//! client loads it with [`crate::protocol::Client::from_hints`].
//!
//! A parity is the XOR of whole database records, so it has the record size
//! of the database. The record size is a field of the file.
//!
//! # File format (version 2)
//!
//! All integers are little-endian. `r` is the record size in bytes.
//!
//! | Offset | Size | Field                                              |
//! |--------|------|----------------------------------------------------|
//! | 0      | 8    | magic `PLNKHINT`                                   |
//! | 8      | 4    | version (`2`)                                      |
//! | 12     | 4    | `record_size` (`r`), 1 to [`MAX_RECORD_SIZE`]      |
//! | 16     | 8    | `num_entries`: real database records               |
//! | 24     | 8    | `block_size` (`w`)                                 |
//! | 32     | 8    | `num_blocks` (`c`)                                 |
//! | 40     | 8    | `num_regular`: number of regular hints             |
//! | 48     | 8    | `num_backup`: number of backup hints               |
//! | 56     | 8    | seed check (see [`seed_check`])                    |
//! | 64     | `4 + r` each | regular hints: cutoff (`u32`), parity      |
//! | ...    | `4 + 2r` each | backup hints: cutoff (`u32`), parity in, parity out |
//! | end-32 | 32   | BLAKE3 hash of all bytes before it                 |
//!
//! Hint `j` of the regular hints has hint ID `j`. Backup hint `b` has hint ID
//! `num_regular + b`. The cutoff defines the block subset of the hint, see
//! [`crate::subsets`].
//!
//! Version 1 had 32-byte parities over the first 32 bytes of a 40-byte record.
//! This module does not read it.
//!
//! The file does not contain the master seed. The client needs both. The file
//! is client-secret all the same: the parities and cutoffs belong to one client.

use sha2::{Digest, Sha256};
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::io::{self, Read, Write};

/// Largest record size in bytes that a hint set can have.
pub const MAX_RECORD_SIZE: usize = 1 << 20;
/// First bytes of a hint file.
pub const HINT_FILE_MAGIC: [u8; 8] = *b"PLNKHINT";
/// Format version that this module reads and writes.
pub const HINT_FILE_VERSION: u32 = 2;

const HEADER_SIZE: usize = 64;
const CHECKSUM_SIZE: usize = 32;
const CUTOFF_SIZE: usize = 4;
const SEED_CHECK_LABEL: &[u8] = b"plinko_seed_check";
/// Upper bound on the hints whose cutoffs are allocated before their records are read.
const MAX_PREALLOCATED_HINTS: usize = 1 << 20;
/// Upper bound on the parity bytes that are allocated before the records are read.
const MAX_PREALLOCATED_BYTES: usize = 1 << 26;

/// Database geometry that a hint set is computed for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HintGeometry {
    /// Number of real database records (without padding).
    pub num_entries: u64,
    /// Bytes of one database record, and of one parity.
    pub record_size: u32,
    /// Records per block (`w`), the iPRF range.
    pub block_size: u64,
    /// Number of blocks (`c`), padding included.
    pub num_blocks: u64,
}

/// All hints of one client for one database and one master seed.
///
/// A regular hint has a cutoff and one parity: the XOR of one record from each
/// block of a subset of `c/2 + 1` blocks. A backup hint has a cutoff and two
/// parities: one for an "in" subset of `c/2` blocks and one for its complement
/// "out". A cutoff of `INVALID_CUTOFF` (see [`crate::subsets`]) marks a hint
/// with no valid subset; its parities stay zero.
///
/// The parities are stored in two flat buffers, so that a hint set of millions
/// of hints is a few allocations.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HintSet {
    geometry: HintGeometry,
    seed_check: [u8; 8],
    regular_cutoffs: Vec<u32>,
    /// `regular_cutoffs.len() * record_size` bytes.
    regular_parities: Vec<u8>,
    backup_cutoffs: Vec<u32>,
    /// `backup_cutoffs.len() * 2 * record_size` bytes: parity in, then parity out.
    backup_parities: Vec<u8>,
}

/// Check value that binds a hint set to its master seed without revealing the seed.
///
/// Computed as `SHA-256(master_seed || "plinko_seed_check")[0..8]`.
///
/// # Examples
///
/// ```
/// use plinko::hints::seed_check;
///
/// assert_eq!(seed_check(&[1u8; 32]), seed_check(&[1u8; 32]));
/// assert_ne!(seed_check(&[1u8; 32]), seed_check(&[2u8; 32]));
/// ```
pub fn seed_check(master_seed: &[u8; 32]) -> [u8; 8] {
    let mut hasher = Sha256::new();
    hasher.update(master_seed);
    hasher.update(SEED_CHECK_LABEL);
    let hash = hasher.finalize();
    let mut check = [0u8; 8];
    check.copy_from_slice(&hash[0..8]);
    check
}

/// Error from [`HintSet::new`] and [`HintSet::read_from`].
#[derive(Debug)]
pub enum HintFileError {
    /// The reader failed, or the data ended before the end of the file format.
    Io(io::Error),
    /// The data does not start with [`HINT_FILE_MAGIC`].
    BadMagic,
    /// The file has a version that this code does not read.
    UnsupportedVersion(u32),
    /// The record size is zero or above [`MAX_RECORD_SIZE`].
    InvalidRecordSize(u32),
    /// The parities of this many hints do not fit in memory on this platform.
    TooManyHints(u64),
    /// The stored hash does not match the content.
    ChecksumMismatch,
    /// Bytes follow the checksum.
    TrailingData,
}

impl Display for HintFileError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(error) => write!(f, "cannot read hint file: {error}"),
            Self::BadMagic => write!(f, "not a Plinko hint file (bad magic)"),
            Self::UnsupportedVersion(version) => {
                write!(f, "unsupported hint file version {version}")
            }
            Self::InvalidRecordSize(size) => {
                write!(
                    f,
                    "record size {size} is not in the range 1 to {MAX_RECORD_SIZE}"
                )
            }
            Self::TooManyHints(count) => write!(f, "hint count {count} is too large"),
            Self::ChecksumMismatch => write!(f, "hint file checksum mismatch"),
            Self::TrailingData => write!(f, "hint file has data after its checksum"),
        }
    }
}

impl Error for HintFileError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            _ => None,
        }
    }
}

impl From<io::Error> for HintFileError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

struct HashingWriter<W: Write> {
    inner: W,
    hasher: blake3::Hasher,
}

impl<W: Write> HashingWriter<W> {
    fn write_hashed(&mut self, bytes: &[u8]) -> io::Result<()> {
        self.hasher.update(bytes);
        self.inner.write_all(bytes)
    }
}

struct HashingReader<R: Read> {
    inner: R,
    hasher: blake3::Hasher,
}

impl<R: Read> HashingReader<R> {
    fn read_hashed(&mut self, bytes: &mut [u8]) -> io::Result<()> {
        self.inner.read_exact(bytes)?;
        self.hasher.update(bytes);
        Ok(())
    }
}

fn u32_at(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(bytes[offset..offset + 4].try_into().expect("4 bytes"))
}

fn u64_at(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(bytes[offset..offset + 8].try_into().expect("8 bytes"))
}

fn hint_count(count: u64) -> Result<usize, HintFileError> {
    usize::try_from(count).map_err(|_| HintFileError::TooManyHints(count))
}

/// Returns the record size as `usize` if it is in the valid range.
fn checked_record_size(record_size: u32) -> Result<usize, HintFileError> {
    match usize::try_from(record_size) {
        Ok(size) if (1..=MAX_RECORD_SIZE).contains(&size) => Ok(size),
        _ => Err(HintFileError::InvalidRecordSize(record_size)),
    }
}

/// Returns `hints * parities_per_hint * record_size`, or an error on overflow.
fn parity_bytes(
    hints: usize,
    parities_per_hint: usize,
    record_size: usize,
) -> Result<usize, HintFileError> {
    hints
        .checked_mul(parities_per_hint)
        .and_then(|parities| parities.checked_mul(record_size))
        .filter(|&bytes| bytes <= isize::MAX as usize)
        .ok_or(HintFileError::TooManyHints(hints as u64))
}

impl HintSet {
    /// Creates a hint set with the given cutoffs and zero parities.
    ///
    /// The number of regular hints is `regular_cutoffs.len()`, and the number
    /// of backup hints is `backup_cutoffs.len()`.
    ///
    /// # Errors
    /// Returns [`HintFileError::InvalidRecordSize`] if `geometry.record_size`
    /// is zero or above [`MAX_RECORD_SIZE`], and [`HintFileError::TooManyHints`]
    /// if the parities do not fit in memory.
    ///
    /// # Examples
    ///
    /// ```
    /// use plinko::hints::{seed_check, HintGeometry, HintSet};
    ///
    /// let geometry = HintGeometry { num_entries: 4, record_size: 5, block_size: 2, num_blocks: 2 };
    /// let mut hints = HintSet::new(geometry, seed_check(&[7u8; 32]), vec![9, 11], vec![13]).unwrap();
    /// assert_eq!((hints.num_regular(), hints.num_backup()), (2, 1));
    /// assert_eq!(hints.regular_parity(1), [0u8; 5]);
    /// hints.regular_parity_mut(1).copy_from_slice(&[1, 2, 3, 4, 5]);
    /// assert_eq!(hints.regular_parity(1), [1, 2, 3, 4, 5]);
    /// assert_eq!(hints.regular_parity(0), [0u8; 5]);
    /// ```
    pub fn new(
        geometry: HintGeometry,
        seed_check: [u8; 8],
        regular_cutoffs: Vec<u32>,
        backup_cutoffs: Vec<u32>,
    ) -> Result<Self, HintFileError> {
        let record_size: usize = checked_record_size(geometry.record_size)?;
        let regular_bytes: usize = parity_bytes(regular_cutoffs.len(), 1, record_size)?;
        let backup_bytes: usize = parity_bytes(backup_cutoffs.len(), 2, record_size)?;
        Ok(Self {
            geometry,
            seed_check,
            regular_parities: vec![0u8; regular_bytes],
            backup_parities: vec![0u8; backup_bytes],
            regular_cutoffs,
            backup_cutoffs,
        })
    }

    /// Geometry of the database that the hints belong to.
    pub fn geometry(&self) -> HintGeometry {
        self.geometry
    }

    /// Check value of the master seed, from [`seed_check`].
    pub fn seed_check(&self) -> [u8; 8] {
        self.seed_check
    }

    /// Bytes of one database record, and of one parity.
    pub fn record_size(&self) -> usize {
        self.geometry.record_size as usize
    }

    /// Number of regular hints.
    pub fn num_regular(&self) -> usize {
        self.regular_cutoffs.len()
    }

    /// Number of backup hints.
    pub fn num_backup(&self) -> usize {
        self.backup_cutoffs.len()
    }

    /// Cutoffs of the regular hints; index `j` is hint ID `j`.
    pub fn regular_cutoffs(&self) -> &[u32] {
        &self.regular_cutoffs
    }

    /// Cutoffs of the backup hints; index `b` is hint ID `num_regular() + b`.
    pub fn backup_cutoffs(&self) -> &[u32] {
        &self.backup_cutoffs
    }

    /// Parity of regular hint `j`: the XOR of the records in its subset.
    ///
    /// # Panics
    /// Panics if `j >= num_regular()`. The other parity accessors panic in the
    /// same way for an index that is out of range.
    pub fn regular_parity(&self, j: usize) -> &[u8] {
        let size: usize = self.record_size();
        &self.regular_parities[j * size..(j + 1) * size]
    }

    /// Mutable parity of regular hint `j`.
    pub fn regular_parity_mut(&mut self, j: usize) -> &mut [u8] {
        let size: usize = self.record_size();
        &mut self.regular_parities[j * size..(j + 1) * size]
    }

    /// Parity of backup hint `b` over the records whose block is in its subset.
    pub fn backup_parity_in(&self, b: usize) -> &[u8] {
        let size: usize = self.record_size();
        &self.backup_parities[2 * b * size..(2 * b + 1) * size]
    }

    /// Mutable "in" parity of backup hint `b`.
    pub fn backup_parity_in_mut(&mut self, b: usize) -> &mut [u8] {
        let size: usize = self.record_size();
        &mut self.backup_parities[2 * b * size..(2 * b + 1) * size]
    }

    /// Parity of backup hint `b` over the records whose block is outside its subset.
    pub fn backup_parity_out(&self, b: usize) -> &[u8] {
        let size: usize = self.record_size();
        &self.backup_parities[(2 * b + 1) * size..(2 * b + 2) * size]
    }

    /// Mutable "out" parity of backup hint `b`.
    pub fn backup_parity_out_mut(&mut self, b: usize) -> &mut [u8] {
        let size: usize = self.record_size();
        &mut self.backup_parities[(2 * b + 1) * size..(2 * b + 2) * size]
    }

    /// Size of the serialized hint set in bytes.
    pub fn serialized_len(&self) -> u64 {
        let size: u64 = self.record_size() as u64;
        HEADER_SIZE as u64
            + self.num_regular() as u64 * (CUTOFF_SIZE as u64 + size)
            + self.num_backup() as u64 * (CUTOFF_SIZE as u64 + 2 * size)
            + CHECKSUM_SIZE as u64
    }

    /// Writes the hint set in the format of the module documentation.
    ///
    /// Pass a buffered writer; the function writes one hint at a time.
    ///
    /// # Examples
    ///
    /// ```
    /// use plinko::hints::{seed_check, HintGeometry, HintSet};
    ///
    /// let geometry = HintGeometry { num_entries: 4, record_size: 40, block_size: 2, num_blocks: 2 };
    /// let mut hints = HintSet::new(geometry, seed_check(&[7u8; 32]), vec![9], Vec::new()).unwrap();
    /// hints.regular_parity_mut(0).fill(1);
    /// let mut bytes: Vec<u8> = Vec::new();
    /// hints.write_to(&mut bytes).unwrap();
    /// assert_eq!(bytes.len() as u64, hints.serialized_len());
    /// assert_eq!(HintSet::read_from(bytes.as_slice()).unwrap(), hints);
    /// ```
    pub fn write_to<W: Write>(&self, writer: W) -> io::Result<()> {
        let mut out = HashingWriter {
            inner: writer,
            hasher: blake3::Hasher::new(),
        };

        let mut header = [0u8; HEADER_SIZE];
        header[0..8].copy_from_slice(&HINT_FILE_MAGIC);
        header[8..12].copy_from_slice(&HINT_FILE_VERSION.to_le_bytes());
        header[12..16].copy_from_slice(&self.geometry.record_size.to_le_bytes());
        header[16..24].copy_from_slice(&self.geometry.num_entries.to_le_bytes());
        header[24..32].copy_from_slice(&self.geometry.block_size.to_le_bytes());
        header[32..40].copy_from_slice(&self.geometry.num_blocks.to_le_bytes());
        header[40..48].copy_from_slice(&(self.num_regular() as u64).to_le_bytes());
        header[48..56].copy_from_slice(&(self.num_backup() as u64).to_le_bytes());
        header[56..64].copy_from_slice(&self.seed_check);
        out.write_hashed(&header)?;

        let size: usize = self.record_size();
        for (cutoff, parity) in self
            .regular_cutoffs
            .iter()
            .zip(self.regular_parities.chunks_exact(size))
        {
            out.write_hashed(&cutoff.to_le_bytes())?;
            out.write_hashed(parity)?;
        }
        for (cutoff, parities) in self
            .backup_cutoffs
            .iter()
            .zip(self.backup_parities.chunks_exact(2 * size))
        {
            out.write_hashed(&cutoff.to_le_bytes())?;
            out.write_hashed(parities)?;
        }

        let checksum = out.hasher.finalize();
        out.inner.write_all(checksum.as_bytes())?;
        out.inner.flush()
    }

    /// Reads a hint set and checks its magic, version, record size, length, and checksum.
    ///
    /// Pass a buffered reader; the function reads one hint at a time. It does
    /// not check the geometry against a database or the seed check against a
    /// seed; [`crate::protocol::Client::from_hints`] does that.
    pub fn read_from<R: Read>(reader: R) -> Result<Self, HintFileError> {
        let mut input = HashingReader {
            inner: reader,
            hasher: blake3::Hasher::new(),
        };

        let mut header = [0u8; HEADER_SIZE];
        input.read_hashed(&mut header)?;
        if header[0..8] != HINT_FILE_MAGIC {
            return Err(HintFileError::BadMagic);
        }
        let version: u32 = u32_at(&header, 8);
        if version != HINT_FILE_VERSION {
            return Err(HintFileError::UnsupportedVersion(version));
        }
        let geometry = HintGeometry {
            record_size: u32_at(&header, 12),
            num_entries: u64_at(&header, 16),
            block_size: u64_at(&header, 24),
            num_blocks: u64_at(&header, 32),
        };
        let size: usize = checked_record_size(geometry.record_size)?;
        let num_regular: usize = hint_count(u64_at(&header, 40))?;
        let num_backup: usize = hint_count(u64_at(&header, 48))?;
        let regular_bytes: usize = parity_bytes(num_regular, 1, size)?;
        let backup_bytes: usize = parity_bytes(num_backup, 2, size)?;
        let mut seed_check = [0u8; 8];
        seed_check.copy_from_slice(&header[56..64]);

        // The counts come from the file: allocate as the records arrive.
        let mut record: Vec<u8> = vec![0u8; CUTOFF_SIZE + 2 * size];

        let mut regular_cutoffs: Vec<u32> =
            Vec::with_capacity(num_regular.min(MAX_PREALLOCATED_HINTS));
        let mut regular_parities: Vec<u8> =
            Vec::with_capacity(regular_bytes.min(MAX_PREALLOCATED_BYTES));
        for _ in 0..num_regular {
            let record: &mut [u8] = &mut record[..CUTOFF_SIZE + size];
            input.read_hashed(record)?;
            regular_cutoffs.push(u32_at(record, 0));
            regular_parities.extend_from_slice(&record[CUTOFF_SIZE..]);
        }

        let mut backup_cutoffs: Vec<u32> =
            Vec::with_capacity(num_backup.min(MAX_PREALLOCATED_HINTS));
        let mut backup_parities: Vec<u8> =
            Vec::with_capacity(backup_bytes.min(MAX_PREALLOCATED_BYTES));
        for _ in 0..num_backup {
            input.read_hashed(&mut record)?;
            backup_cutoffs.push(u32_at(&record, 0));
            backup_parities.extend_from_slice(&record[CUTOFF_SIZE..]);
        }

        let mut stored = [0u8; CHECKSUM_SIZE];
        input.inner.read_exact(&mut stored)?;
        if input.hasher.finalize() != stored {
            return Err(HintFileError::ChecksumMismatch);
        }

        let mut extra = [0u8; 1];
        loop {
            match input.inner.read(&mut extra) {
                Ok(0) => break,
                Ok(_) => return Err(HintFileError::TrailingData),
                Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
                Err(error) => return Err(HintFileError::Io(error)),
            }
        }

        Ok(Self {
            geometry,
            seed_check,
            regular_cutoffs,
            regular_parities,
            backup_cutoffs,
            backup_parities,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Record size 5: not a multiple of 8, and smaller than a cutoff plus a word.
    fn sample() -> HintSet {
        let geometry = HintGeometry {
            num_entries: 90,
            record_size: 5,
            block_size: 16,
            num_blocks: 6,
        };
        let mut hints: HintSet = HintSet::new(
            geometry,
            [1, 2, 3, 4, 5, 6, 7, 8],
            vec![0x0A0B_0C0D, 0],
            vec![0xFFFF_FFFE],
        )
        .expect("valid record size");
        hints.regular_parity_mut(0).fill(0x11);
        hints.backup_parity_in_mut(0).fill(0x22);
        hints.backup_parity_out_mut(0).fill(0x33);
        hints
    }

    fn serialized(hints: &HintSet) -> Vec<u8> {
        let mut bytes: Vec<u8> = Vec::new();
        hints.write_to(&mut bytes).expect("write to a Vec");
        bytes
    }

    #[test]
    fn layout_matches_the_documented_offsets() {
        let hints: HintSet = sample();
        let bytes: Vec<u8> = serialized(&hints);

        assert_eq!(bytes.len(), 64 + 2 * 9 + 14 + 32);
        assert_eq!(bytes.len() as u64, hints.serialized_len());
        assert_eq!(&bytes[0..8], b"PLNKHINT");
        assert_eq!(bytes[8..12], [2, 0, 0, 0]);
        assert_eq!(bytes[12..16], [5, 0, 0, 0]);
        assert_eq!(bytes[16..24], [90, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(bytes[24..32], [16, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(bytes[32..40], [6, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(bytes[40..48], [2, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(bytes[48..56], [1, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(bytes[56..64], [1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(bytes[64..68], [0x0D, 0x0C, 0x0B, 0x0A]);
        assert_eq!(bytes[68..73], [0x11; 5]);
        assert_eq!(bytes[73..82], [0; 9]);
        let backup_start: usize = 64 + 2 * 9;
        assert_eq!(
            bytes[backup_start..backup_start + 4],
            [0xFE, 0xFF, 0xFF, 0xFF]
        );
        assert_eq!(bytes[backup_start + 4..backup_start + 9], [0x22; 5]);
        assert_eq!(bytes[backup_start + 9..backup_start + 14], [0x33; 5]);
        assert_eq!(
            bytes[backup_start + 14..],
            *blake3::hash(&bytes[..backup_start + 14]).as_bytes()
        );
    }

    #[test]
    fn accessors_address_one_parity_each() {
        let geometry = HintGeometry {
            num_entries: 8,
            record_size: 3,
            block_size: 4,
            num_blocks: 2,
        };
        let mut hints: HintSet =
            HintSet::new(geometry, [0u8; 8], vec![1, 2, 3], vec![4, 5]).expect("hint set");
        assert_eq!(hints.regular_cutoffs(), [1, 2, 3]);
        assert_eq!(hints.backup_cutoffs(), [4, 5]);

        hints.regular_parity_mut(1).copy_from_slice(&[1, 2, 3]);
        hints.backup_parity_in_mut(1).copy_from_slice(&[4, 5, 6]);
        hints.backup_parity_out_mut(0).copy_from_slice(&[7, 8, 9]);

        assert_eq!(hints.regular_parity(0), [0, 0, 0]);
        assert_eq!(hints.regular_parity(1), [1, 2, 3]);
        assert_eq!(hints.regular_parity(2), [0, 0, 0]);
        assert_eq!(hints.backup_parity_in(0), [0, 0, 0]);
        assert_eq!(hints.backup_parity_out(0), [7, 8, 9]);
        assert_eq!(hints.backup_parity_in(1), [4, 5, 6]);
        assert_eq!(hints.backup_parity_out(1), [0, 0, 0]);
    }

    #[test]
    fn roundtrip_preserves_the_hint_set() {
        let hints: HintSet = sample();
        let read: HintSet = HintSet::read_from(serialized(&hints).as_slice()).expect("read");
        assert_eq!(read, hints);

        let empty: HintSet =
            HintSet::new(hints.geometry(), hints.seed_check(), Vec::new(), Vec::new())
                .expect("empty hint set");
        let read: HintSet = HintSet::read_from(serialized(&empty).as_slice()).expect("read");
        assert_eq!(read, empty);

        for record_size in [1u32, 8, 40, 256] {
            let geometry = HintGeometry {
                record_size,
                ..hints.geometry()
            };
            let mut wide: HintSet =
                HintSet::new(geometry, [9u8; 8], vec![7, 8], vec![9]).expect("hint set");
            for (i, byte) in wide.regular_parity_mut(1).iter_mut().enumerate() {
                *byte = i as u8 ^ 0x5A;
            }
            wide.backup_parity_out_mut(0).fill(0xC3);
            let bytes: Vec<u8> = serialized(&wide);
            assert_eq!(bytes.len() as u64, wide.serialized_len());
            assert_eq!(
                HintSet::read_from(bytes.as_slice()).expect("read"),
                wide,
                "record size {record_size}"
            );
        }
    }

    #[test]
    fn every_flipped_bit_position_is_detected() {
        let bytes: Vec<u8> = serialized(&sample());
        for position in 0..bytes.len() {
            let mut corrupted: Vec<u8> = bytes.clone();
            corrupted[position] ^= 0x01;
            assert!(
                HintSet::read_from(corrupted.as_slice()).is_err(),
                "a flipped bit in byte {position} was accepted"
            );
        }
    }

    #[test]
    fn corruption_of_a_parity_is_a_checksum_mismatch() {
        let mut bytes: Vec<u8> = serialized(&sample());
        bytes[70] ^= 0x80;
        assert!(matches!(
            HintSet::read_from(bytes.as_slice()),
            Err(HintFileError::ChecksumMismatch)
        ));
    }

    #[test]
    fn every_truncation_is_detected() {
        let bytes: Vec<u8> = serialized(&sample());
        for len in 0..bytes.len() {
            let result = HintSet::read_from(&bytes[..len]);
            assert!(
                matches!(&result, Err(HintFileError::Io(error))
                    if error.kind() == io::ErrorKind::UnexpectedEof),
                "truncation to {len} bytes gave {result:?}"
            );
        }
    }

    #[test]
    fn trailing_data_is_rejected() {
        let mut bytes: Vec<u8> = serialized(&sample());
        bytes.push(0);
        assert!(matches!(
            HintSet::read_from(bytes.as_slice()),
            Err(HintFileError::TrailingData)
        ));
    }

    #[test]
    fn header_fields_are_validated() {
        let bytes: Vec<u8> = serialized(&sample());

        let mut bad_magic: Vec<u8> = bytes.clone();
        bad_magic[0] = b'X';
        assert!(matches!(
            HintSet::read_from(bad_magic.as_slice()),
            Err(HintFileError::BadMagic)
        ));

        for version in [1u8, 3] {
            let mut bad_version: Vec<u8> = bytes.clone();
            bad_version[8] = version;
            assert!(matches!(
                HintSet::read_from(bad_version.as_slice()),
                Err(HintFileError::UnsupportedVersion(found)) if found == u32::from(version)
            ));
        }

        let too_large: u32 = MAX_RECORD_SIZE as u32 + 1;
        for record_size in [0u32, too_large, u32::MAX] {
            let mut bad_record_size: Vec<u8> = bytes.clone();
            bad_record_size[12..16].copy_from_slice(&record_size.to_le_bytes());
            assert!(matches!(
                HintSet::read_from(bad_record_size.as_slice()),
                Err(HintFileError::InvalidRecordSize(found)) if found == record_size
            ));
        }
    }

    #[test]
    fn new_rejects_an_invalid_record_size() {
        for record_size in [0u32, MAX_RECORD_SIZE as u32 + 1] {
            let geometry = HintGeometry {
                record_size,
                ..sample().geometry()
            };
            assert!(matches!(
                HintSet::new(geometry, [0u8; 8], vec![1], vec![1]),
                Err(HintFileError::InvalidRecordSize(found)) if found == record_size
            ));
        }
    }

    #[test]
    fn a_huge_hint_count_fails_without_a_huge_allocation() {
        let mut bytes: Vec<u8> = serialized(&sample());
        bytes[40..48].copy_from_slice(&(u64::MAX / 2).to_le_bytes());
        assert!(HintSet::read_from(bytes.as_slice()).is_err());

        // A count that fits with this record size: the reader stops at the end of the data.
        let mut bytes: Vec<u8> = serialized(&sample());
        bytes[40..48].copy_from_slice(&(1u64 << 40).to_le_bytes());
        assert!(matches!(
            HintSet::read_from(bytes.as_slice()),
            Err(HintFileError::Io(error)) if error.kind() == io::ErrorKind::UnexpectedEof
        ));
    }
}
