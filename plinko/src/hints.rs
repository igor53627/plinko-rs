//! Hint set of a Plinko client and its on-disk format.
//!
//! The hint generator (`plinko_hints`) writes a [`HintSet`], and the protocol
//! client loads it with [`crate::protocol::Client::from_hints`].
//!
//! # File format (version 1)
//!
//! All integers are little-endian.
//!
//! | Offset | Size | Field                                              |
//! |--------|------|----------------------------------------------------|
//! | 0      | 8    | magic `PLNKHINT`                                   |
//! | 8      | 4    | version (`1`)                                      |
//! | 12     | 4    | parity size in bytes (`32`)                        |
//! | 16     | 8    | `num_entries`: real database entries               |
//! | 24     | 8    | `block_size` (`w`)                                 |
//! | 32     | 8    | `num_blocks` (`c`)                                 |
//! | 40     | 8    | `num_regular`: number of regular hints             |
//! | 48     | 8    | `num_backup`: number of backup hints               |
//! | 56     | 8    | seed check (see [`seed_check`])                    |
//! | 64     | 36 each | regular hints: cutoff (`u32`), parity           |
//! | ...    | 68 each | backup hints: cutoff (`u32`), parity in, parity out |
//! | end-32 | 32   | BLAKE3 hash of all bytes before it                 |
//!
//! Hint `j` of the regular hints has hint ID `j`. Backup hint `b` has hint ID
//! `num_regular + b`. The cutoff defines the block subset of the hint, see
//! [`crate::subsets`].
//!
//! The file does not contain the master seed. The client needs both. The file
//! is client-secret all the same: the parities and cutoffs belong to one client.

use sha2::{Digest, Sha256};
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::io::{self, Read, Write};

/// Bytes of a hint parity.
pub const PARITY_SIZE: usize = 32;
/// First bytes of a hint file.
pub const HINT_FILE_MAGIC: [u8; 8] = *b"PLNKHINT";
/// Format version that this module reads and writes.
pub const HINT_FILE_VERSION: u32 = 1;

const HEADER_SIZE: usize = 64;
const CHECKSUM_SIZE: usize = 32;
const REGULAR_RECORD_SIZE: usize = 4 + PARITY_SIZE;
const BACKUP_RECORD_SIZE: usize = 4 + 2 * PARITY_SIZE;
const SEED_CHECK_LABEL: &[u8] = b"plinko_seed_check";
/// Upper bound on the hints that are allocated before their records are read.
const MAX_PREALLOCATED_HINTS: usize = 1 << 20;

/// A regular hint: the XOR parity of one entry from each block of a subset of `c/2 + 1` blocks.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RegularHint {
    /// Cutoff that defines the block subset (see [`crate::subsets`]).
    /// `INVALID_CUTOFF` marks a hint with no valid subset; its parity stays zero.
    pub cutoff: u32,
    /// XOR of the entries in the subset.
    pub parity: [u8; PARITY_SIZE],
}

/// A backup hint: an "in" subset of `c/2` blocks and its complement "out", with one parity each.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BackupHint {
    /// Cutoff that defines the in/out partition (see [`crate::subsets`]).
    /// `INVALID_CUTOFF` marks a hint with no valid partition; its parities stay zero.
    pub cutoff: u32,
    /// XOR of the entries whose block is in the subset.
    pub parity_in: [u8; PARITY_SIZE],
    /// XOR of the entries whose block is outside the subset.
    pub parity_out: [u8; PARITY_SIZE],
}

/// Database geometry that a hint set is computed for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HintGeometry {
    /// Number of real database entries (without padding).
    pub num_entries: u64,
    /// Entries per block (`w`), the iPRF range.
    pub block_size: u64,
    /// Number of blocks (`c`), padding included.
    pub num_blocks: u64,
}

/// All hints of one client for one database and one master seed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HintSet {
    /// Geometry of the database.
    pub geometry: HintGeometry,
    /// Check value of the master seed, from [`seed_check`].
    pub seed_check: [u8; 8],
    /// Regular hints; index `j` is hint ID `j`.
    pub regular: Vec<RegularHint>,
    /// Backup hints; index `b` is hint ID `regular.len() + b`.
    pub backup: Vec<BackupHint>,
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

/// Error from [`HintSet::read_from`].
#[derive(Debug)]
pub enum HintFileError {
    /// The reader failed, or the data ended before the end of the file format.
    Io(io::Error),
    /// The data does not start with [`HINT_FILE_MAGIC`].
    BadMagic,
    /// The file has a version that this code does not read.
    UnsupportedVersion(u32),
    /// The file has a parity size other than [`PARITY_SIZE`].
    UnsupportedParitySize(u32),
    /// A hint count does not fit in memory on this platform.
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
            Self::UnsupportedParitySize(size) => {
                write!(f, "unsupported parity size {size} in hint file")
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

fn parity_at(bytes: &[u8], offset: usize) -> [u8; PARITY_SIZE] {
    bytes[offset..offset + PARITY_SIZE]
        .try_into()
        .expect("one parity")
}

fn hint_count(count: u64) -> Result<usize, HintFileError> {
    usize::try_from(count).map_err(|_| HintFileError::TooManyHints(count))
}

impl HintSet {
    /// Size of the serialized hint set in bytes.
    pub fn serialized_len(&self) -> u64 {
        HEADER_SIZE as u64
            + self.regular.len() as u64 * REGULAR_RECORD_SIZE as u64
            + self.backup.len() as u64 * BACKUP_RECORD_SIZE as u64
            + CHECKSUM_SIZE as u64
    }

    /// Writes the hint set in the format of the module documentation.
    ///
    /// Pass a buffered writer; the function writes one hint at a time.
    ///
    /// # Examples
    ///
    /// ```
    /// use plinko::hints::{seed_check, HintGeometry, HintSet, RegularHint};
    ///
    /// let hints = HintSet {
    ///     geometry: HintGeometry { num_entries: 4, block_size: 2, num_blocks: 2 },
    ///     seed_check: seed_check(&[7u8; 32]),
    ///     regular: vec![RegularHint { cutoff: 9, parity: [1u8; 32] }],
    ///     backup: Vec::new(),
    /// };
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
        header[12..16].copy_from_slice(&(PARITY_SIZE as u32).to_le_bytes());
        header[16..24].copy_from_slice(&self.geometry.num_entries.to_le_bytes());
        header[24..32].copy_from_slice(&self.geometry.block_size.to_le_bytes());
        header[32..40].copy_from_slice(&self.geometry.num_blocks.to_le_bytes());
        header[40..48].copy_from_slice(&(self.regular.len() as u64).to_le_bytes());
        header[48..56].copy_from_slice(&(self.backup.len() as u64).to_le_bytes());
        header[56..64].copy_from_slice(&self.seed_check);
        out.write_hashed(&header)?;

        for hint in &self.regular {
            let mut record = [0u8; REGULAR_RECORD_SIZE];
            record[0..4].copy_from_slice(&hint.cutoff.to_le_bytes());
            record[4..].copy_from_slice(&hint.parity);
            out.write_hashed(&record)?;
        }
        for hint in &self.backup {
            let mut record = [0u8; BACKUP_RECORD_SIZE];
            record[0..4].copy_from_slice(&hint.cutoff.to_le_bytes());
            record[4..4 + PARITY_SIZE].copy_from_slice(&hint.parity_in);
            record[4 + PARITY_SIZE..].copy_from_slice(&hint.parity_out);
            out.write_hashed(&record)?;
        }

        let checksum = out.hasher.finalize();
        out.inner.write_all(checksum.as_bytes())?;
        out.inner.flush()
    }

    /// Reads a hint set and checks its magic, version, length, and checksum.
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
        let parity_size: u32 = u32_at(&header, 12);
        if parity_size as usize != PARITY_SIZE {
            return Err(HintFileError::UnsupportedParitySize(parity_size));
        }
        let geometry = HintGeometry {
            num_entries: u64_at(&header, 16),
            block_size: u64_at(&header, 24),
            num_blocks: u64_at(&header, 32),
        };
        let num_regular: usize = hint_count(u64_at(&header, 40))?;
        let num_backup: usize = hint_count(u64_at(&header, 48))?;
        let mut seed_check = [0u8; 8];
        seed_check.copy_from_slice(&header[56..64]);

        let mut regular: Vec<RegularHint> =
            Vec::with_capacity(num_regular.min(MAX_PREALLOCATED_HINTS));
        for _ in 0..num_regular {
            let mut record = [0u8; REGULAR_RECORD_SIZE];
            input.read_hashed(&mut record)?;
            regular.push(RegularHint {
                cutoff: u32_at(&record, 0),
                parity: parity_at(&record, 4),
            });
        }

        let mut backup: Vec<BackupHint> =
            Vec::with_capacity(num_backup.min(MAX_PREALLOCATED_HINTS));
        for _ in 0..num_backup {
            let mut record = [0u8; BACKUP_RECORD_SIZE];
            input.read_hashed(&mut record)?;
            backup.push(BackupHint {
                cutoff: u32_at(&record, 0),
                parity_in: parity_at(&record, 4),
                parity_out: parity_at(&record, 4 + PARITY_SIZE),
            });
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
            regular,
            backup,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> HintSet {
        HintSet {
            geometry: HintGeometry {
                num_entries: 90,
                block_size: 16,
                num_blocks: 6,
            },
            seed_check: [1, 2, 3, 4, 5, 6, 7, 8],
            regular: vec![
                RegularHint {
                    cutoff: 0x0A0B_0C0D,
                    parity: [0x11; 32],
                },
                RegularHint {
                    cutoff: 0,
                    parity: [0; 32],
                },
            ],
            backup: vec![BackupHint {
                cutoff: 0xFFFF_FFFE,
                parity_in: [0x22; 32],
                parity_out: [0x33; 32],
            }],
        }
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

        assert_eq!(bytes.len(), 64 + 2 * 36 + 68 + 32);
        assert_eq!(bytes.len() as u64, hints.serialized_len());
        assert_eq!(&bytes[0..8], b"PLNKHINT");
        assert_eq!(bytes[8..12], [1, 0, 0, 0]);
        assert_eq!(bytes[12..16], [32, 0, 0, 0]);
        assert_eq!(bytes[16..24], [90, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(bytes[24..32], [16, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(bytes[32..40], [6, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(bytes[40..48], [2, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(bytes[48..56], [1, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(bytes[56..64], [1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(bytes[64..68], [0x0D, 0x0C, 0x0B, 0x0A]);
        assert_eq!(bytes[68..100], [0x11; 32]);
        let backup_start: usize = 64 + 2 * 36;
        assert_eq!(
            bytes[backup_start..backup_start + 4],
            [0xFE, 0xFF, 0xFF, 0xFF]
        );
        assert_eq!(bytes[backup_start + 4..backup_start + 36], [0x22; 32]);
        assert_eq!(bytes[backup_start + 36..backup_start + 68], [0x33; 32]);
        assert_eq!(
            bytes[backup_start + 68..],
            *blake3::hash(&bytes[..backup_start + 68]).as_bytes()
        );
    }

    #[test]
    fn roundtrip_preserves_the_hint_set() {
        let hints: HintSet = sample();
        let read: HintSet = HintSet::read_from(serialized(&hints).as_slice()).expect("read");
        assert_eq!(read, hints);

        let empty = HintSet {
            regular: Vec::new(),
            backup: Vec::new(),
            ..sample()
        };
        let read: HintSet = HintSet::read_from(serialized(&empty).as_slice()).expect("read");
        assert_eq!(read, empty);
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

        let mut bad_version: Vec<u8> = bytes.clone();
        bad_version[8] = 2;
        assert!(matches!(
            HintSet::read_from(bad_version.as_slice()),
            Err(HintFileError::UnsupportedVersion(2))
        ));

        let mut bad_parity_size: Vec<u8> = bytes.clone();
        bad_parity_size[12] = 40;
        assert!(matches!(
            HintSet::read_from(bad_parity_size.as_slice()),
            Err(HintFileError::UnsupportedParitySize(40))
        ));
    }

    #[test]
    fn a_huge_hint_count_fails_without_a_huge_allocation() {
        let mut bytes: Vec<u8> = serialized(&sample());
        bytes[40..48].copy_from_slice(&(u64::MAX / 2).to_le_bytes());
        assert!(HintSet::read_from(bytes.as_slice()).is_err());
    }
}
