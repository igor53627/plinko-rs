//! Delta of a database: the changes from one database version to a later one.
//!
//! The publisher of a database gives each state of the database a version
//! number. A [`Delta`] holds the records that differ between two versions,
//! each as `old value XOR new value`. A client applies the deltas in order to
//! keep its hints current ([`crate::protocol::Client::apply_delta`]); it
//! needs no old value for that.
//!
//! A delta changes records at fixed indices. The number of records and the
//! record size are the same before and after it.
//!
//! # File format (version 1)
//!
//! All integers are little-endian. `r` is the record size in bytes.
//!
//! | Offset | Size | Field                                                   |
//! |--------|------|---------------------------------------------------------|
//! | 0      | 8    | magic `PLNKDLTA`                                        |
//! | 8      | 4    | format version (`1`)                                    |
//! | 12     | 4    | `record_size` (`r`), 1 to [`MAX_RECORD_SIZE`]           |
//! | 16     | 8    | `num_entries`: records of the database                  |
//! | 24     | 8    | `from_version`: database version before the changes     |
//! | 32     | 8    | `to_version`: database version after them, above `from_version` |
//! | 40     | 8    | `count`: number of changed records                      |
//! | 48     | `8 + r` each | changes: index (`u64`), `old XOR new` (`r` bytes) |
//! | end-32 | 32   | BLAKE3 hash of all bytes before it                      |
//!
//! The indices of the changes are strictly increasing, so a record occurs at
//! most once. A change has at least one byte that is not zero.
//!
//! A version number identifies a state of one database. A delta has no field
//! that identifies its database: the caller must give a client only the
//! deltas of the database that its hints belong to.
//!
//! The hash detects damage. It does not authenticate the delta: a client
//! cannot check that a delta holds the changes that the publisher made.

use crate::checked_io::{
    u32_at, u64_at, HashingReader, HashingWriter, TrailerError, CHECKSUM_SIZE,
};
use crate::hints::MAX_RECORD_SIZE;
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::io::{self, Read, Write};

/// First bytes of a delta file.
pub const DELTA_FILE_MAGIC: [u8; 8] = *b"PLNKDLTA";
/// Format version that this module reads and writes.
pub const DELTA_FILE_VERSION: u32 = 1;

const HEADER_SIZE: usize = 48;
const INDEX_SIZE: usize = 8;
/// Upper bound on the changes whose indices are allocated before they are read.
const MAX_PREALLOCATED_CHANGES: usize = 1 << 20;
/// Upper bound on the delta bytes that are allocated before the changes are read.
const MAX_PREALLOCATED_BYTES: usize = 1 << 26;

/// Database and versions that a delta belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeltaHeader {
    /// Number of records of the database, before and after the delta.
    pub num_entries: u64,
    /// Bytes of one database record.
    pub record_size: u32,
    /// Version of the database before the changes.
    pub from_version: u64,
    /// Version of the database after the changes; above `from_version`.
    pub to_version: u64,
}

/// The records that differ between two versions of a database.
///
/// Each change is `(index, old value XOR new value)`. The changes are in the
/// order of their indices, and each index occurs one time.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Delta {
    header: DeltaHeader,
    indices: Vec<u64>,
    /// `indices.len() * record_size` bytes.
    deltas: Vec<u8>,
}

/// Error from [`Delta::new`] and [`Delta::read_from`].
#[derive(Debug)]
pub enum DeltaError {
    /// The reader failed, or the data ended before the end of the file format.
    Io(io::Error),
    /// The data does not start with [`DELTA_FILE_MAGIC`].
    BadMagic,
    /// The file has a format version that this code does not read.
    UnsupportedVersion(u32),
    /// The record size is zero or above [`MAX_RECORD_SIZE`].
    InvalidRecordSize(u32),
    /// `to_version` is not above `from_version`.
    InvalidVersions { from_version: u64, to_version: u64 },
    /// A change has an index that is not below `num_entries`.
    IndexOutOfRange { index: u64, num_entries: u64 },
    /// The indices of a file are not strictly increasing at this index.
    IndexOrder { index: u64 },
    /// A change of a file has only zero bytes: the record did not change.
    ZeroChange { index: u64 },
    /// A change does not have the record size of the delta.
    RecordSizeMismatch { expected: usize, got: usize },
    /// The file has more changes than the database has records, or more than
    /// fit in memory on this platform.
    TooManyChanges(u64),
    /// The stored hash does not match the content.
    ChecksumMismatch,
    /// Bytes follow the checksum.
    TrailingData,
}

impl Display for DeltaError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(error) => write!(f, "cannot read delta file: {error}"),
            Self::BadMagic => write!(f, "not a Plinko delta file (bad magic)"),
            Self::UnsupportedVersion(version) => {
                write!(f, "unsupported delta file version {version}")
            }
            Self::InvalidRecordSize(size) => {
                write!(
                    f,
                    "record size {size} is not in the range 1 to {MAX_RECORD_SIZE}"
                )
            }
            Self::InvalidVersions {
                from_version,
                to_version,
            } => write!(
                f,
                "delta version {to_version} is not above the version {from_version} that it starts from"
            ),
            Self::IndexOutOfRange { index, num_entries } => {
                write!(f, "delta index {index} out of range [0, {num_entries})")
            }
            Self::IndexOrder { index } => {
                write!(f, "delta indices are not strictly increasing at index {index}")
            }
            Self::ZeroChange { index } => {
                write!(f, "delta change of index {index} has only zero bytes")
            }
            Self::RecordSizeMismatch { expected, got } => {
                write!(
                    f,
                    "record size mismatch: expected {expected} bytes, got {got}"
                )
            }
            Self::TooManyChanges(count) => write!(f, "change count {count} is too large"),
            Self::ChecksumMismatch => write!(f, "delta file checksum mismatch"),
            Self::TrailingData => write!(f, "delta file has data after its checksum"),
        }
    }
}

impl Error for DeltaError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            _ => None,
        }
    }
}

impl From<io::Error> for DeltaError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<TrailerError> for DeltaError {
    fn from(error: TrailerError) -> Self {
        match error {
            TrailerError::Io(error) => Self::Io(error),
            TrailerError::ChecksumMismatch => Self::ChecksumMismatch,
            TrailerError::TrailingData => Self::TrailingData,
        }
    }
}

/// Checks the record size and the versions of a header. Returns the record size.
fn check_header(header: &DeltaHeader) -> Result<usize, DeltaError> {
    let size: usize = match usize::try_from(header.record_size) {
        Ok(size) if (1..=MAX_RECORD_SIZE).contains(&size) => size,
        _ => return Err(DeltaError::InvalidRecordSize(header.record_size)),
    };
    if header.to_version <= header.from_version {
        return Err(DeltaError::InvalidVersions {
            from_version: header.from_version,
            to_version: header.to_version,
        });
    }
    Ok(size)
}

impl Delta {
    /// Builds a delta from changes `(index, old value XOR new value)` in any order.
    ///
    /// Changes of one index are combined in one change by XOR: the changes of
    /// a record that was written more than one time give the difference
    /// between its first and its last value. A change of zero bytes (the
    /// record has its old value) is removed.
    ///
    /// # Errors
    /// Returns [`DeltaError::InvalidRecordSize`] and
    /// [`DeltaError::InvalidVersions`] for a header that is not valid,
    /// [`DeltaError::IndexOutOfRange`] for an index that is not below
    /// `num_entries`, and [`DeltaError::RecordSizeMismatch`] for a change
    /// that does not have `record_size` bytes.
    ///
    /// # Examples
    ///
    /// ```
    /// use plinko::delta::{Delta, DeltaHeader};
    ///
    /// let header = DeltaHeader { num_entries: 10, record_size: 2, from_version: 4, to_version: 5 };
    /// let delta = Delta::new(header, vec![(7, vec![1, 2]), (3, vec![5, 5]), (7, vec![1, 0])]).unwrap();
    /// let changes: Vec<(u64, &[u8])> = delta.changes().collect();
    /// assert_eq!(changes, [(3, &[5u8, 5][..]), (7, &[0u8, 2][..])]);
    /// ```
    pub fn new(header: DeltaHeader, mut changes: Vec<(u64, Vec<u8>)>) -> Result<Self, DeltaError> {
        let size: usize = check_header(&header)?;
        for (index, delta) in &changes {
            if *index >= header.num_entries {
                return Err(DeltaError::IndexOutOfRange {
                    index: *index,
                    num_entries: header.num_entries,
                });
            }
            if delta.len() != size {
                return Err(DeltaError::RecordSizeMismatch {
                    expected: size,
                    got: delta.len(),
                });
            }
        }

        // The sort is stable. XOR does not depend on the order, so the result
        // would be the same with an unstable sort.
        changes.sort_by_key(|(index, _)| *index);

        let mut indices: Vec<u64> = Vec::with_capacity(changes.len());
        let mut deltas: Vec<u8> = Vec::with_capacity(changes.len() * size);
        for (index, delta) in changes {
            if indices.last() == Some(&index) {
                let start: usize = deltas.len() - size;
                for (combined, byte) in deltas[start..].iter_mut().zip(&delta) {
                    *combined ^= byte;
                }
            } else {
                drop_last_change_if_zero(&mut indices, &mut deltas, size);
                indices.push(index);
                deltas.extend_from_slice(&delta);
            }
        }
        drop_last_change_if_zero(&mut indices, &mut deltas, size);

        Ok(Self {
            header,
            indices,
            deltas,
        })
    }

    /// Database and versions that the delta belongs to.
    pub fn header(&self) -> DeltaHeader {
        self.header
    }

    /// Number of changed records.
    pub fn len(&self) -> usize {
        self.indices.len()
    }

    /// `true` if the delta changes no record. Such a delta still moves the
    /// database to `to_version`.
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }

    /// The changes `(index, old value XOR new value)`, in the order of their indices.
    pub fn changes(&self) -> impl Iterator<Item = (u64, &[u8])> + '_ {
        self.indices
            .iter()
            .copied()
            .zip(self.deltas.chunks_exact(self.header.record_size as usize))
    }

    /// Size of the serialized delta in bytes.
    pub fn serialized_len(&self) -> u64 {
        HEADER_SIZE as u64
            + self.len() as u64 * (INDEX_SIZE as u64 + u64::from(self.header.record_size))
            + CHECKSUM_SIZE as u64
    }

    /// Writes the delta in the format of the module documentation.
    ///
    /// Pass a buffered writer; the function writes one change at a time.
    ///
    /// # Examples
    ///
    /// ```
    /// use plinko::delta::{Delta, DeltaHeader};
    ///
    /// let header = DeltaHeader { num_entries: 10, record_size: 3, from_version: 0, to_version: 1 };
    /// let delta = Delta::new(header, vec![(9, vec![1, 2, 3])]).unwrap();
    /// let mut bytes: Vec<u8> = Vec::new();
    /// delta.write_to(&mut bytes).unwrap();
    /// assert_eq!(bytes.len() as u64, delta.serialized_len());
    /// assert_eq!(Delta::read_from(bytes.as_slice()).unwrap(), delta);
    /// ```
    pub fn write_to<W: Write>(&self, writer: W) -> io::Result<()> {
        let mut out = HashingWriter::new(writer);

        let mut header = [0u8; HEADER_SIZE];
        header[0..8].copy_from_slice(&DELTA_FILE_MAGIC);
        header[8..12].copy_from_slice(&DELTA_FILE_VERSION.to_le_bytes());
        header[12..16].copy_from_slice(&self.header.record_size.to_le_bytes());
        header[16..24].copy_from_slice(&self.header.num_entries.to_le_bytes());
        header[24..32].copy_from_slice(&self.header.from_version.to_le_bytes());
        header[32..40].copy_from_slice(&self.header.to_version.to_le_bytes());
        header[40..48].copy_from_slice(&(self.len() as u64).to_le_bytes());
        out.write_hashed(&header)?;

        for (index, delta) in self.changes() {
            out.write_hashed(&index.to_le_bytes())?;
            out.write_hashed(delta)?;
        }

        out.finish()
    }

    /// Reads a delta and checks its magic, format version, header, indices,
    /// length, and checksum.
    ///
    /// Pass a buffered reader; the function reads one change at a time. It
    /// does not compare the delta with a hint set.
    /// [`crate::protocol::Client::apply_delta`] compares the record size,
    /// the number of records and the start version. Nothing identifies the
    /// database; see the module documentation.
    pub fn read_from<R: Read>(reader: R) -> Result<Self, DeltaError> {
        let mut input = HashingReader::new(reader);

        let mut head = [0u8; HEADER_SIZE];
        input.read_hashed(&mut head)?;
        if head[0..8] != DELTA_FILE_MAGIC {
            return Err(DeltaError::BadMagic);
        }
        let version: u32 = u32_at(&head, 8);
        if version != DELTA_FILE_VERSION {
            return Err(DeltaError::UnsupportedVersion(version));
        }
        let header = DeltaHeader {
            record_size: u32_at(&head, 12),
            num_entries: u64_at(&head, 16),
            from_version: u64_at(&head, 24),
            to_version: u64_at(&head, 32),
        };
        let size: usize = check_header(&header)?;

        // Each record occurs at most once, so a valid count is at most `num_entries`.
        let stored_count: u64 = u64_at(&head, 40);
        let count: usize = usize::try_from(stored_count)
            .ok()
            .filter(|_| stored_count <= header.num_entries)
            .ok_or(DeltaError::TooManyChanges(stored_count))?;
        let delta_bytes: usize = count
            .checked_mul(size)
            .filter(|&bytes| bytes <= isize::MAX as usize)
            .ok_or(DeltaError::TooManyChanges(stored_count))?;

        // The count comes from the file: allocate as the changes arrive.
        let mut indices: Vec<u64> = Vec::with_capacity(count.min(MAX_PREALLOCATED_CHANGES));
        let mut deltas: Vec<u8> = Vec::with_capacity(delta_bytes.min(MAX_PREALLOCATED_BYTES));
        let mut change: Vec<u8> = vec![0u8; INDEX_SIZE + size];
        for _ in 0..count {
            input.read_hashed(&mut change)?;
            let index: u64 = u64_at(&change, 0);
            if index >= header.num_entries {
                return Err(DeltaError::IndexOutOfRange {
                    index,
                    num_entries: header.num_entries,
                });
            }
            if indices.last().is_some_and(|&last| index <= last) {
                return Err(DeltaError::IndexOrder { index });
            }
            if change[INDEX_SIZE..].iter().all(|&byte| byte == 0) {
                return Err(DeltaError::ZeroChange { index });
            }
            indices.push(index);
            deltas.extend_from_slice(&change[INDEX_SIZE..]);
        }

        input.finish()?;

        Ok(Self {
            header,
            indices,
            deltas,
        })
    }
}

/// Removes the last change if all of its bytes are zero.
fn drop_last_change_if_zero(indices: &mut Vec<u64>, deltas: &mut Vec<u8>, size: usize) {
    if indices.is_empty() {
        return;
    }
    let start: usize = deltas.len() - size;
    if deltas[start..].iter().all(|&byte| byte == 0) {
        deltas.truncate(start);
        indices.pop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const HEADER: DeltaHeader = DeltaHeader {
        num_entries: 0x0190,
        record_size: 5,
        from_version: 0x0203,
        to_version: 0x0207,
    };

    /// Record size 5: not a multiple of 8, and smaller than an index.
    fn sample() -> Delta {
        Delta::new(HEADER, vec![(0x0111, vec![0x22; 5]), (9, vec![0x11; 5])]).expect("delta")
    }

    fn serialized(delta: &Delta) -> Vec<u8> {
        let mut bytes: Vec<u8> = Vec::new();
        delta.write_to(&mut bytes).expect("write to a Vec");
        bytes
    }

    /// A file with the header of `HEADER` and the given changes, which need
    /// not be valid, and a correct checksum.
    fn file_with_changes(changes: &[(u64, [u8; 5])]) -> Vec<u8> {
        let mut bytes: Vec<u8> = serialized(&sample())[..HEADER_SIZE].to_vec();
        bytes[40..48].copy_from_slice(&(changes.len() as u64).to_le_bytes());
        for (index, delta) in changes {
            bytes.extend_from_slice(&index.to_le_bytes());
            bytes.extend_from_slice(delta);
        }
        with_checksum(bytes)
    }

    fn with_checksum(mut body: Vec<u8>) -> Vec<u8> {
        let checksum: blake3::Hash = blake3::hash(&body);
        body.extend_from_slice(checksum.as_bytes());
        body
    }

    #[test]
    fn layout_matches_the_documented_offsets() {
        let delta: Delta = sample();
        let bytes: Vec<u8> = serialized(&delta);

        assert_eq!(bytes.len(), 48 + 2 * 13 + 32);
        assert_eq!(bytes.len() as u64, delta.serialized_len());
        assert_eq!(&bytes[0..8], b"PLNKDLTA");
        assert_eq!(bytes[8..12], [1, 0, 0, 0]);
        assert_eq!(bytes[12..16], [5, 0, 0, 0]);
        assert_eq!(bytes[16..24], [0x90, 0x01, 0, 0, 0, 0, 0, 0]);
        assert_eq!(bytes[24..32], [0x03, 0x02, 0, 0, 0, 0, 0, 0]);
        assert_eq!(bytes[32..40], [0x07, 0x02, 0, 0, 0, 0, 0, 0]);
        assert_eq!(bytes[40..48], [2, 0, 0, 0, 0, 0, 0, 0]);
        // The changes are in index order, not in the order of the input.
        assert_eq!(bytes[48..56], [9, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(bytes[56..61], [0x11; 5]);
        assert_eq!(bytes[61..69], [0x11, 0x01, 0, 0, 0, 0, 0, 0]);
        assert_eq!(bytes[69..74], [0x22; 5]);
        assert_eq!(bytes[74..], *blake3::hash(&bytes[..74]).as_bytes());
    }

    #[test]
    fn new_sorts_combines_and_removes_changes() {
        let delta: Delta = Delta::new(
            HEADER,
            vec![
                (5, vec![0x0F, 0, 0, 0, 1]),
                (2, vec![7; 5]),
                // Index 7 gets its old value back: no change.
                (7, vec![0xAA; 5]),
                (5, vec![0xF0, 0, 0, 0, 1]),
                (7, vec![0xAA; 5]),
                (0, vec![0; 5]),
                (5, vec![0, 0, 3, 0, 0]),
            ],
        )
        .expect("delta");

        let changes: Vec<(u64, Vec<u8>)> = delta
            .changes()
            .map(|(index, bytes)| (index, bytes.to_vec()))
            .collect();
        assert_eq!(
            changes,
            [(2, vec![7; 5]), (5, vec![0xFF, 0, 3, 0, 0])],
            "one change for each index, in index order, and no zero change"
        );
        assert_eq!(delta.len(), 2);
        assert!(!delta.is_empty());
        assert_eq!(delta.header(), HEADER);

        // A zero change at the highest index is removed too.
        let delta: Delta =
            Delta::new(HEADER, vec![(3, vec![1; 5]), (8, vec![0; 5])]).expect("delta");
        assert_eq!(
            delta
                .changes()
                .map(|(index, _)| index)
                .collect::<Vec<u64>>(),
            [3]
        );
    }

    #[test]
    fn new_rejects_invalid_input() {
        assert!(matches!(
            Delta::new(HEADER, vec![(HEADER.num_entries, vec![1; 5])]),
            Err(DeltaError::IndexOutOfRange { index, num_entries })
                if index == HEADER.num_entries && num_entries == HEADER.num_entries
        ));
        assert!(Delta::new(HEADER, vec![(HEADER.num_entries - 1, vec![1; 5])]).is_ok());
        assert!(matches!(
            Delta::new(HEADER, vec![(1, vec![1; 5]), (2, vec![1; 4])]),
            Err(DeltaError::RecordSizeMismatch {
                expected: 5,
                got: 4
            })
        ));
        for to_version in [HEADER.from_version, HEADER.from_version - 1, 0] {
            assert!(matches!(
                Delta::new(
                    DeltaHeader {
                        to_version,
                        ..HEADER
                    },
                    Vec::new()
                ),
                Err(DeltaError::InvalidVersions { .. })
            ));
        }
        for record_size in [0u32, MAX_RECORD_SIZE as u32 + 1] {
            assert!(matches!(
                Delta::new(DeltaHeader { record_size, ..HEADER }, Vec::new()),
                Err(DeltaError::InvalidRecordSize(found)) if found == record_size
            ));
        }
    }

    #[test]
    fn roundtrip_preserves_the_delta() {
        let delta: Delta = sample();
        assert_eq!(
            Delta::read_from(serialized(&delta).as_slice()).expect("read"),
            delta
        );

        let empty: Delta = Delta::new(HEADER, Vec::new()).expect("empty delta");
        assert!(empty.is_empty());
        let bytes: Vec<u8> = serialized(&empty);
        assert_eq!(bytes.len(), 48 + 32);
        assert_eq!(Delta::read_from(bytes.as_slice()).expect("read"), empty);

        for record_size in [1u32, 8, 40, 256] {
            let header = DeltaHeader {
                record_size,
                ..HEADER
            };
            let changes: Vec<(u64, Vec<u8>)> = (0..7u64)
                .map(|k| {
                    let bytes: Vec<u8> = (0..record_size)
                        .map(|i| (i as u8).wrapping_mul(13) ^ (k as u8 + 1))
                        .collect();
                    (k * 31, bytes)
                })
                .collect();
            let wide: Delta = Delta::new(header, changes).expect("delta");
            assert_eq!(wide.len(), 7);
            let bytes: Vec<u8> = serialized(&wide);
            assert_eq!(bytes.len() as u64, wide.serialized_len());
            assert_eq!(
                Delta::read_from(bytes.as_slice()).expect("read"),
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
                Delta::read_from(corrupted.as_slice()).is_err(),
                "a flipped bit in byte {position} was accepted"
            );
        }
    }

    #[test]
    fn corruption_of_a_change_is_a_checksum_mismatch() {
        let mut bytes: Vec<u8> = serialized(&sample());
        bytes[58] ^= 0x80;
        assert!(matches!(
            Delta::read_from(bytes.as_slice()),
            Err(DeltaError::ChecksumMismatch)
        ));
    }

    #[test]
    fn every_truncation_is_detected() {
        let bytes: Vec<u8> = serialized(&sample());
        for len in 0..bytes.len() {
            let result = Delta::read_from(&bytes[..len]);
            assert!(
                matches!(&result, Err(DeltaError::Io(error))
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
            Delta::read_from(bytes.as_slice()),
            Err(DeltaError::TrailingData)
        ));
    }

    #[test]
    fn header_fields_are_validated() {
        let bytes: Vec<u8> = serialized(&sample());

        let mut bad_magic: Vec<u8> = bytes.clone();
        bad_magic[7] = b'B';
        assert!(matches!(
            Delta::read_from(bad_magic.as_slice()),
            Err(DeltaError::BadMagic)
        ));

        for version in [0u8, 2] {
            let mut bad_version: Vec<u8> = bytes.clone();
            bad_version[8] = version;
            assert!(matches!(
                Delta::read_from(bad_version.as_slice()),
                Err(DeltaError::UnsupportedVersion(found)) if found == u32::from(version)
            ));
        }

        for record_size in [0u32, MAX_RECORD_SIZE as u32 + 1, u32::MAX] {
            let mut bad_record_size: Vec<u8> = bytes.clone();
            bad_record_size[12..16].copy_from_slice(&record_size.to_le_bytes());
            assert!(matches!(
                Delta::read_from(bad_record_size.as_slice()),
                Err(DeltaError::InvalidRecordSize(found)) if found == record_size
            ));
        }

        // The version after the changes must be above the version before them.
        for to_version in [HEADER.from_version, HEADER.from_version - 1] {
            let mut bad_versions: Vec<u8> = bytes.clone();
            bad_versions[32..40].copy_from_slice(&to_version.to_le_bytes());
            assert!(matches!(
                Delta::read_from(bad_versions.as_slice()),
                Err(DeltaError::InvalidVersions { from_version, to_version: found })
                    if from_version == HEADER.from_version && found == to_version
            ));
        }
    }

    /// The reader must reject a file with a correct checksum whose changes
    /// are not in the form that `Delta::new` gives.
    #[test]
    fn indices_of_a_file_are_validated() {
        let ordered: Vec<u8> = file_with_changes(&[(3, [1; 5]), (4, [2; 5])]);
        assert_eq!(Delta::read_from(ordered.as_slice()).expect("read").len(), 2);

        for second in [3u64, 2] {
            let unordered: Vec<u8> = file_with_changes(&[(3, [1; 5]), (second, [2; 5])]);
            assert!(
                matches!(
                    Delta::read_from(unordered.as_slice()),
                    Err(DeltaError::IndexOrder { index }) if index == second
                ),
                "index {second} after index 3"
            );
        }

        // `Delta::new` removes a change of zero bytes; a file must not hold one.
        let zero: Vec<u8> = file_with_changes(&[(3, [1; 5]), (4, [0; 5])]);
        assert!(matches!(
            Delta::read_from(zero.as_slice()),
            Err(DeltaError::ZeroChange { index: 4 })
        ));
        let one_bit: Vec<u8> = file_with_changes(&[(3, [1; 5]), (4, [0, 0, 0, 0, 0x80])]);
        assert!(Delta::read_from(one_bit.as_slice()).is_ok());

        let out_of_range: Vec<u8> = file_with_changes(&[(HEADER.num_entries, [1; 5])]);
        assert!(matches!(
            Delta::read_from(out_of_range.as_slice()),
            Err(DeltaError::IndexOutOfRange { index, .. }) if index == HEADER.num_entries
        ));
        let last: Vec<u8> = file_with_changes(&[(HEADER.num_entries - 1, [1; 5])]);
        assert!(Delta::read_from(last.as_slice()).is_ok());
    }

    #[test]
    fn a_huge_change_count_fails_without_a_huge_allocation() {
        // More changes than the database has records.
        let mut bytes: Vec<u8> = serialized(&sample());
        bytes[40..48].copy_from_slice(&(HEADER.num_entries + 1).to_le_bytes());
        assert!(matches!(
            Delta::read_from(bytes.as_slice()),
            Err(DeltaError::TooManyChanges(count)) if count == HEADER.num_entries + 1
        ));

        // A count that the header allows, and no change after the header: the
        // reader stops at the end of the data.
        let mut bytes: Vec<u8> = serialized(&sample())[..HEADER_SIZE].to_vec();
        bytes[16..24].copy_from_slice(&(1u64 << 41).to_le_bytes());
        bytes[40..48].copy_from_slice(&(1u64 << 40).to_le_bytes());
        assert!(matches!(
            Delta::read_from(bytes.as_slice()),
            Err(DeltaError::Io(error)) if error.kind() == io::ErrorKind::UnexpectedEof
        ));
    }
}
