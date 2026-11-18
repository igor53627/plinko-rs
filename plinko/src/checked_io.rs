//! Checksummed reading and writing for the file formats of this crate
//! ([`crate::hints`], [`crate::delta`]).
//!
//! A file ends with the BLAKE3 hash of all bytes before it. The hash detects
//! damage. It does not authenticate the file: whoever changes the content can
//! write a new hash.

use std::io::{self, Read, Write};

/// Bytes of the hash at the end of a file.
pub(crate) const CHECKSUM_SIZE: usize = 32;

/// Writer that hashes the bytes that it writes.
pub(crate) struct HashingWriter<W: Write> {
    inner: W,
    hasher: blake3::Hasher,
}

impl<W: Write> HashingWriter<W> {
    pub(crate) fn new(inner: W) -> Self {
        Self {
            inner,
            hasher: blake3::Hasher::new(),
        }
    }

    pub(crate) fn write_hashed(&mut self, bytes: &[u8]) -> io::Result<()> {
        self.hasher.update(bytes);
        self.inner.write_all(bytes)
    }

    /// Writes the hash of the bytes so far and flushes the writer.
    pub(crate) fn finish(mut self) -> io::Result<()> {
        let checksum: blake3::Hash = self.hasher.finalize();
        self.inner.write_all(checksum.as_bytes())?;
        self.inner.flush()
    }
}

/// Reader that hashes the bytes that it reads.
pub(crate) struct HashingReader<R: Read> {
    inner: R,
    hasher: blake3::Hasher,
}

/// Error at the end of a checksummed file.
pub(crate) enum TrailerError {
    Io(io::Error),
    /// The stored hash does not match the content.
    ChecksumMismatch,
    /// Bytes follow the hash.
    TrailingData,
}

impl<R: Read> HashingReader<R> {
    pub(crate) fn new(inner: R) -> Self {
        Self {
            inner,
            hasher: blake3::Hasher::new(),
        }
    }

    pub(crate) fn read_hashed(&mut self, bytes: &mut [u8]) -> io::Result<()> {
        self.inner.read_exact(bytes)?;
        self.hasher.update(bytes);
        Ok(())
    }

    /// Reads the stored hash, compares it with the hash of the bytes so far,
    /// and checks that the data ends there.
    pub(crate) fn finish(mut self) -> Result<(), TrailerError> {
        let mut stored = [0u8; CHECKSUM_SIZE];
        self.inner
            .read_exact(&mut stored)
            .map_err(TrailerError::Io)?;
        if self.hasher.finalize() != stored {
            return Err(TrailerError::ChecksumMismatch);
        }

        let mut extra = [0u8; 1];
        loop {
            match self.inner.read(&mut extra) {
                Ok(0) => return Ok(()),
                Ok(_) => return Err(TrailerError::TrailingData),
                Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
                Err(error) => return Err(TrailerError::Io(error)),
            }
        }
    }
}

pub(crate) fn u32_at(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(bytes[offset..offset + 4].try_into().expect("4 bytes"))
}

pub(crate) fn u64_at(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(bytes[offset..offset + 8].try_into().expect("8 bytes"))
}
