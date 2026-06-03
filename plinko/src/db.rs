use crate::schema40::ENTRY_SIZE as ENTRY_SIZE_V3;
use eyre::{ensure, Result};
use memmap2::MmapMut;
use std::fs::OpenOptions;
use std::path::Path;

/// Database entry size for v3 schema (40 bytes).
pub const DB_ENTRY_SIZE_V3: usize = ENTRY_SIZE_V3;
pub const DB_ENTRY_U64_COUNT_V3: usize = 5; // 40 bytes = 5 * u64

/// 40-byte entry database (v3 schema).
pub struct Database40 {
    pub mmap: MmapMut,
    pub num_entries: u64,
    pub chunk_size: u64,
    pub set_size: u64,
}

impl Database40 {
    /// Loads a 40-byte entry database file into memory.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let file = OpenOptions::new().read(true).write(true).open(path)?;
        let len = file.metadata()?.len();

        ensure!(
            len % DB_ENTRY_SIZE_V3 as u64 == 0,
            "Database size {} is not a multiple of {} (v3 schema)",
            len,
            DB_ENTRY_SIZE_V3
        );

        let num_entries = len / DB_ENTRY_SIZE_V3 as u64;
        let (chunk_size, set_size) = derive_plinko_params(num_entries);

        let mmap = unsafe { MmapMut::map_mut(&file)? };

        Ok(Self {
            mmap,
            num_entries,
            chunk_size,
            set_size,
        })
    }

    /// Returns a 40-byte slice for the database entry at the given index.
    pub fn get(&self, index: u64) -> Option<&[u8]> {
        let idx = index as usize * DB_ENTRY_SIZE_V3;
        if idx + DB_ENTRY_SIZE_V3 > self.mmap.len() {
            return None;
        }
        Some(&self.mmap[idx..idx + DB_ENTRY_SIZE_V3])
    }

    /// Returns a mutable 40-byte slice for the database entry at the given index.
    pub fn get_mut(&mut self, index: u64) -> Option<&mut [u8]> {
        let idx = index as usize * DB_ENTRY_SIZE_V3;
        if idx + DB_ENTRY_SIZE_V3 > self.mmap.len() {
            return None;
        }
        Some(&mut self.mmap[idx..idx + DB_ENTRY_SIZE_V3])
    }

    /// Overwrites the database entry at the given index with raw bytes.
    pub fn update(&mut self, index: u64, new_val: &[u8; DB_ENTRY_SIZE_V3]) {
        let idx = index as usize * DB_ENTRY_SIZE_V3;
        if idx + DB_ENTRY_SIZE_V3 <= self.mmap.len() {
            self.mmap[idx..idx + DB_ENTRY_SIZE_V3].copy_from_slice(new_val);
        }
    }

    /// Overwrites the database entry with five u64 values in little-endian order.
    pub fn update_u64s(&mut self, index: u64, new_val: [u64; DB_ENTRY_U64_COUNT_V3]) {
        let idx = index as usize * DB_ENTRY_SIZE_V3;
        if idx + DB_ENTRY_SIZE_V3 <= self.mmap.len() {
            for (i, val) in new_val.iter().enumerate() {
                let bytes = val.to_le_bytes();
                self.mmap[idx + i * 8..idx + (i + 1) * 8].copy_from_slice(&bytes);
            }
        }
    }

    /// Flushes in-memory changes to the database file.
    pub fn flush(&self) -> Result<()> {
        self.mmap.flush()?;
        Ok(())
    }

    /// Returns the entry size in bytes (40).
    pub const fn entry_size(&self) -> usize {
        DB_ENTRY_SIZE_V3
    }
}

/// Compute chunk and set sizes for partitioning a database of entries.
///
/// Matches the Coq spec in `plinko/formal/specs/DbSpec.v` (integer arithmetic).
pub fn derive_plinko_params(db_entries: u64) -> (u64, u64) {
    if db_entries == 0 {
        return (1, 1);
    }
    let target_chunk = isqrt(4u64.saturating_mul(db_entries));

    let mut chunk_size = 1u64;
    while chunk_size < target_chunk {
        chunk_size = chunk_size.saturating_mul(2);
    }

    let mut set_size = db_entries.div_ceil(chunk_size);
    set_size = set_size.div_ceil(4) * 4;
    (chunk_size, set_size)
}

/// Integer square root: returns floor(sqrt(n))
fn isqrt(n: u64) -> u64 {
    if n <= 1 {
        return n;
    }
    let bits = 64 - n.leading_zeros();
    let shift = bits.div_ceil(2);
    let mut x = 1u64 << shift;

    loop {
        let y = (x + n / x) / 2;
        if y >= x {
            return x;
        }
        x = y;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn derive_plinko_params_zero_entries() {
        let (chunk, set) = derive_plinko_params(0);
        assert_eq!((chunk, set), (1, 1));
    }

    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 512,
            .. ProptestConfig::default()
        })]

        #[test]
        fn derive_plinko_params_invariants(db_entries in 1u64..10_000_000_000u64) {
            let (chunk_size, set_size) = derive_plinko_params(db_entries);

            prop_assert!(chunk_size.is_power_of_two());
            prop_assert!(chunk_size >= 1);

            let target = isqrt(4u64.saturating_mul(db_entries));
            prop_assert!(chunk_size >= target);

            let half_chunk = chunk_size / 2;
            if half_chunk > 0 && target > 0 {
                prop_assert!(half_chunk < target);
            }

            prop_assert!(set_size > 0);
            prop_assert_eq!(set_size % 4, 0);

            let base_sets = db_entries.div_ceil(chunk_size);
            prop_assert!(set_size >= base_sets);
            prop_assert!(set_size <= base_sets + 3);

            let capacity = chunk_size.checked_mul(set_size).expect("overflow in capacity");
            prop_assert!(capacity >= db_entries);
        }
    }

    #[test]
    fn derive_plinko_params_edge_cases() {
        for exp in [0, 1, 2, 10, 20, 30] {
            let n = 1u64 << exp;
            let (chunk, set) = derive_plinko_params(n);
            assert!(chunk.is_power_of_two());
            assert!(set % 4 == 0);
            assert!(chunk * set >= n);
        }

        for base in [15u64, 16, 17, 1023, 1024, 1025, 65535, 65536, 65537] {
            let (chunk, set) = derive_plinko_params(base);
            let target = isqrt(4 * base);
            assert!(chunk.is_power_of_two());
            assert!(chunk >= target);
            assert!(chunk * set >= base);
        }

        for root in [1u64, 2, 10, 100, 1000, 10000] {
            let n = root * root;
            let (chunk, set) = derive_plinko_params(n);
            assert!(chunk.is_power_of_two());
            assert!(chunk * set >= n);
        }
    }

    #[test]
    fn test_isqrt() {
        assert_eq!(isqrt(0), 0);
        assert_eq!(isqrt(1), 1);
        assert_eq!(isqrt(2), 1);
        assert_eq!(isqrt(3), 1);
        assert_eq!(isqrt(4), 2);
        assert_eq!(isqrt(9), 3);
        assert_eq!(isqrt(16), 4);
        assert_eq!(isqrt(17), 4);
        assert_eq!(isqrt(24), 4);
        assert_eq!(isqrt(25), 5);
        assert_eq!(isqrt(100), 10);
        assert_eq!(isqrt(u64::MAX), 4294967295);
    }

    proptest! {
        #![proptest_config(ProptestConfig { cases: 1000, .. ProptestConfig::default() })]

        #[test]
        fn isqrt_property(n in 0u64..1_000_000_000_000u64) {
            let s = isqrt(n);
            prop_assert!(s.saturating_mul(s) <= n);
            let s_plus_1_sq = (s + 1).saturating_mul(s + 1);
            prop_assert!(n < s_plus_1_sq || s_plus_1_sq == u64::MAX);
        }
    }

    #[test]
    fn derive_plinko_params_matches_f64_reference() {
        fn derive_f64(db_entries: u64) -> (u64, u64) {
            if db_entries == 0 {
                return (1, 1);
            }
            let target_chunk = (2.0 * (db_entries as f64).sqrt()) as u64;
            let mut chunk_size = 1u64;
            while chunk_size < target_chunk {
                chunk_size *= 2;
            }
            let mut set_size = (db_entries as f64 / chunk_size as f64).ceil() as u64;
            set_size = set_size.div_ceil(4) * 4;
            (chunk_size, set_size)
        }

        for n in 1..10000u64 {
            let (chunk_int, set_int) = derive_plinko_params(n);
            let (chunk_f64, set_f64) = derive_f64(n);
            assert_eq!(
                (chunk_int, set_int),
                (chunk_f64, set_f64),
                "Mismatch at n={}: int=({}, {}), f64=({}, {})",
                n,
                chunk_int,
                set_int,
                chunk_f64,
                set_f64
            );
        }
    }
}
