use crate::schema40::ENTRY_SIZE as ENTRY_SIZE_V3;
#[cfg(feature = "cli")]
use eyre::{ensure, Result};
#[cfg(feature = "cli")]
use memmap2::MmapMut;
#[cfg(feature = "cli")]
use std::fs::OpenOptions;
#[cfg(feature = "cli")]
use std::path::Path;

/// Database entry size for v3 schema (40 bytes).
pub const DB_ENTRY_SIZE_V3: usize = ENTRY_SIZE_V3;
pub const DB_ENTRY_U64_COUNT_V3: usize = 5; // 40 bytes = 5 * u64

/// 40-byte entry database (v3 schema), backed by a memory-mapped file.
///
/// Requires the `cli` feature.
#[cfg(feature = "cli")]
pub struct Database40 {
    pub mmap: MmapMut,
    pub num_entries: u64,
    pub chunk_size: u64,
    pub set_size: u64,
}

#[cfg(feature = "cli")]
impl Database40 {
    /// Loads a 40-byte entry database file into memory.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let file = OpenOptions::new().read(true).write(true).open(path)?;
        let len = file.metadata()?.len();
        ensure!(len > 0, "Database file is empty");

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

/// Compute chunk (`w`) and set (`c`) sizes for partitioning a database of
/// entries.
///
/// Uses [`crate::protocol::default_block_size`] for `w` and rounds `c` up to
/// an even integer of at least 2, matching [`crate::protocol::ProtocolParams::new`].
/// Returns `(1, 1)` when `db_entries == 0`.
pub fn derive_plinko_params(db_entries: u64) -> (u64, u64) {
    if db_entries == 0 {
        return (1, 1);
    }
    let chunk_size = match usize::try_from(db_entries) {
        Ok(n) => crate::protocol::default_block_size(n) as u64,
        Err(_) => {
            let raw = (db_entries as f64).sqrt().ceil() as u64;
            raw.max(1).next_power_of_two()
        }
    };

    let mut set_size = db_entries.div_ceil(chunk_size).max(2);
    if set_size % 2 != 0 {
        set_size = set_size.saturating_add(1);
    }
    (chunk_size, set_size)
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    #[cfg(feature = "cli")]
    use std::io::Write;

    #[cfg(feature = "cli")]
    #[test]
    fn load_rejects_empty_database_file() {
        let path =
            std::env::temp_dir().join(format!("plinko_empty_db_test_{}.db", std::process::id()));
        std::fs::File::create(&path).unwrap();
        let result = Database40::load(&path);
        std::fs::remove_file(&path).ok();
        match result {
            Err(err) => assert!(err.to_string().contains("Database file is empty")),
            Ok(_) => panic!("expected empty database load to fail"),
        }
    }

    #[cfg(feature = "cli")]
    #[test]
    fn load_rejects_non_multiple_size() {
        let path =
            std::env::temp_dir().join(format!("plinko_bad_db_test_{}.db", std::process::id()));
        let mut file = std::fs::File::create(&path).unwrap();
        file.write_all(&[0u8; 39]).unwrap();
        let result = Database40::load(&path);
        std::fs::remove_file(&path).ok();
        match result {
            Err(err) => assert!(err.to_string().contains("not a multiple of")),
            Ok(_) => panic!("expected invalid-size database load to fail"),
        }
    }

    #[test]
    fn derive_plinko_params_zero_entries() {
        let (chunk, set) = derive_plinko_params(0);
        assert_eq!((chunk, set), (1, 1));
    }

    #[test]
    fn derive_plinko_params_matches_protocol_params_and_mainnet() {
        use crate::protocol::{default_block_size, ProtocolParams};

        for n in [
            1usize,
            2,
            15,
            16,
            17,
            1000,
            1024,
            1025,
            65_536,
            1_834_095_877,
        ] {
            let (w, c) = derive_plinko_params(n as u64);
            assert_eq!(w, default_block_size(n) as u64);
            let params = ProtocolParams::new(n, DB_ENTRY_SIZE_V3, None, 1, None).unwrap();
            assert_eq!((w, c), (params.block_size as u64, params.num_blocks as u64));
        }
        assert_eq!(derive_plinko_params(1_834_095_877), (65_536, 27_988));
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

            if let Ok(n) = usize::try_from(db_entries) {
                prop_assert_eq!(chunk_size, crate::protocol::default_block_size(n) as u64);
            }

            prop_assert!(set_size >= 2);
            prop_assert_eq!(set_size % 2, 0);

            let base_sets = db_entries.div_ceil(chunk_size).max(2);
            prop_assert!(set_size >= base_sets);
            prop_assert!(set_size <= base_sets + 1);

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
            assert!(set >= 2 && set % 2 == 0);
            assert!(chunk * set >= n);
        }

        for base in [15u64, 16, 17, 1023, 1024, 1025, 65535, 65536, 65537] {
            let (chunk, set) = derive_plinko_params(base);
            assert!(chunk.is_power_of_two());
            assert!(set >= 2 && set % 2 == 0);
            assert!(chunk * set >= base);
        }

        for root in [1u64, 2, 10, 100, 1000, 10000] {
            let n = root * root;
            let (chunk, set) = derive_plinko_params(n);
            assert!(chunk.is_power_of_two());
            assert!(chunk >= root);
            assert!(chunk * set >= n);
        }
    }
}
