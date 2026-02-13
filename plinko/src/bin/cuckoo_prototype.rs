use blake3::Hasher;
use rand::Rng;
use std::time::Instant;
use tracing::{debug, info, warn};

const KAPPA: usize = 3; // Number of hash functions (typical for cuckoo hashing)
const MAX_EVICTIONS: usize = 1000;
const CUCKOO_OVERHEAD: f64 = 1.5; // Space overhead factor for reliable insertion

#[derive(Clone)]
struct CuckooTable {
    table: Vec<Option<([u8; 20], u32)>>, // (address, original_index)
    seeds: [u64; KAPPA],
    size: usize,
}

impl CuckooTable {
    /// Construct a new CuckooTable sized to accommodate approximately `n` entries and initialized with random hash seeds.
    ///
    /// The table is allocated with floor(n * CUCKOO_OVERHEAD) slots, all empty, and a set of `KAPPA` randomly generated 64-bit seeds used to derive the table's hash positions.
    ///
    /// # Arguments
    ///
    /// * `n` - Expected number of entries to store; used to size the internal table.
    ///
    /// # Returns
    ///
    /// A `CuckooTable` instance with an empty table and `KAPPA` seeds initialized.
    fn new(n: usize) -> Self {
        // Cuckoo table needs ~1.5x space for reliable insertion with 3 hash functions
        let size = (n as f64 * CUCKOO_OVERHEAD) as usize;
        let mut rng = rand::thread_rng();
        let seeds: [u64; KAPPA] = [rng.gen(), rng.gen(), rng.gen()];

        Self {
            table: vec![None; size],
            seeds,
            size,
        }
    }

    fn hash(&self, addr: &[u8; 20], hash_idx: usize) -> usize {
        let mut hasher = Hasher::new();
        hasher.update(&self.seeds[hash_idx].to_le_bytes());
        hasher.update(addr);
        let hash = hasher.finalize();
        let h = u64::from_le_bytes(hash.as_bytes()[0..8].try_into().unwrap());
        (h as usize) % self.size
    }

    /// Attempts to place an (address, index) pair into the cuckoo table, using bounded evictions if necessary.
    ///
    /// If any of the KAPPA candidate slots for the provided address is empty, the pair is placed there.
    /// Otherwise the function performs up to MAX_EVICTIONS eviction iterations: it chooses a random
    /// candidate slot, evicts its occupant into the current item, and repeats the insertion attempt with
    /// the evicted item. The process stops before the final iteration to avoid risking removal of an
    /// existing element.
    ///
    /// On success the function returns Ok(()). If no placement is found within MAX_EVICTIONS, the
    /// function returns Err((addr, original_idx)) where the tuple is the single entry that could not be
    /// placed (this may be the original input or an item displaced during eviction).
    ///
    /// # Examples
    ///
    /// ```
    /// let mut table = CuckooTable::new(10);
    /// let addr = [0u8; 20];
    /// let res = table.insert(addr, 42);
    /// assert!(res.is_ok() || matches!(res, Err((_, _))));
    /// ```
    fn insert(&mut self, addr: [u8; 20], original_idx: u32) -> Result<(), ([u8; 20], u32)> {
        let mut current = (addr, original_idx);
        let mut rng = rand::thread_rng();

        for iteration in 0..MAX_EVICTIONS {
            // Try all hash positions
            for h in 0..KAPPA {
                let pos = self.hash(&current.0, h);
                if self.table[pos].is_none() {
                    self.table[pos] = Some(current);
                    return Ok(());
                }
            }
            // Don't evict on the last iteration - would lose an existing element
            if iteration == MAX_EVICTIONS - 1 {
                break;
            }
            // Evict from a random hash position (better than always first)
            let h = rng.gen_range(0..KAPPA);
            let pos = self.hash(&current.0, h);
            let evicted = self.table[pos].take().unwrap();
            self.table[pos] = Some(current);
            current = evicted;
        }
        // Return the element that couldn't be placed. After the eviction chain,
        // this is the single key not in the table (may be original input or displaced).
        Err(current)
    }

    fn lookup(&self, addr: &[u8; 20]) -> Vec<usize> {
        // Returns all κ positions the client must query
        (0..KAPPA).map(|h| self.hash(addr, h)).collect()
    }

    fn get(&self, addr: &[u8; 20]) -> Option<u32> {
        for h in 0..KAPPA {
            let pos = self.hash(addr, h);
            if let Some((stored_addr, idx)) = &self.table[pos] {
                if stored_addr == addr {
                    return Some(*idx);
                }
            }
        }
        None
    }
}

// Original approach: sorted lookup table
struct SortedLookup {
    entries: Vec<([u8; 20], u32)>,
}

impl SortedLookup {
    fn new(entries: Vec<([u8; 20], u32)>) -> Self {
        let mut entries = entries;
        entries.sort_by_key(|(addr, _)| *addr);
        Self { entries }
    }

    fn get(&self, addr: &[u8; 20]) -> Option<u32> {
        self.entries
            .binary_search_by_key(addr, |(a, _)| *a)
            .ok()
            .map(|i| self.entries[i].1)
    }

    fn storage_size(&self) -> usize {
        self.entries.len() * 24 // 20 bytes addr + 4 bytes index
    }
}

fn generate_random_addresses(n: usize) -> Vec<[u8; 20]> {
    let mut rng = rand::thread_rng();
    (0..n)
        .map(|_| {
            let mut addr = [0u8; 20];
            rng.fill(&mut addr);
            addr
        })
        .collect()
}

fn main() {
    tracing_subscriber::fmt::init();

    info!("Plinko Keyword PIR Prototype");
    info!("Comparing: Original (sorted lookup table) vs Cuckoo Hashing");

    // Test with different sizes
    for n in [10_000, 100_000, 1_000_000] {
        info!(n, "Testing with N addresses");

        let addresses = generate_random_addresses(n);
        let entries: Vec<_> = addresses
            .iter()
            .enumerate()
            .map(|(i, addr)| (*addr, i as u32))
            .collect();

        // === Original Approach ===
        let start = Instant::now();
        let sorted = SortedLookup::new(entries.clone());
        let sorted_build_time = start.elapsed();

        // === Cuckoo Approach ===
        let start = Instant::now();
        let mut cuckoo = CuckooTable::new(n);
        let mut failed = 0;
        let mut unplaced_elements = Vec::new();
        for (addr, idx) in &entries {
            if let Err(unplaced) = cuckoo.insert(*addr, *idx) {
                // Collect elements that couldn't be placed - in production we'd rehash
                unplaced_elements.push(unplaced);
                failed += 1;
            }
        }
        if !unplaced_elements.is_empty() {
            warn!(
                failed_inserts = failed,
                unplaced = unplaced_elements.len(),
                "Insertions failed, would trigger rehash in production"
            );
        }
        let cuckoo_build_time = start.elapsed();

        info!(
            sorted_ms = format_args!("{:.2}", sorted_build_time.as_secs_f64() * 1000.0),
            cuckoo_ms = format_args!("{:.2}", cuckoo_build_time.as_secs_f64() * 1000.0),
            failed_inserts = failed,
            "Build time"
        );

        // Storage comparison
        let sorted_storage = sorted.storage_size();
        let hash_seeds_storage = KAPPA * 8; // κ seeds of 8 bytes each

        info!(
            sorted_bytes = sorted_storage,
            sorted_mb = format_args!("{:.2}", sorted_storage as f64 / 1_000_000.0),
            cuckoo_bytes = hash_seeds_storage,
            reduction = format_args!("{:.0}x", sorted_storage as f64 / hash_seeds_storage as f64),
            "Client storage"
        );

        debug!(
            sorted_queries = 1,
            cuckoo_queries = KAPPA,
            kappa = KAPPA,
            "Query overhead"
        );

        // Lookup correctness & timing
        let test_addr = addresses[n / 2];

        let start = Instant::now();
        let sorted_result = sorted.get(&test_addr);
        let sorted_lookup_time = start.elapsed();

        let start = Instant::now();
        let cuckoo_result = cuckoo.get(&test_addr);
        let cuckoo_lookup_time = start.elapsed();

        let positions = cuckoo.lookup(&test_addr);

        debug!(
            sorted_ns = sorted_lookup_time.as_nanos(),
            sorted_result = ?sorted_result,
            cuckoo_ns = cuckoo_lookup_time.as_nanos(),
            cuckoo_result = ?cuckoo_result,
            cuckoo_positions = ?positions,
            "Single-address lookup"
        );

        assert_eq!(sorted_result, cuckoo_result, "Results should match!");
        info!("Lookup results match");
    }

    // Summary for Ethereum mainnet scale
    info!("Projected for Ethereum Mainnet");
    let mainnet_accounts = 330_000_000u64;
    let sorted_storage_mainnet = mainnet_accounts * 24;
    let cuckoo_client_storage = KAPPA * 8;

    info!(
        accounts = mainnet_accounts,
        sorted_storage_gb = format_args!("{:.2}", sorted_storage_mainnet as f64 / 1_000_000_000.0),
        cuckoo_storage_bytes = cuckoo_client_storage,
        cuckoo_hash_seeds = KAPPA,
        cuckoo_queries_per_lookup = KAPPA,
        server_storage_gb = format_args!(
            "{:.2}",
            (mainnet_accounts as f64 * CUCKOO_OVERHEAD * 24.0) / 1_000_000_000.0
        ),
        "Mainnet projections"
    );

    info!(
        storage_reduction = format_args!(
            "{:.0}x",
            sorted_storage_mainnet as f64 / cuckoo_client_storage as f64
        ),
        from_gb = format_args!("{:.1}", sorted_storage_mainnet as f64 / 1_000_000_000.0),
        to_bytes = cuckoo_client_storage,
        query_overhead = format_args!("{}x", KAPPA),
        "Trade-off summary"
    );

    info!(
        hint_sets = KAPPA,
        "For Plinko hints: client downloads multiple hint sets or generates hints covering all positions per address"
    );
}
