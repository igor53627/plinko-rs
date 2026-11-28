//! State file of a client: all that a client needs to start again after it
//! sent queries.
//!
//! A hint file holds the hints of a client that has sent no query. A query
//! changes the client: the hint that it sent must not serve another index, a
//! backup hint takes its place, and the answer goes to the cache. A client
//! that loses this state and loads its hint file again uses hints a second
//! time, and the server then gets queried indices
//! (`docs/online_protocol.md`). [`Client::write_state`] writes the state and
//! [`Client::read_state`] builds a client from it.
//!
//! # Rules for the caller
//!
//! - **Record a query before you send it.** A query that the server saw and
//!   the stored state does not hold has the same effect as a hint file that
//!   is loaded again. There are two ways:
//!   - Write the state after [`Client::prepare_query`] and before the query
//!     leaves the client. The state has the size of the hint file, so this
//!     is a large write for each query.
//!   - Write the state at times, and between two writes append
//!     [`PreparedQuery::hint_index`] of each query to a log before the query
//!     leaves the client. After a restart, load the state and pass the log to
//!     [`Client::retire_hints`]. The log of a state holds each query that
//!     the client prepared after the [`Client::write_state`] call of that
//!     state. Keep the old state and its log until the new state and its
//!     log are durable.
//! - **Make the write durable** before the query leaves the client, and
//!   replace a state only with a complete one.
//! - **Without the log, load only the newest state.** The format cannot
//!   detect a state that is older than the last query of the client.
//! - **Keep the deltas that a state needs.** A state has the database
//!   version of its hints ([`Client::database_version`]); a log for
//!   [`Client::retire_hints`] does not change it. After a restart, apply
//!   the deltas from that version before you accept an answer of a later
//!   version; no delta is necessary when the versions are equal. Keep each
//!   delta, or a source for it, until a durable state has its `to_version`
//!   or a later version.
//! - **Use new random bytes as the query seed of each start.** The state does
//!   not hold the query randomness.
//! - **Keep the file secret.** It holds the hints, the indices of the
//!   answered queries with their records (the cache), and the indices of the
//!   queries in flight. It does not hold the master seed.
//!
//! # File format (version 1)
//!
//! All integers are little-endian. `r` is the record size in bytes.
//!
//! | Size | Field |
//! |------|-------|
//! | 8    | magic `PLNKSTAT` |
//! | 4    | format version (`1`) |
//! | ...  | hint set: a hint file ([`crate::hints`]) without its last 32 bytes |
//! | `ceil(num_regular / 8)` | bit `j % 8` of byte `j / 8`: regular hint `j` was sent |
//! | 8    | `answered`: number of answered queries |
//! | `17 + r` each | index (`u64`), backup hint (`u64`), flags (`u8`), record (`r` bytes) |
//! | 8    | `in_flight`: number of queries in flight |
//! | 25 each | queried index (`u64`), requested index (`u64`), hint (`u64`), flags (`u8`) |
//! | 8    | `retired`: number of backup hints that were sent with no answered query |
//! | 8 each | backup hint (`u64`) |
//! | 32   | BLAKE3 hash of all bytes before it |
//!
//! An answered query is a cache entry with the backup hint that was promoted
//! for it. Flag bit 0: the promoted hint is present (it was not sent in a
//! later query). Flag bit 1: its subset is the complement of the cutoff
//! subset; only with bit 0. The entries are in the order of their indices.
//!
//! A query in flight has the hint ID that it reserves. Flag bit 0: the real
//! subset is the first one of the message. The entries are in the order of
//! their queried indices.
//!
//! A retired backup hint comes from [`Client::retire_hints`] only. The
//! entries are in the order of their numbers.
//!
//! The parities of the hint set are those of the client: the "in" parity of
//! a promoted backup hint is the parity of its slot.

use super::{
    block_iprfs, CacheEntry, Client, HintSlot, InFlight, PreparedQuery, ProtocolError,
    ProtocolParams, CLIENT_CACHED_PMNS_LEVELS, PARALLEL_QUERY_MIN_BLOCKS,
};
use crate::checked_io::{u32_at, u64_at, HashingReader, HashingWriter, TrailerError};
use crate::hints::{self, HintFileError, HintHeader, HintSet};
use crate::iprf::{Iprf, PrfKey128, SwapOrNotSr};
use crate::subsets::{SubsetPrf, INVALID_CUTOFF};
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use std::collections::{BTreeSet, HashMap, HashSet};
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::io::{self, Read, Write};

/// First bytes of a client state file.
pub const CLIENT_STATE_MAGIC: [u8; 8] = *b"PLNKSTAT";
/// Format version that this module reads and writes.
pub const CLIENT_STATE_VERSION: u32 = 1;

const PREFIX_SIZE: usize = 12;
const ANSWERED_FIXED_SIZE: usize = 17;
const IN_FLIGHT_SIZE: usize = 25;
const FLAG_PRESENT: u8 = 1;
const FLAG_COMPLEMENTED: u8 = 2;
const FLAG_REAL_IS_FIRST: u8 = 1;

/// Error from [`Client::read_state`].
#[derive(Debug)]
pub enum ClientStateError {
    /// The reader failed, or the data ended before the end of the file format.
    Io(io::Error),
    /// The data does not start with [`CLIENT_STATE_MAGIC`].
    BadMagic,
    /// The file has a format version that this code does not read.
    UnsupportedVersion(u32),
    /// The hint set of the file is not valid.
    Hints(HintFileError),
    /// The hint set does not belong to the master seed, or its header is not
    /// the header of a database.
    Mismatch(ProtocolError),
    /// A field of the client state breaks a rule of the format.
    Invalid(&'static str),
    /// The stored hash does not match the content.
    ChecksumMismatch,
    /// Bytes follow the checksum.
    TrailingData,
}

impl Display for ClientStateError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(error) => write!(f, "cannot read client state file: {error}"),
            Self::BadMagic => write!(f, "not a Plinko client state file (bad magic)"),
            Self::UnsupportedVersion(version) => {
                write!(f, "unsupported client state file version {version}")
            }
            Self::Hints(error) => write!(f, "hint set of the client state file: {error}"),
            Self::Mismatch(error) => write!(f, "client state file: {error}"),
            Self::Invalid(what) => write!(f, "client state file is not valid: {what}"),
            Self::ChecksumMismatch => write!(f, "client state file checksum mismatch"),
            Self::TrailingData => write!(f, "client state file has data after its checksum"),
        }
    }
}

impl Error for ClientStateError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            Self::Hints(error) => Some(error),
            Self::Mismatch(error) => Some(error),
            _ => None,
        }
    }
}

impl From<io::Error> for ClientStateError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<HintFileError> for ClientStateError {
    fn from(error: HintFileError) -> Self {
        match error {
            HintFileError::Io(error) => Self::Io(error),
            other => Self::Hints(other),
        }
    }
}

impl From<TrailerError> for ClientStateError {
    fn from(error: TrailerError) -> Self {
        match error {
            TrailerError::Io(error) => Self::Io(error),
            TrailerError::ChecksumMismatch => Self::ChecksumMismatch,
            TrailerError::TrailingData => Self::TrailingData,
        }
    }
}

/// Worst-case hashbrown bucket table + control bytes (7/8 load factor, power-of-two
/// buckets, plus a 64-byte trailing SIMD control group bound) for `count` entries.
pub(crate) fn hashmap_table_bytes<K, V>(count: usize) -> Option<usize> {
    if count == 0 {
        return Some(0);
    }
    let min_buckets: usize = count.checked_mul(8)?.div_ceil(7);
    let table_buckets: usize = min_buckets.max(4).checked_next_power_of_two()?;
    table_buckets
        .checked_mul(std::mem::size_of::<(K, V)>() + 1)?
        .checked_add(64)
}

/// Reads a count and checks it against the largest value that the hint set allows.
fn read_count<R: Read>(
    input: &mut HashingReader<R>,
    max: usize,
    too_many: &'static str,
) -> Result<usize, ClientStateError> {
    let mut bytes = [0u8; 8];
    input.read_hashed(&mut bytes)?;
    usize::try_from(u64::from_le_bytes(bytes))
        .ok()
        .filter(|&count| count <= max)
        .ok_or(ClientStateError::Invalid(too_many))
}

/// Returns the value at `offset` if it is below `bound`.
fn index_below(bytes: &[u8], offset: usize, bound: usize) -> Option<usize> {
    usize::try_from(u64_at(bytes, offset))
        .ok()
        .filter(|&index| index < bound)
}

impl Client {
    /// Writes the state of the client in the format of
    /// `plinko::protocol::state` (see `docs/data_format.md`).
    ///
    /// The state holds the hints with the database version, the hints that
    /// the client sent, the promoted backup hints, the cache and the queries
    /// in flight. It does not hold the query randomness.
    ///
    /// Each query must be recorded before it leaves the client: in a state,
    /// or in a log for [`Client::retire_hints`]. Replace the last state only
    /// with a state that is complete and durable. The file is client-secret:
    /// it holds the queried indices, and the records of the answered queries.
    ///
    /// Pass a buffered writer; the function writes one hint at a time.
    pub fn write_state<W: Write>(&self, writer: W) -> io::Result<()> {
        let mut out = HashingWriter::new(writer);
        out.write_hashed(&CLIENT_STATE_MAGIC)?;
        out.write_hashed(&CLIENT_STATE_VERSION.to_le_bytes())?;
        self.hints.write_body(&mut out)?;

        // A regular slot is empty if its hint was sent, or if its cutoff is not valid.
        let num_regular: usize = self.params.num_regular_hints;
        let mut sent: Vec<u8> = vec![0u8; num_regular.div_ceil(8)];
        for (j, (slot, &cutoff)) in self.slots[..num_regular]
            .iter()
            .zip(self.hints.regular_cutoffs())
            .enumerate()
        {
            if slot.is_none() && cutoff != INVALID_CUTOFF {
                sent[j / 8] |= 1 << (j % 8);
            }
        }
        out.write_hashed(&sent)?;

        let mut answered: Vec<(usize, &CacheEntry)> = self
            .cache
            .iter()
            .map(|(&index, entry)| (index, entry))
            .collect();
        answered.sort_unstable_by_key(|&(index, _)| index);
        out.write_hashed(&(answered.len() as u64).to_le_bytes())?;
        for (index, entry) in answered {
            let flags: u8 = match &self.slots[entry.hint_idx] {
                Some(slot) if slot.complemented => FLAG_PRESENT | FLAG_COMPLEMENTED,
                Some(_) => FLAG_PRESENT,
                None => 0,
            };
            out.write_hashed(&(index as u64).to_le_bytes())?;
            out.write_hashed(&((entry.hint_idx - num_regular) as u64).to_le_bytes())?;
            out.write_hashed(&[flags])?;
            out.write_hashed(&entry.value)?;
        }

        let mut in_flight: Vec<(usize, InFlight)> = self
            .in_flight
            .iter()
            .map(|(&queried_index, &query)| (queried_index, query))
            .collect();
        in_flight.sort_unstable_by_key(|&(queried_index, _)| queried_index);
        out.write_hashed(&(in_flight.len() as u64).to_le_bytes())?;
        for (queried_index, query) in in_flight {
            out.write_hashed(&(queried_index as u64).to_le_bytes())?;
            out.write_hashed(&(query.requested_index as u64).to_le_bytes())?;
            out.write_hashed(&(query.hint_idx as u64).to_le_bytes())?;
            out.write_hashed(&[u8::from(query.real_is_first) * FLAG_REAL_IS_FIRST])?;
        }

        out.write_hashed(&(self.retired_backups.len() as u64).to_le_bytes())?;
        for &backup_idx in &self.retired_backups {
            out.write_hashed(&(backup_idx as u64).to_le_bytes())?;
        }

        out.finish()
    }

    /// Builds a client from a state that [`Client::write_state`] wrote.
    ///
    /// Returns the client and its queries in flight, in the order of their
    /// queried indices. Each of them is the query that the client built
    /// before: send `query` again, or call [`Client::abandon_query`].
    ///
    /// If the caller has a log for [`Client::retire_hints`], pass the log
    /// first. A returned query whose [`PreparedQuery::hint_index`] is in the
    /// log is then no longer in flight: do not send it again.
    /// [`Client::reconstruct_and_replenish`] and [`Client::abandon_query`]
    /// return [`ProtocolError::UnknownPreparedQuery`] for it.
    ///
    /// **The state must hold each query that the client sent, or the caller
    /// must pass the hints of the later queries to [`Client::retire_hints`]
    /// before the next query.** If not, the client uses a hint a second
    /// time, and the server then gets queried indices. The function cannot
    /// detect an old state.
    ///
    /// **`query_rng_seed` must be new random bytes**, not the seed of an
    /// earlier start. With an old seed the client repeats random choices of
    /// queries that the server saw.
    ///
    /// Pass a buffered reader; the function reads one hint at a time.
    ///
    /// # Errors
    /// Returns [`ClientStateError::Mismatch`] if the state belongs to another
    /// master seed, and another [`ClientStateError`] for data that is not a
    /// valid state file.
    pub fn read_state<R: Read>(
        master_seed: [u8; 32],
        query_rng_seed: [u8; 32],
        reader: R,
    ) -> Result<(Self, Vec<PreparedQuery>), ClientStateError> {
        Self::read_state_with_limit(master_seed, query_rng_seed, reader, usize::MAX)
            .map(|(client, prepared, _)| (client, prepared))
    }

    /// Reads a client state like [`Self::read_state`] while enforcing an explicit
    /// memory budget `max_alloc_bytes` across every decoded structure — including
    /// per-block [`Iprf`] and key allocations that scale with `params.num_blocks` —
    /// before allocating, and returns `(client, in_flight, retained_bytes)`.
    pub(crate) fn read_state_with_limit<R: Read>(
        master_seed: [u8; 32],
        query_rng_seed: [u8; 32],
        reader: R,
        max_alloc_bytes: usize,
    ) -> Result<(Self, Vec<PreparedQuery>, usize), ClientStateError> {
        let mut input = HashingReader::new(reader);

        let mut prefix = [0u8; PREFIX_SIZE];
        input.read_hashed(&mut prefix)?;
        if prefix[0..8] != CLIENT_STATE_MAGIC {
            return Err(ClientStateError::BadMagic);
        }
        let version: u32 = u32_at(&prefix, 8);
        if version != CLIENT_STATE_VERSION {
            return Err(ClientStateError::UnsupportedVersion(version));
        }

        let hint_header: HintHeader = HintSet::read_header(&mut input)?;
        let params: ProtocolParams = ProtocolParams::for_hint_geometry(
            hint_header.geometry,
            hint_header.num_regular,
            hint_header.num_backup,
        )
        .map_err(ClientStateError::Mismatch)?;
        if hint_header.seed_check != hints::seed_check(&master_seed) {
            return Err(ClientStateError::Mismatch(ProtocolError::HintSetMismatch(
                "master seed",
            )));
        }

        let check_budget = |bytes: usize| -> Result<(), ClientStateError> {
            if bytes > max_alloc_bytes {
                return Err(ClientStateError::Invalid(
                    "client state exceeds memory budget",
                ));
            }
            Ok(())
        };

        let num_regular: usize = params.num_regular_hints;
        let size: usize = params.record_size;

        // Account for every per-hint and per-block allocation before reading records
        // or deriving per-block keys/iPRFs.
        let reg_cutoff_bytes: usize = num_regular.checked_mul(std::mem::size_of::<u32>()).ok_or(
            ClientStateError::Invalid("client state exceeds memory budget"),
        )?;
        let bak_cutoff_bytes: usize = params
            .num_backup_hints
            .checked_mul(std::mem::size_of::<u32>())
            .ok_or(ClientStateError::Invalid(
                "client state exceeds memory budget",
            ))?;
        let slots_bytes: usize = params
            .num_total_hints
            .checked_mul(std::mem::size_of::<Option<HintSlot>>())
            .ok_or(ClientStateError::Invalid(
                "client state exceeds memory budget",
            ))?;
        let backup_avail_bytes: usize = params
            .num_backup_hints
            .checked_mul(std::mem::size_of::<bool>())
            .ok_or(ClientStateError::Invalid(
                "client state exceeds memory budget",
            ))?;
        let iprfs_bytes: usize = params
            .num_blocks
            .checked_mul(std::mem::size_of::<Iprf>())
            .ok_or(ClientStateError::Invalid(
                "client state exceeds memory budget",
            ))?;
        let block_keys_bytes: usize = params
            .num_blocks
            .checked_mul(std::mem::size_of::<PrfKey128>())
            .ok_or(ClientStateError::Invalid(
                "client state exceeds memory budget",
            ))?;
        let sent_bytes: usize = num_regular.div_ceil(8);
        let hint_record_scratch: usize =
            size.checked_mul(2)
                .and_then(|v| v.checked_add(4))
                .ok_or(ClientStateError::Invalid(
                    "client state exceeds memory budget",
                ))?;

        let mut retained_bytes: usize = reg_cutoff_bytes
            .checked_add(hint_header.regular_bytes)
            .and_then(|v| v.checked_add(bak_cutoff_bytes))
            .and_then(|v| v.checked_add(hint_header.backup_bytes))
            .and_then(|v| v.checked_add(slots_bytes))
            .and_then(|v| v.checked_add(backup_avail_bytes))
            .and_then(|v| v.checked_add(iprfs_bytes))
            .ok_or(ClientStateError::Invalid(
                "client state exceeds memory budget",
            ))?;
        let upfront_peak: usize = retained_bytes
            .checked_add(block_keys_bytes.max(sent_bytes).max(hint_record_scratch))
            .ok_or(ClientStateError::Invalid(
                "client state exceeds memory budget",
            ))?;
        check_budget(upfront_peak)?;

        let hints: HintSet = if max_alloc_bytes < usize::MAX {
            HintSet::read_records_exact(&mut input, hint_header)?
        } else {
            HintSet::read_records(&mut input, hint_header)?
        };

        // The hint set has `num_regular` hints in memory, so this allocation
        // is smaller than the data that was read.
        let mut sent: Vec<u8> = vec![0u8; sent_bytes];
        input.read_hashed(&mut sent)?;
        let mut slots: Vec<Option<HintSlot>> = Vec::with_capacity(params.num_total_hints);
        for (j, &cutoff) in hints.regular_cutoffs().iter().enumerate() {
            let was_sent: bool = (sent[j / 8] >> (j % 8)) & 1 == 1;
            if was_sent && cutoff == INVALID_CUTOFF {
                return Err(ClientStateError::Invalid(
                    "a regular hint with no valid cutoff is marked as sent",
                ));
            }
            slots.push((!was_sent && cutoff != INVALID_CUTOFF).then_some(HintSlot {
                cutoff,
                complemented: false,
                extra_index: None,
            }));
        }
        let unused_bits: usize = sent.len() * 8 - num_regular;
        if unused_bits > 0 && sent[sent.len() - 1] >> (8 - unused_bits) != 0 {
            return Err(ClientStateError::Invalid(
                "a bit after the last regular hint is set",
            ));
        }
        slots.resize(params.num_total_hints, None);
        let any_sent: bool = sent.iter().any(|&byte| byte != 0);
        drop(sent);

        let mut backup_available: Vec<bool> = hints
            .backup_cutoffs()
            .iter()
            .map(|&cutoff| cutoff != INVALID_CUTOFF)
            .collect();

        // Each answered query has its own backup hint.
        let num_answered: usize = read_count(
            &mut input,
            params.num_backup_hints,
            "more answered queries than backup hints",
        )?;
        let cache_table_bytes: usize = hashmap_table_bytes::<usize, CacheEntry>(num_answered)
            .ok_or(ClientStateError::Invalid(
                "client state exceeds memory budget",
            ))?;
        let cache_values_bytes: usize =
            num_answered
                .checked_mul(size)
                .ok_or(ClientStateError::Invalid(
                    "client state exceeds memory budget",
                ))?;
        let cache_bytes: usize =
            cache_table_bytes
                .checked_add(cache_values_bytes)
                .ok_or(ClientStateError::Invalid(
                    "client state exceeds memory budget",
                ))?;
        let answer_scratch: usize = if num_answered > 0 {
            ANSWERED_FIXED_SIZE
                .checked_add(size)
                .ok_or(ClientStateError::Invalid(
                    "client state exceeds memory budget",
                ))?
        } else {
            0
        };
        retained_bytes =
            retained_bytes
                .checked_add(cache_bytes)
                .ok_or(ClientStateError::Invalid(
                    "client state exceeds memory budget",
                ))?;
        check_budget(
            retained_bytes
                .checked_add(block_keys_bytes.max(answer_scratch))
                .ok_or(ClientStateError::Invalid(
                    "client state exceeds memory budget",
                ))?,
        )?;
        let mut cache: HashMap<usize, CacheEntry> = HashMap::new();
        cache
            .try_reserve(num_answered)
            .map_err(|_| ClientStateError::Invalid("client state exceeds memory budget"))?;
        if num_answered > 0 {
            let mut entry: Vec<u8> = vec![0u8; ANSWERED_FIXED_SIZE + size];
            let mut last_index: Option<usize> = None;
            for _ in 0..num_answered {
                input.read_hashed(&mut entry)?;
                let index: usize = index_below(&entry, 0, params.num_entries)
                    .ok_or(ClientStateError::Invalid("index of an answered query"))?;
                if last_index.is_some_and(|last| index <= last) {
                    return Err(ClientStateError::Invalid(
                        "the answered queries are not in the order of their indices",
                    ));
                }
                last_index = Some(index);

                let backup_idx: usize = index_below(&entry, 8, params.num_backup_hints).ok_or(
                    ClientStateError::Invalid("backup hint of an answered query"),
                )?;
                if !backup_available[backup_idx] {
                    return Err(ClientStateError::Invalid(
                        "a backup hint is promoted two times, or it has no valid cutoff",
                    ));
                }
                backup_available[backup_idx] = false;

                let flags: u8 = entry[16];
                let present: bool = flags & FLAG_PRESENT != 0;
                let complemented: bool = flags & FLAG_COMPLEMENTED != 0;
                if flags > (FLAG_PRESENT | FLAG_COMPLEMENTED) || (complemented && !present) {
                    return Err(ClientStateError::Invalid("flags of an answered query"));
                }
                let hint_idx: usize = num_regular + backup_idx;
                if present {
                    slots[hint_idx] = Some(HintSlot {
                        cutoff: hints.backup_cutoffs()[backup_idx],
                        complemented,
                        extra_index: Some(index),
                    });
                }
                cache.insert(
                    index,
                    CacheEntry {
                        value: entry[ANSWERED_FIXED_SIZE..].to_vec(),
                        hint_idx,
                    },
                );
            }
        }

        // Each query in flight reserves one backup hint that is not promoted.
        let free_backups: usize = backup_available.iter().filter(|&&free| free).count();
        let num_in_flight: usize = read_count(
            &mut input,
            free_backups,
            "more queries in flight than free backup hints",
        )?;
        let in_flight_retained: usize = hashmap_table_bytes::<usize, InFlight>(num_in_flight)
            .ok_or(ClientStateError::Invalid(
                "client state exceeds memory budget",
            ))?;
        let order_bytes: usize = num_in_flight
            .checked_mul(std::mem::size_of::<usize>())
            .ok_or(ClientStateError::Invalid(
                "client state exceeds memory budget",
            ))?;
        let reserved_table_bytes: usize = hashmap_table_bytes::<usize, ()>(num_in_flight).ok_or(
            ClientStateError::Invalid("client state exceeds memory budget"),
        )?;
        let per_prepared_bytes: usize = std::mem::size_of::<PreparedQuery>()
            .checked_add(params.num_blocks.div_ceil(8))
            .and_then(|v| {
                v.checked_add((params.num_blocks / 2).checked_mul(std::mem::size_of::<u32>())?)
            })
            .ok_or(ClientStateError::Invalid(
                "client state exceeds memory budget",
            ))?;
        let prepared_bytes: usize =
            num_in_flight
                .checked_mul(per_prepared_bytes)
                .ok_or(ClientStateError::Invalid(
                    "client state exceeds memory budget",
                ))?;
        let build_query_temp: usize = if num_in_flight > 0 {
            let half: usize = params.num_blocks / 2;
            // Inside `try_build_query`, live buffers during `Iprf::forward_many` are:
            // - `select_values: Vec<u32>` (`num_blocks`)
            // - `query_iprfs: Vec<&Iprf>` (`<= half`)
            // - `Iprf::forward_many` scratch across `<= half` iPRFs: `prps: Vec<&SwapOrNotSr>`,
            //   `vals: Vec<u64>`, and either `active: Vec<usize>` or the output `Vec<u64>`
            // - parallel `out: Vec<u64>` (`<= half`) on native targets when `half >= PARALLEL_QUERY_MIN_BLOCKS`
            let forward_many_per_iprf: usize = std::mem::size_of::<&SwapOrNotSr>()
                .checked_add(std::mem::size_of::<u64>())
                .and_then(|v| {
                    v.checked_add(std::mem::size_of::<usize>().max(std::mem::size_of::<u64>()))
                })
                .ok_or(ClientStateError::Invalid(
                    "client state exceeds memory budget",
                ))?;
            let parallel_out_bytes: usize =
                if !cfg!(target_arch = "wasm32") && half >= PARALLEL_QUERY_MIN_BLOCKS {
                    half.checked_mul(std::mem::size_of::<u64>()).ok_or(
                        ClientStateError::Invalid("client state exceeds memory budget"),
                    )?
                } else {
                    0
                };
            params
                .num_blocks
                .checked_mul(std::mem::size_of::<u32>())
                .and_then(|v| v.checked_add(half.checked_mul(std::mem::size_of::<&Iprf>())?))
                .and_then(|v| v.checked_add(half.checked_mul(forward_many_per_iprf)?))
                .and_then(|v| v.checked_add(parallel_out_bytes))
                .ok_or(ClientStateError::Invalid(
                    "client state exceeds memory budget",
                ))?
        } else {
            0
        };
        let pmns_cache_bytes: usize =
            if num_in_flight > 0 && (params.num_total_hints as u64) <= u64::from(u32::MAX) {
                let cached_levels: u32 =
                    CLIENT_CACHED_PMNS_LEVELS.min(params.block_size.trailing_zeros());
                let nodes_per_block: usize = (1usize << cached_levels).saturating_sub(1);
                params
                    .num_blocks
                    .checked_mul(nodes_per_block)
                    .and_then(|v| v.checked_mul(std::mem::size_of::<u32>()))
                    .ok_or(ClientStateError::Invalid(
                        "client state exceeds memory budget",
                    ))?
            } else {
                0
            };
        // Overlapping temporary peaks:
        // 1) in-flight validation loop: `order` + `reserved` (dropped after the loop)
        // 2) `block_iprfs`: `order` + `block_keys`
        // 3) query rebuild loop: `order` + `prepared` + `try_build_query` scratch
        let validation_temp: usize =
            order_bytes
                .checked_add(reserved_table_bytes)
                .ok_or(ClientStateError::Invalid(
                    "client state exceeds memory budget",
                ))?;
        let block_iprfs_temp: usize =
            order_bytes
                .checked_add(block_keys_bytes)
                .ok_or(ClientStateError::Invalid(
                    "client state exceeds memory budget",
                ))?;
        let rebuild_temp: usize = order_bytes
            .checked_add(prepared_bytes)
            .and_then(|v| v.checked_add(build_query_temp))
            .ok_or(ClientStateError::Invalid(
                "client state exceeds memory budget",
            ))?;
        retained_bytes = retained_bytes
            .checked_add(in_flight_retained)
            .and_then(|v| v.checked_add(pmns_cache_bytes))
            .ok_or(ClientStateError::Invalid(
                "client state exceeds memory budget",
            ))?;
        let peak_temp: usize = validation_temp.max(block_iprfs_temp).max(rebuild_temp);
        check_budget(
            retained_bytes
                .checked_add(peak_temp)
                .ok_or(ClientStateError::Invalid(
                    "client state exceeds memory budget",
                ))?,
        )?;

        let mut in_flight: HashMap<usize, InFlight> = HashMap::new();
        in_flight
            .try_reserve(num_in_flight)
            .map_err(|_| ClientStateError::Invalid("client state exceeds memory budget"))?;
        let mut order: Vec<usize> = Vec::new();
        order
            .try_reserve_exact(num_in_flight)
            .map_err(|_| ClientStateError::Invalid("client state exceeds memory budget"))?;
        let mut reserved: HashSet<usize> = HashSet::new();
        reserved
            .try_reserve(num_in_flight)
            .map_err(|_| ClientStateError::Invalid("client state exceeds memory budget"))?;
        let mut record = [0u8; IN_FLIGHT_SIZE];
        for _ in 0..num_in_flight {
            input.read_hashed(&mut record)?;
            let queried_index: usize = index_below(&record, 0, params.num_entries).ok_or(
                ClientStateError::Invalid("queried index of a query in flight"),
            )?;
            if order.last().is_some_and(|&last| queried_index <= last) {
                return Err(ClientStateError::Invalid(
                    "the queries in flight are not in the order of their queried indices",
                ));
            }
            if cache.contains_key(&queried_index) {
                return Err(ClientStateError::Invalid(
                    "a query in flight goes to an index that has an answer",
                ));
            }
            // A decoy query is for an index that has an answer.
            let requested_index: usize = index_below(&record, 8, params.num_entries)
                .filter(|index| *index == queried_index || cache.contains_key(index))
                .ok_or(ClientStateError::Invalid(
                    "requested index of a query in flight",
                ))?;
            let hint_idx: usize = index_below(&record, 16, params.num_total_hints)
                .filter(|&hint_idx| slots[hint_idx].is_some() && reserved.insert(hint_idx))
                .ok_or(ClientStateError::Invalid(
                    "a query in flight has no hint, or two of them have the same hint",
                ))?;
            if record[24] > FLAG_REAL_IS_FIRST {
                return Err(ClientStateError::Invalid("flags of a query in flight"));
            }
            order.push(queried_index);
            in_flight.insert(
                queried_index,
                InFlight {
                    hint_idx,
                    requested_index,
                    real_is_first: record[24] == FLAG_REAL_IS_FIRST,
                },
            );
        }
        drop(reserved);

        // A retired backup hint is not promoted. After `retire_hints`, the
        // client can have more queries in flight than free backup hints.
        let num_retired: usize = read_count(
            &mut input,
            free_backups,
            "more retired backup hints than free backup hints",
        )?;
        let retired_bytes: usize = num_retired
            .checked_mul(std::mem::size_of::<usize>() + 16)
            .ok_or(ClientStateError::Invalid(
                "client state exceeds memory budget",
            ))?;
        retained_bytes =
            retained_bytes
                .checked_add(retired_bytes)
                .ok_or(ClientStateError::Invalid(
                    "client state exceeds memory budget",
                ))?;
        check_budget(
            retained_bytes
                .checked_add(peak_temp)
                .ok_or(ClientStateError::Invalid(
                    "client state exceeds memory budget",
                ))?,
        )?;
        let mut retired_backups: BTreeSet<usize> = BTreeSet::new();
        let mut number = [0u8; 8];
        for _ in 0..num_retired {
            input.read_hashed(&mut number)?;
            let backup_idx: usize = index_below(&number, 0, params.num_backup_hints)
                .filter(|&backup_idx| backup_available[backup_idx])
                .ok_or(ClientStateError::Invalid(
                    "a retired backup hint is promoted, or it has no valid cutoff",
                ))?;
            if retired_backups
                .last()
                .is_some_and(|&last| backup_idx <= last)
            {
                return Err(ClientStateError::Invalid(
                    "the retired backup hints are not in the order of their numbers",
                ));
            }
            backup_available[backup_idx] = false;
            retired_backups.insert(backup_idx);
        }

        input.finish()?;

        let queried: bool =
            any_sent || !cache.is_empty() || !in_flight.is_empty() || !retired_backups.is_empty();
        let remaining_backups: usize = backup_available.iter().filter(|&&free| free).count();
        let next_backup_hint: usize = backup_available
            .iter()
            .position(|&free| free)
            .unwrap_or(backup_available.len());
        let client = Self {
            iprfs: block_iprfs(&params, &master_seed),
            subset_prf: SubsetPrf::new(&master_seed),
            params,
            hints,
            slots,
            backup_available,
            remaining_backups,
            next_backup_hint,
            cache,
            rng: ChaCha20Rng::from_seed(query_rng_seed),
            in_flight,
            retired_backups,
            queried,
            threads: crate::hint_init::available_threads(),
        };

        // A message is a function of its hint, so the client builds each
        // query in flight again.
        let mut prepared: Vec<PreparedQuery> = Vec::with_capacity(order.len());
        for queried_index in order {
            let query: InFlight = client.in_flight[&queried_index];
            // `prepare_query` takes a hint from the iPRF inverse of the
            // offset, with the block in its subset.
            let block: usize = client.params.block_of(queried_index);
            let offset: usize = client.params.offset_in_block(queried_index);
            let message = client.slots[query.hint_idx]
                .as_ref()
                .filter(|slot| {
                    client.iprfs[block].forward(query.hint_idx as u64) == offset as u64
                        && (client.slot_block_in_subset(query.hint_idx, slot, block)
                            || slot.extra_index == Some(queried_index))
                })
                .and_then(|slot| {
                    client.try_build_query(query.hint_idx, queried_index, slot, query.real_is_first)
                });
            let Some(message) = message else {
                return Err(ClientStateError::Invalid(
                    "the hint of a query in flight does not hold its index",
                ));
            };
            prepared.push(PreparedQuery {
                query: message,
                requested_index: query.requested_index,
                queried_index,
                hint_idx: query.hint_idx,
                real_is_first: query.real_is_first,
            });
        }

        Ok((client, prepared, retained_bytes))
    }

    /// Removes hints that the client sent in queries that its state does not hold.
    ///
    /// Use it after [`Client::read_state`] with a state that is older than
    /// the last query: pass [`PreparedQuery::hint_index`] of each query that
    /// the client prepared after the [`Client::write_state`] call of that
    /// state. The caller keeps these numbers in a log and appends a number
    /// before its query leaves the client. After the call, no query uses one
    /// of these hints.
    ///
    /// A query in flight that uses one of the hints ends as with
    /// [`Client::abandon_query`]; the caller must drop the `PreparedQuery`
    /// that [`Client::read_state`] returned for it. A backup hint of the list is not promoted
    /// later, so the client can then have more queries in flight than backup
    /// hints: [`Client::reconstruct_and_replenish`] returns
    /// [`ProtocolError::NoBackupHintsLeft`] for the queries that have no
    /// backup hint. A hint that occurs two times, and a hint that is not
    /// present, are not errors. A list that removes no hint leaves
    /// [`Client::unused_hints`] as it is.
    ///
    /// The client has fewer hints after the call, with no replacement, as
    /// after [`Client::abandon_query`] (`docs/online_protocol.md`). The
    /// answers of the queries of the list are lost.
    ///
    /// # Errors
    /// Returns [`ProtocolError::InvalidParams`] if a number is not a hint of
    /// this client. No hint is removed then.
    pub fn retire_hints(&mut self, hint_indices: &[usize]) -> Result<(), ProtocolError> {
        if hint_indices
            .iter()
            .any(|&hint_idx| hint_idx >= self.params.num_total_hints)
        {
            return Err(ProtocolError::InvalidParams(
                "a hint to retire is not a hint of this client",
            ));
        }
        let retired: HashSet<usize> = hint_indices.iter().copied().collect();
        self.in_flight
            .retain(|_, query| !retired.contains(&query.hint_idx));
        for &hint_idx in hint_indices {
            let mut removed: bool = self.slots[hint_idx].take().is_some();
            if let Some(backup_idx) = hint_idx.checked_sub(self.params.num_regular_hints) {
                if self.consume_backup(backup_idx) {
                    self.retired_backups.insert(backup_idx);
                    removed = true;
                }
            }
            self.queried |= removed;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
