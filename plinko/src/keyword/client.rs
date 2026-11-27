//! Sans-I/O keyword PIR client (`KeywordClient<B>`), backend adapter trait
//! (`IndexPirClient`), checkpointing, WAL hint retirement, and sync transport
//! convenience layer (`KeywordSession`).
//!
//! Works identically over 2-segment [`MultiLevelClient`] and 1-level
//! [`crate::protocol::Client`] on both native targets and `wasm32-unknown-unknown`.

use super::{KeywordError, KeywordTableSpec, MutationOutcome, StashEntry, StashOp, NUM_SEGMENTS};
use crate::checked_io::{u32_at, u64_at, HashingReader, HashingWriter, TrailerError};
use crate::delta::{Delta, DeltaHeader};
use crate::multilevel::{
    MultiLevelAnswer, MultiLevelClient, MultiLevelError, MultiLevelFinishPlan, MultiLevelQuery,
    PendingMultiLevelQuery,
};
use crate::protocol::{
    Client as ProtocolClient, ClientStateError, PreparedQuery, ProtocolError, ProtocolFinishPlan,
    Query as ProtocolQuery, Response as ProtocolResponse,
};
use rand::Rng;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::error::Error;
use std::fmt;
use std::io::{self, Read, Write};
use std::sync::atomic::{AtomicU64, Ordering};

/// Magic bytes prefixing a serialized [`KeywordClient`] checkpoint.
pub const KEYWORD_CHECKPOINT_MAGIC: [u8; 8] = *b"PLNKKEYW";

/// Binary format version of a serialized [`KeywordClient`] checkpoint.
pub const KEYWORD_CHECKPOINT_VERSION: u32 = 1;

/// Default maximum decoded byte budget accepted when loading a [`KeywordClient`]
/// checkpoint (`512 MiB`, matching [`MultiLevelClient::read_hints`]).
pub const MAX_KEYWORD_CHECKPOINT_BYTES: usize = 512 * 1024 * 1024;

/// Monotonic per-process session ID generator preventing cross-session pending misuse.
static NEXT_KEYWORD_SESSION_ID: AtomicU64 = AtomicU64::new(1);

fn next_keyword_session_id() -> u64 {
    let id: u64 = NEXT_KEYWORD_SESSION_ID.fetch_add(1, Ordering::Relaxed);
    if id == 0 {
        NEXT_KEYWORD_SESSION_ID.fetch_add(1, Ordering::Relaxed)
    } else {
        id
    }
}

/// Client-side caching policy controlling the privacy vs. bandwidth tradeoff.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CachePolicy {
    /// Always issue 2 PIR queries per input key (no local bucket caching, no
    /// stash shortcut, and no batch size reduction on duplicate keys).
    ///
    /// Under [`crate::protocol::Client`], querying a bucket index that the
    /// protocol client already cached internally substitutes a random uncached,
    /// non-in-flight decoy index; both single and batch lookups then fail with
    /// [`ProtocolError::NoDecoyIndexAvailable`] once a segment has no remaining
    /// uncached, non-in-flight decoy index, even if backup hints remain. Under
    /// [`MultiLevelClient`], no internal index cache is kept and concurrent
    /// queries to the same bucket remain allowed while matching slots are free.
    #[default]
    Off,
    /// Answer cached/stashed hits locally and deduplicate repeated keys or shared
    /// buckets inside batches without spending backup hints `Q`.
    ///
    /// Reveals hit and deduplication counts on the wire.
    LocalHit,
    /// Answer cached/stashed/duplicate hits locally while still issuing 2 decoy
    /// PIR queries per such input key so wire traffic stays uniform (`1` query
    /// pair per input key).
    ///
    /// For cached, stashed, or duplicate entries that need a decoy index, both
    /// single and batch lookups fail with
    /// [`KeywordClientError::NoDecoyIndexAvailable`] once a segment has no
    /// remaining uncached, non-in-flight index, even if backup hints remain. For
    /// an uncached bucket whose index is already in flight on
    /// [`crate::protocol::Client`], the backend rejects the concurrent query with
    /// [`ProtocolError::QueryInFlight`] (surfaced as
    /// [`KeywordClientError::Backend`]).
    ConstantTraffic,
}

impl CachePolicy {
    fn to_byte(self) -> u8 {
        match self {
            Self::Off => 0,
            Self::LocalHit => 1,
            Self::ConstantTraffic => 2,
        }
    }

    fn from_byte(b: u8) -> Option<Self> {
        match b {
            0 => Some(Self::Off),
            1 => Some(Self::LocalHit),
            2 => Some(Self::ConstantTraffic),
            _ => None,
        }
    }
}

/// Errors returned by [`KeywordClient`] operations.
#[derive(Debug)]
pub enum KeywordClientError<E> {
    /// Error from KV-SCF hashing, scanning, or parameter validation.
    Keyword(KeywordError),
    /// Error from one of the underlying segment index PIR backends.
    Backend {
        /// Segment index (`0` or `1`) that reported the error.
        segment: usize,
        /// Underlying backend error.
        error: E,
    },
    /// Backend geometry (`num_entries` or `record_size`) does not match `KeywordTableSpec`.
    BackendGeometryMismatch {
        /// Segment index (`0` or `1`).
        segment: usize,
        /// Expected `(num_entries, record_size)`.
        expected: (usize, usize),
        /// Observed `(num_entries, record_size)`.
        actual: (usize, usize),
    },
    /// Answer or update `table_epoch` does not match the client's `KeywordTableSpec`.
    TableEpochMismatch {
        /// Client's current `table_epoch`.
        expected: u64,
        /// Received `table_epoch`.
        got: u64,
    },
    /// Answer, pending handle, or update `state_epoch` does not match the client.
    StateEpochMismatch {
        /// Client's current `state_epoch`.
        expected: u64,
        /// Received `state_epoch`.
        got: u64,
    },
    /// Pending handle belongs to another [`KeywordClient`] instance or reloaded session.
    SessionMismatch,
    /// Batch answer length does not match the pending batch query count for a segment.
    AnswerCountMismatch {
        /// Segment index (`0` or `1`).
        segment: usize,
        /// Expected number of segment answers.
        expected: usize,
        /// Received number of segment answers.
        actual: usize,
    },
    /// Segment has no available non-cached, non-in-flight decoy index for the batch.
    NoDecoyIndexAvailable {
        /// Segment index (`0` or `1`).
        segment: usize,
    },
    /// Epoch update is malformed (non-increasing epoch, invalid delta, or invalid stash op).
    InvalidEpochUpdate(&'static str),
    /// Checkpoint cannot be written while queries are in flight.
    QueryInFlight,
    /// Checkpoint serialization or deserialization failed.
    Checkpoint(&'static str),
    /// I/O error during checkpoint serialization or deserialization.
    Io(io::Error),
}

impl<E: fmt::Display> fmt::Display for KeywordClientError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Keyword(err) => write!(f, "{err}"),
            Self::Backend { segment, error } => write!(f, "segment {segment} PIR error: {error}"),
            Self::BackendGeometryMismatch {
                segment,
                expected,
                actual,
            } => write!(
                f,
                "segment {segment} backend geometry mismatch: expected (entries={}, record_size={}), got (entries={}, record_size={})",
                expected.0, expected.1, actual.0, actual.1
            ),
            Self::TableEpochMismatch { expected, got } => {
                write!(f, "table epoch mismatch: expected {expected}, got {got}")
            }
            Self::StateEpochMismatch { expected, got } => {
                write!(f, "state epoch mismatch: expected {expected}, got {got}")
            }
            Self::SessionMismatch => {
                write!(f, "pending lookup belongs to another KeywordClient session")
            }
            Self::AnswerCountMismatch {
                segment,
                expected,
                actual,
            } => write!(
                f,
                "segment {segment} batch answer count mismatch: expected {expected}, got {actual}"
            ),
            Self::NoDecoyIndexAvailable { segment } => {
                write!(f, "segment {segment} has no free decoy index available")
            }
            Self::InvalidEpochUpdate(msg) => write!(f, "invalid keyword epoch update: {msg}"),
            Self::QueryInFlight => {
                write!(
                    f,
                    "cannot write keyword checkpoint while queries are in flight"
                )
            }
            Self::Checkpoint(msg) => write!(f, "keyword checkpoint error: {msg}"),
            Self::Io(err) => write!(f, "I/O error: {err}"),
        }
    }
}

impl<E: Error + 'static> Error for KeywordClientError<E> {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Keyword(err) => Some(err),
            Self::Backend { error, .. } => Some(error),
            Self::Io(err) => Some(err),
            _ => None,
        }
    }
}

impl<E> From<KeywordError> for KeywordClientError<E> {
    fn from(err: KeywordError) -> Self {
        Self::Keyword(err)
    }
}

impl<E> From<io::Error> for KeywordClientError<E> {
    fn from(err: io::Error) -> Self {
        Self::Io(err)
    }
}

/// Backend-agnostic interface for an index PIR client driving one KV-SCF segment.
///
/// Implemented out of the box for [`MultiLevelClient`] and [`crate::protocol::Client`].
pub trait IndexPirClient {
    /// Wire query type sent to the server for one index lookup.
    type Query: Clone;
    /// Wire answer type returned by the server for one index lookup.
    type Answer;
    /// Client-side pending handle for an in-flight index query.
    type Pending;
    /// Stable hint identifier logged to a WAL before transmission and passed to
    /// [`IndexPirClient::retire_index_hints`] after a crash/reload.
    type HintId: Copy + Eq + fmt::Debug;
    /// Staged, validated batch of answer reconstructions and hint refreshes
    /// produced by [`IndexPirClient::plan_finish_batch`] and committed infallibly
    /// by [`IndexPirClient::commit_finish_batch`].
    type FinishPlan;
    /// Staged, validated batch of XOR bucket deltas produced by
    /// [`IndexPirClient::prepare_index_deltas`] and committed infallibly by
    /// [`IndexPirClient::commit_index_deltas`].
    type DeltaPlan;
    /// Backend error type.
    type Error: Error + Send + Sync + 'static;

    /// Number of logical records (`s`) served by this segment backend.
    fn num_entries(&self) -> usize;

    /// Byte length of each record (`b * (tag_size + value_size)`).
    fn record_size(&self) -> usize;

    /// Current backend snapshot epoch / database version for this segment.
    fn epoch(&self) -> u64;

    /// Conservative lower bound on additional queries this segment can prepare.
    fn remaining_queries(&self) -> usize;

    /// Number of index queries currently in flight on this segment.
    fn queries_in_flight(&self) -> usize;

    /// Returns `true` if `index` is already cached internally by this backend
    /// (causing `prepare_index_query(index)` to emit an internal decoy query).
    fn is_index_cached(&self, index: usize) -> bool;

    /// Returns `true` if the backend forbids preparing another concurrent query
    /// targeting wire index `index`.
    fn is_index_in_flight(&self, index: usize) -> bool;

    /// Prepares a query for `index`.
    fn prepare_index_query<R: Rng + ?Sized>(
        &mut self,
        index: usize,
        rng: &mut R,
    ) -> Result<Self::Pending, Self::Error>;

    /// Returns the wire query to transmit for `pending`.
    fn pending_query(pending: &Self::Pending) -> &Self::Query;

    /// Returns the database index targeted on the wire by `pending`.
    fn pending_queried_index(pending: &Self::Pending) -> usize;

    /// Returns the WAL hint identifier exposed by `pending`.
    fn pending_hint_id(pending: &Self::Pending) -> Self::HintId;

    /// Validates `(pendings, answers)` and plans all answer reconstructions and
    /// hint refreshes (including any random draws) without mutating `self`.
    ///
    /// Returning `Ok(plan)` guarantees [`Self::commit_finish_batch`] will succeed
    /// infallibly when `self` is not mutated in between.
    fn plan_finish_batch<R: Rng + ?Sized>(
        &self,
        pendings: &[&Self::Pending],
        answers: &[Self::Answer],
        rng: &mut R,
    ) -> Result<Self::FinishPlan, Self::Error>;

    /// Returns the reconstructed record bytes staged inside `plan` in `pendings`
    /// order so callers can validate reconstructed payloads before calling
    /// [`Self::commit_finish_batch`].
    fn plan_finish_values(plan: &Self::FinishPlan) -> &[Vec<u8>];

    /// Commits a staged [`Self::FinishPlan`] produced by [`Self::plan_finish_batch`]
    /// and returns the reconstructed records in `pendings` order.
    fn commit_finish_batch(&mut self, plan: Self::FinishPlan) -> Vec<Vec<u8>>;

    /// Aborts a potentially-sent in-flight query, permanently retiring its hint.
    fn abort_index_query(&mut self, pending: Self::Pending) -> Result<(), Self::Error>;

    /// Cancels an unsent in-flight query without retiring its hint.
    fn cancel_unsent_index_query(&mut self, pending: Self::Pending) -> Result<(), Self::Error>;

    /// Validates WAL hint identifiers before mutating state.
    fn validate_retire_hints(&self, hints: &[Self::HintId]) -> Result<(), Self::Error>;

    /// Retires WAL-logged hint identifiers after reloading a checkpoint.
    fn retire_index_hints(&mut self, hints: &[Self::HintId]) -> Result<(), Self::Error>;

    /// Validates and stages a batch of `(bucket_idx, xor_delta)` updates without
    /// mutating `self`.
    ///
    /// An empty slice must produce a no-op plan that does not advance
    /// [`IndexPirClient::epoch`]. Returning `Ok(plan)` guarantees
    /// [`Self::commit_index_deltas`] will succeed infallibly when `self` is not
    /// mutated in between.
    fn prepare_index_deltas(
        &self,
        deltas: &[(usize, Vec<u8>)],
    ) -> Result<Self::DeltaPlan, Self::Error>;

    /// Commits a staged [`Self::DeltaPlan`] produced by [`Self::prepare_index_deltas`].
    fn commit_index_deltas(&mut self, plan: Self::DeltaPlan, threads: usize);
}

impl IndexPirClient for MultiLevelClient {
    type Query = MultiLevelQuery;
    type Answer = MultiLevelAnswer;
    type Pending = PendingMultiLevelQuery;
    type HintId = (usize, u32);
    type FinishPlan = MultiLevelFinishPlan;
    type DeltaPlan = Vec<(usize, Vec<u8>)>;
    type Error = MultiLevelError;

    fn num_entries(&self) -> usize {
        self.params().num_entries
    }

    fn record_size(&self) -> usize {
        self.params().record_size
    }

    fn epoch(&self) -> u64 {
        MultiLevelClient::epoch(self)
    }

    fn remaining_queries(&self) -> usize {
        self.remaining_backups()
            .into_iter()
            .min()
            .unwrap_or(0)
            .saturating_sub(MultiLevelClient::queries_in_flight(self))
    }

    fn queries_in_flight(&self) -> usize {
        MultiLevelClient::queries_in_flight(self)
    }

    fn is_index_cached(&self, _index: usize) -> bool {
        false
    }

    fn is_index_in_flight(&self, _index: usize) -> bool {
        false
    }

    fn prepare_index_query<R: Rng + ?Sized>(
        &mut self,
        index: usize,
        rng: &mut R,
    ) -> Result<Self::Pending, Self::Error> {
        self.prepare_query(index, rng)
    }

    fn pending_query(pending: &Self::Pending) -> &Self::Query {
        &pending.query
    }

    fn pending_queried_index(pending: &Self::Pending) -> usize {
        pending.index
    }

    fn pending_hint_id(pending: &Self::Pending) -> Self::HintId {
        pending.hint_id()
    }

    fn plan_finish_batch<R: Rng + ?Sized>(
        &self,
        pendings: &[&Self::Pending],
        answers: &[Self::Answer],
        rng: &mut R,
    ) -> Result<Self::FinishPlan, Self::Error> {
        MultiLevelClient::plan_finish_batch(self, pendings, answers, rng)
    }

    fn plan_finish_values(plan: &Self::FinishPlan) -> &[Vec<u8>] {
        plan.values()
    }

    fn commit_finish_batch(&mut self, plan: Self::FinishPlan) -> Vec<Vec<u8>> {
        MultiLevelClient::commit_finish_batch(self, plan)
    }

    fn abort_index_query(&mut self, pending: Self::Pending) -> Result<(), Self::Error> {
        self.abort_query(&pending)
    }

    fn cancel_unsent_index_query(&mut self, pending: Self::Pending) -> Result<(), Self::Error> {
        self.cancel_unsent_query(&pending)
    }

    fn validate_retire_hints(&self, hints: &[Self::HintId]) -> Result<(), Self::Error> {
        MultiLevelClient::validate_retire_hints(self, hints)
    }

    fn retire_index_hints(&mut self, hints: &[Self::HintId]) -> Result<(), Self::Error> {
        self.retire_hints(hints)
    }

    fn prepare_index_deltas(
        &self,
        deltas: &[(usize, Vec<u8>)],
    ) -> Result<Self::DeltaPlan, Self::Error> {
        let n: usize = self.params().num_entries;
        let r: usize = self.params().record_size;
        for (idx, delta) in deltas {
            if *idx >= n {
                return Err(MultiLevelError::IndexOutOfRange {
                    index: *idx,
                    num_entries: n,
                });
            }
            if delta.len() != r {
                return Err(MultiLevelError::DeltaLengthMismatch {
                    expected: r,
                    actual: delta.len(),
                });
            }
        }
        Ok(deltas.to_vec())
    }

    fn commit_index_deltas(&mut self, plan: Self::DeltaPlan, threads: usize) {
        if !plan.is_empty() {
            self.apply_deltas_batch(&plan, threads)
                .expect("deltas validated in prepare_index_deltas");
        }
    }
}

impl IndexPirClient for ProtocolClient {
    type Query = ProtocolQuery;
    type Answer = ProtocolResponse;
    type Pending = PreparedQuery;
    type HintId = usize;
    type FinishPlan = ProtocolFinishPlan;
    type DeltaPlan = Option<Delta>;
    type Error = ProtocolError;

    fn num_entries(&self) -> usize {
        self.params().num_entries
    }

    fn record_size(&self) -> usize {
        self.params().record_size
    }

    fn epoch(&self) -> u64 {
        self.database_version()
    }

    fn remaining_queries(&self) -> usize {
        self.remaining_backup_hints()
            .saturating_sub(ProtocolClient::queries_in_flight(self))
    }

    fn queries_in_flight(&self) -> usize {
        ProtocolClient::queries_in_flight(self)
    }

    fn is_index_cached(&self, index: usize) -> bool {
        self.is_cached(index)
    }

    fn is_index_in_flight(&self, index: usize) -> bool {
        self.is_in_flight(index)
    }

    fn prepare_index_query<R: Rng + ?Sized>(
        &mut self,
        index: usize,
        _rng: &mut R,
    ) -> Result<Self::Pending, Self::Error> {
        self.prepare_query(index)
    }

    fn pending_query(pending: &Self::Pending) -> &Self::Query {
        &pending.query
    }

    fn pending_queried_index(pending: &Self::Pending) -> usize {
        pending.queried_index()
    }

    fn pending_hint_id(pending: &Self::Pending) -> Self::HintId {
        pending.hint_index()
    }

    fn plan_finish_batch<R: Rng + ?Sized>(
        &self,
        pendings: &[&Self::Pending],
        answers: &[Self::Answer],
        _rng: &mut R,
    ) -> Result<Self::FinishPlan, Self::Error> {
        ProtocolClient::plan_finish_batch(self, pendings, answers)
    }

    fn plan_finish_values(plan: &Self::FinishPlan) -> &[Vec<u8>] {
        plan.values()
    }

    fn commit_finish_batch(&mut self, plan: Self::FinishPlan) -> Vec<Vec<u8>> {
        ProtocolClient::commit_finish_batch(self, plan)
    }

    fn abort_index_query(&mut self, pending: Self::Pending) -> Result<(), Self::Error> {
        self.abandon_query(pending)
    }

    fn cancel_unsent_index_query(&mut self, pending: Self::Pending) -> Result<(), Self::Error> {
        self.cancel_unsent_query(pending)
    }

    fn validate_retire_hints(&self, hints: &[Self::HintId]) -> Result<(), Self::Error> {
        ProtocolClient::validate_retire_hints(self, hints)
    }

    fn retire_index_hints(&mut self, hints: &[Self::HintId]) -> Result<(), Self::Error> {
        self.retire_hints(hints)
    }

    fn prepare_index_deltas(
        &self,
        deltas: &[(usize, Vec<u8>)],
    ) -> Result<Self::DeltaPlan, Self::Error> {
        if deltas.is_empty() {
            return Ok(None);
        }
        let from_version: u64 = self.database_version();
        let to_version: u64 = from_version
            .checked_add(1)
            .ok_or(ProtocolError::InvalidParams("database_version overflow"))?;
        let record_size: u32 = u32::try_from(self.params().record_size)
            .map_err(|_| ProtocolError::InvalidParams("record_size exceeds u32::MAX"))?;
        let header: DeltaHeader = DeltaHeader {
            num_entries: self.params().num_entries as u64,
            record_size,
            from_version,
            to_version,
        };
        let changes: Vec<(u64, Vec<u8>)> = deltas
            .iter()
            .map(|(idx, xor_delta)| (*idx as u64, xor_delta.clone()))
            .collect();
        let delta: Delta = Delta::new(header, changes)
            .map_err(|_| ProtocolError::InvalidParams("invalid keyword segment delta"))?;
        Ok(Some(delta))
    }

    fn commit_index_deltas(&mut self, plan: Self::DeltaPlan, _threads: usize) {
        if let Some(delta) = plan {
            self.apply_delta(&delta)
                .expect("delta validated in prepare_index_deltas");
        }
    }
}

/// Wire query for a single 2-segment keyword lookup.
///
/// Omits local `session_id` and `query_id` fields so callers can route requests
/// through an OHTTP relay (RFC 9458) without forwarding those identifiers. Any
/// cross-request unlinkability guarantees depend on the transport and deployment.
#[derive(Debug, Clone)]
pub struct KeywordLookupQuery<Q> {
    /// Expected table major epoch.
    pub table_epoch: u64,
    /// Expected table minor state epoch.
    pub state_epoch: u64,
    /// Segment 0 and segment 1 index PIR queries.
    pub segments: [Q; NUM_SEGMENTS],
}

/// Wire answer returned by the server for a [`KeywordLookupQuery`].
///
/// Each `segments[j]` carries its own segment backend epoch (`snapshot_epoch` or
/// `database_version`), validated independently against `backend[j].epoch()`.
#[derive(Debug, Clone)]
pub struct KeywordLookupAnswer<A> {
    /// Server's table major epoch.
    pub table_epoch: u64,
    /// Server's global keyword state epoch.
    pub state_epoch: u64,
    /// Segment 0 and segment 1 index PIR answers.
    pub segments: [A; NUM_SEGMENTS],
}

/// Wire query for a 1-RTT batch of keyword lookups.
///
/// Under [`CachePolicy::Off`] and [`CachePolicy::ConstantTraffic`], both segment
/// vectors have length equal to the caller's input batch size. Under
/// [`CachePolicy::LocalHit`], cached/stashed/duplicate entries may reduce either
/// segment's vector independently.
#[derive(Debug, Clone)]
pub struct KeywordBatchQuery<Q> {
    /// Expected table major epoch.
    pub table_epoch: u64,
    /// Expected table minor state epoch.
    pub state_epoch: u64,
    /// Segment 0 and segment 1 index PIR query lists.
    pub segments: [Vec<Q>; NUM_SEGMENTS],
}

/// Wire answer returned by the server for a [`KeywordBatchQuery`].
#[derive(Debug, Clone)]
pub struct KeywordBatchAnswer<A> {
    /// Server's table major epoch.
    pub table_epoch: u64,
    /// Server's global keyword state epoch.
    pub state_epoch: u64,
    /// Segment 0 and segment 1 index PIR answer lists.
    pub segments: [Vec<A>; NUM_SEGMENTS],
}

/// Synchronized keyword table epoch update combining per-segment XOR bucket
/// deltas and runtime-stash operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KeywordEpochUpdate {
    /// Table major epoch this update applies to.
    pub table_epoch: u64,
    /// Required client `state_epoch` immediately before applying this update.
    pub from_state_epoch: u64,
    /// Resulting client `state_epoch` after applying this update (`> from_state_epoch`).
    pub to_state_epoch: u64,
    /// Per-segment `(bucket_idx, xor_delta)` updates. An empty vector leaves that
    /// segment's backend epoch unchanged.
    pub segment_deltas: [Vec<(usize, Vec<u8>)>; NUM_SEGMENTS],
    /// Ordered runtime-stash mutations to apply to the client's public stash copy.
    pub stash_ops: Vec<StashOp>,
}

#[inline]
fn xor_bytes_in_place(dst: &mut [u8], src: &[u8]) {
    for (d, &s) in dst.iter_mut().zip(src.iter()) {
        *d ^= s;
    }
}

#[inline]
fn is_all_zero(bytes: &[u8]) -> bool {
    bytes.iter().all(|&b| b == 0)
}

impl KeywordEpochUpdate {
    /// Builds a [`KeywordEpochUpdate`] from a sequence of [`MutationOutcome`]s
    /// produced by [`super::KvScfTable::insert`], `update`, `upsert`, or `delete`.
    ///
    /// Coalesces multiple deltas targeting the same `(segment, bucket_idx)` by XOR
    /// and omits buckets whose net XOR delta is all-zero.
    pub fn from_outcomes(
        table_epoch: u64,
        from_state_epoch: u64,
        to_state_epoch: u64,
        outcomes: &[MutationOutcome],
    ) -> Self {
        let mut combined: [BTreeMap<usize, Vec<u8>>; NUM_SEGMENTS] =
            [BTreeMap::new(), BTreeMap::new()];
        let mut stash_ops: Vec<StashOp> = Vec::new();
        for outcome in outcomes {
            for delta in &outcome.deltas {
                let seg_map: &mut BTreeMap<usize, Vec<u8>> = &mut combined[delta.segment];
                seg_map
                    .entry(delta.bucket_idx)
                    .and_modify(|existing| xor_bytes_in_place(existing, &delta.xor_delta))
                    .or_insert_with(|| delta.xor_delta.clone());
            }
            if let Some(op) = &outcome.stash_op {
                stash_ops.push(op.clone());
            }
        }
        let seg0: Vec<(usize, Vec<u8>)> = combined[0]
            .iter()
            .filter(|(_, d)| d.iter().any(|&b| b != 0))
            .map(|(&idx, d)| (idx, d.clone()))
            .collect();
        let seg1: Vec<(usize, Vec<u8>)> = combined[1]
            .iter()
            .filter(|(_, d)| d.iter().any(|&b| b != 0))
            .map(|(&idx, d)| (idx, d.clone()))
            .collect();
        Self {
            table_epoch,
            from_state_epoch,
            to_state_epoch,
            segment_deltas: [seg0, seg1],
            stash_ops,
        }
    }
}

/// Result returned by [`KeywordClient::prepare_lookup`].
pub type PrepareLookupResult<B> = Result<
    LookupPlan<<B as IndexPirClient>::Query, PendingKeywordLookup<B>>,
    KeywordClientError<<B as IndexPirClient>::Error>,
>;

/// Result returned by [`KeywordClient::prepare_batch`].
pub type PrepareBatchResult<B> = Result<
    BatchLookupPlan<<B as IndexPirClient>::Query, PendingBatchLookup<B>>,
    KeywordClientError<<B as IndexPirClient>::Error>,
>;

/// Result returned by [`KeywordSession::get`] and [`KeywordSession::get_batch`].
pub type SessionResult<V, B, T> = Result<
    V,
    KeywordSessionError<
        <B as IndexPirClient>::Error,
        <T as KeywordTransport<<B as IndexPirClient>::Query, <B as IndexPirClient>::Answer>>::Error,
    >,
>;

/// Result of [`KeywordClient::prepare_lookup`].
#[derive(Debug)]
pub enum LookupPlan<Q, P> {
    /// Answered locally under [`CachePolicy::LocalHit`] without sending a PIR query.
    Local(Option<Vec<u8>>),
    /// Requires sending `query` to the server and completing with `finish_lookup`.
    Query {
        /// Wire query to send to the server.
        query: KeywordLookupQuery<Q>,
        /// Client-side pending handle to keep until `finish_lookup` or `abort_lookup`.
        pending: P,
    },
}

/// Result of [`KeywordClient::prepare_batch`].
#[derive(Debug)]
pub enum BatchLookupPlan<Q, P> {
    /// Every key in the batch was answered locally (empty batch or [`CachePolicy::LocalHit`]).
    Local(Vec<Option<Vec<u8>>>),
    /// Requires sending `query` to the server and completing with `finish_batch`.
    Query {
        /// Wire batch query to send to the server.
        query: KeywordBatchQuery<Q>,
        /// Client-side pending handle to keep until `finish_batch` or `abort_batch`.
        pending: P,
    },
}

#[derive(Debug, Clone)]
enum LookupResolution {
    /// Stash hit resolved locally under `ConstantTraffic` (wire queries are decoys).
    StashHit(Vec<u8>),
    /// Reconstruct from wire answers and/or keyword bucket cache, then scan with stash.
    Buckets {
        bucket_seg0: usize,
        bucket_seg1: usize,
        tag: Vec<u8>,
        /// Whether segment `j` wire query carries the real bucket (`true`) or a decoy (`false`).
        wire_is_real: [bool; NUM_SEGMENTS],
    },
}

/// Client-side pending state for an in-flight single keyword lookup.
pub struct PendingKeywordLookup<B: IndexPirClient> {
    session_id: u64,
    table_epoch: u64,
    state_epoch: u64,
    resolution: LookupResolution,
    hint_ids: [B::HintId; NUM_SEGMENTS],
    segments: [Option<B::Pending>; NUM_SEGMENTS],
}

impl<B: IndexPirClient> fmt::Debug for PendingKeywordLookup<B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PendingKeywordLookup")
            .field("table_epoch", &self.table_epoch)
            .field("state_epoch", &self.state_epoch)
            .field("hint_ids", &self.hint_ids)
            .finish()
    }
}

impl<B: IndexPirClient> PendingKeywordLookup<B> {
    /// Returns the `[seg0_hint_id, seg1_hint_id]` pair to append durably to a
    /// write-ahead log before transmitting the wire query.
    pub fn hint_ids(&self) -> [B::HintId; NUM_SEGMENTS] {
        self.hint_ids
    }

    /// Returns the `table_epoch` at which this lookup was prepared.
    pub fn table_epoch(&self) -> u64 {
        self.table_epoch
    }

    /// Returns the `state_epoch` at which this lookup was prepared.
    pub fn state_epoch(&self) -> u64 {
        self.state_epoch
    }
}

#[derive(Debug, Clone, Copy)]
enum BatchBucketSource {
    /// Read from recovered wire answer at `wire_idx` on segment `j`.
    WireSlot(usize),
    /// Read from `self.bucket_cache[j]` at `bucket_idx`.
    Cached(usize),
}

#[derive(Debug, Clone)]
enum BatchEntryPlan {
    /// Local hit already resolved at prepare time (stash hit or fully cached under `LocalHit`).
    Local(Option<Vec<u8>>),
    /// Scan buckets resolved from wire slots and/or cache together with runtime-stash.
    Scan {
        bucket_seg0: usize,
        tag: Vec<u8>,
        source0: BatchBucketSource,
        source1: BatchBucketSource,
    },
}

/// Client-side pending state for an in-flight batch of keyword lookups.
pub struct PendingBatchLookup<B: IndexPirClient> {
    session_id: u64,
    table_epoch: u64,
    state_epoch: u64,
    /// Per-segment WAL hint IDs exposed by this batch.
    hint_ids: [Vec<B::HintId>; NUM_SEGMENTS],
    /// Per-segment in-flight backend handles (`None` after abort).
    segments: [Vec<Option<B::Pending>>; NUM_SEGMENTS],
    /// For each wire slot on segment `j`: `Some(bucket_idx)` if the slot fetches a
    /// real bucket (to populate `bucket_cache[j]` when enabled), or `None` for a decoy.
    wire_real_buckets: [Vec<Option<usize>>; NUM_SEGMENTS],
    /// How each input key `0..keys.len()` reconstructs its result.
    entries: Vec<BatchEntryPlan>,
}

impl<B: IndexPirClient> fmt::Debug for PendingBatchLookup<B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PendingBatchLookup")
            .field("table_epoch", &self.table_epoch)
            .field("state_epoch", &self.state_epoch)
            .field("wire_query_counts", &self.wire_query_counts())
            .field("hint_ids", &self.hint_ids)
            .finish()
    }
}

impl<B: IndexPirClient> PendingBatchLookup<B> {
    /// Returns the per-segment WAL hint IDs exposed by this batch query.
    ///
    /// Append both lists durably to a write-ahead log before transmitting the batch.
    pub fn hint_ids(&self) -> [Vec<B::HintId>; NUM_SEGMENTS] {
        self.hint_ids.clone()
    }

    /// Returns the `table_epoch` at which this batch was prepared.
    pub fn table_epoch(&self) -> u64 {
        self.table_epoch
    }

    /// Returns the `state_epoch` at which this batch was prepared.
    pub fn state_epoch(&self) -> u64 {
        self.state_epoch
    }

    /// Returns `[seg0_wire_count, seg1_wire_count]` for this batch.
    pub fn wire_query_counts(&self) -> [usize; NUM_SEGMENTS] {
        [self.segments[0].len(), self.segments[1].len()]
    }
}

/// Sans-I/O 2-segment KV-SCF keyword PIR client over any [`IndexPirClient`].
pub struct KeywordClient<B: IndexPirClient> {
    session_id: u64,
    spec: KeywordTableSpec,
    backends: [B; NUM_SEGMENTS],
    cache_policy: CachePolicy,
    /// Optional public runtime-stash keyed by `(bucket_seg0, tag) -> value`.
    stash: HashMap<(usize, Vec<u8>), Vec<u8>>,
    /// Optional XOR-maintained keyword bucket cache (populated only when
    /// `cache_policy != CachePolicy::Off`).
    bucket_cache: [HashMap<usize, Vec<u8>>; NUM_SEGMENTS],
}

impl<B: IndexPirClient> KeywordClient<B> {
    /// Creates a new [`KeywordClient`] from a public [`KeywordTableSpec`], two
    /// initialized segment PIR clients (`[segment_0, segment_1]`), and a
    /// [`CachePolicy`].
    ///
    /// Verifies that both segment backends match `(spec.buckets_per_segment(),
    /// spec.record_size())`.
    pub fn new(
        spec: KeywordTableSpec,
        backends: [B; NUM_SEGMENTS],
        cache_policy: CachePolicy,
    ) -> Result<Self, KeywordClientError<B::Error>> {
        Self::with_stash(spec, backends, cache_policy, &[])
    }

    /// Creates a new [`KeywordClient`] pre-populated with an initial public
    /// runtime-stash snapshot.
    pub fn with_stash(
        spec: KeywordTableSpec,
        backends: [B; NUM_SEGMENTS],
        cache_policy: CachePolicy,
        initial_stash: &[StashEntry],
    ) -> Result<Self, KeywordClientError<B::Error>> {
        let expected: (usize, usize) = (spec.buckets_per_segment(), spec.record_size());
        for (seg, backend) in backends.iter().enumerate() {
            let actual: (usize, usize) = (backend.num_entries(), backend.record_size());
            if actual != expected {
                return Err(KeywordClientError::BackendGeometryMismatch {
                    segment: seg,
                    expected,
                    actual,
                });
            }
        }
        let mut stash: HashMap<(usize, Vec<u8>), Vec<u8>> =
            HashMap::with_capacity(initial_stash.len());
        for entry in initial_stash {
            validate_stash_fields(&spec, entry.bucket_seg0, &entry.tag, Some(&entry.value))?;
            if stash
                .insert((entry.bucket_seg0, entry.tag.clone()), entry.value.clone())
                .is_some()
            {
                return Err(KeywordClientError::Keyword(KeywordError::DuplicateKey));
            }
        }
        Ok(Self {
            session_id: next_keyword_session_id(),
            spec,
            backends,
            cache_policy,
            stash,
            bucket_cache: [HashMap::new(), HashMap::new()],
        })
    }

    /// Returns the public [`KeywordTableSpec`].
    pub fn spec(&self) -> &KeywordTableSpec {
        &self.spec
    }

    /// Returns the configured [`CachePolicy`].
    pub fn cache_policy(&self) -> CachePolicy {
        self.cache_policy
    }

    /// Returns the current number of entries in the client's public runtime-stash copy.
    pub fn stash_len(&self) -> usize {
        self.stash.len()
    }

    /// Returns read-only references to `[segment_0, segment_1]` backends.
    pub fn backends(&self) -> &[B; NUM_SEGMENTS] {
        &self.backends
    }

    /// Returns the current backend snapshot epochs `[seg0_epoch, seg1_epoch]`.
    pub fn segment_epochs(&self) -> [u64; NUM_SEGMENTS] {
        [self.backends[0].epoch(), self.backends[1].epoch()]
    }

    /// Conservative lower bound on additional 2-segment lookups this client can prepare.
    pub fn remaining_queries(&self) -> usize {
        self.backends[0]
            .remaining_queries()
            .min(self.backends[1].remaining_queries())
    }

    /// Total number of segment queries currently in flight across both segments.
    pub fn queries_in_flight(&self) -> usize {
        self.backends[0].queries_in_flight() + self.backends[1].queries_in_flight()
    }

    /// Returns the number of cached buckets `[seg0_cached, seg1_cached]` held by
    /// the keyword client layer (always `[0, 0]` under [`CachePolicy::Off`]).
    pub fn cached_bucket_counts(&self) -> [usize; NUM_SEGMENTS] {
        [self.bucket_cache[0].len(), self.bucket_cache[1].len()]
    }

    fn stash_lookup(&self, bucket_seg0: usize, tag: &[u8]) -> Option<&[u8]> {
        // Look up without allocating when possible is nice, but tag is 8-16 bytes.
        self.stash
            .get(&(bucket_seg0, tag.to_vec()))
            .map(Vec::as_slice)
    }

    /// Chooses a uniform decoy bucket index in `0..s` on segment `seg` that is not
    /// cached and not in flight in `backend[seg]`, preferring indices outside `avoid`.
    fn pick_decoy_index<R: Rng + ?Sized>(
        &self,
        seg: usize,
        avoid: &HashSet<usize>,
        rng: &mut R,
    ) -> Result<usize, KeywordClientError<B::Error>> {
        let s: usize = self.spec.buckets_per_segment();
        let backend: &B = &self.backends[seg];

        // Fast rejection path: prefer indices that are neither backend-busy nor in `avoid`.
        for _ in 0..16 {
            let cand: usize = rng.gen_range(0..s);
            if !avoid.contains(&cand)
                && !backend.is_index_cached(cand)
                && !backend.is_index_in_flight(cand)
            {
                return Ok(cand);
            }
        }

        // Exact reservoir scan preferring `!avoid`, falling back to any backend-free index
        // (lets MultiLevelClient serve `batch_len > s` while protocol::Client rejects it).
        let mut preferred: Option<usize> = None;
        let mut preferred_count: usize = 0;
        let mut fallback: Option<usize> = None;
        let mut fallback_count: usize = 0;
        for cand in 0..s {
            if backend.is_index_cached(cand) || backend.is_index_in_flight(cand) {
                continue;
            }
            if !avoid.contains(&cand) {
                preferred_count += 1;
                if rng.gen_range(0..preferred_count) == 0 {
                    preferred = Some(cand);
                }
            } else if preferred.is_none() {
                fallback_count += 1;
                if rng.gen_range(0..fallback_count) == 0 {
                    fallback = Some(cand);
                }
            }
        }
        preferred
            .or(fallback)
            .ok_or(KeywordClientError::NoDecoyIndexAvailable { segment: seg })
    }

    /// Prepares a private lookup for `(kind, key)`.
    ///
    /// The key never leaves the client: only the two index PIR queries (for
    /// `bucket_seg0` in segment 0 and `bucket_seg1` in segment 1, or decoys under
    /// [`CachePolicy::ConstantTraffic`]) appear in the returned wire query.
    ///
    /// If preparing segment 1 fails after segment 0 succeeds, segment 0's unsent
    /// reservation is rolled back via [`IndexPirClient::cancel_unsent_index_query`]
    /// so no hint is leaked.
    pub fn prepare_lookup<R: Rng + ?Sized>(
        &mut self,
        kind: u8,
        key: &[u8],
        rng: &mut R,
    ) -> PrepareLookupResult<B> {
        let loc = self.spec.locate(kind, key)?;
        let empty_avoid: HashSet<usize> = HashSet::new();

        match self.cache_policy {
            CachePolicy::LocalHit => {
                if let Some(val) = self.stash_lookup(loc.bucket_seg0, &loc.tag) {
                    return Ok(LookupPlan::Local(Some(val.to_vec())));
                }
                if let (Some(b0), Some(b1)) = (
                    self.bucket_cache[0].get(&loc.bucket_seg0),
                    self.bucket_cache[1].get(&loc.bucket_seg1),
                ) {
                    let found = self
                        .spec
                        .scan_buckets(b0, b1, &loc.tag, None)?
                        .map(ToOwned::to_owned);
                    return Ok(LookupPlan::Local(found));
                }
                let need0: bool = !self.bucket_cache[0].contains_key(&loc.bucket_seg0);
                let need1: bool = !self.bucket_cache[1].contains_key(&loc.bucket_seg1);
                let idx0: usize = if need0 {
                    loc.bucket_seg0
                } else {
                    self.pick_decoy_index(0, &empty_avoid, rng)?
                };
                let idx1: usize = if need1 {
                    loc.bucket_seg1
                } else {
                    self.pick_decoy_index(1, &empty_avoid, rng)?
                };
                self.prepare_two_segment_query(
                    idx0,
                    idx1,
                    LookupResolution::Buckets {
                        bucket_seg0: loc.bucket_seg0,
                        bucket_seg1: loc.bucket_seg1,
                        tag: loc.tag,
                        wire_is_real: [need0, need1],
                    },
                    rng,
                )
            }
            CachePolicy::ConstantTraffic => {
                if let Some(val) = self.stash_lookup(loc.bucket_seg0, &loc.tag) {
                    let val_owned: Vec<u8> = val.to_vec();
                    let idx0: usize = self.pick_decoy_index(0, &empty_avoid, rng)?;
                    let idx1: usize = self.pick_decoy_index(1, &empty_avoid, rng)?;
                    return self.prepare_two_segment_query(
                        idx0,
                        idx1,
                        LookupResolution::StashHit(val_owned),
                        rng,
                    );
                }
                let need0: bool = !self.bucket_cache[0].contains_key(&loc.bucket_seg0);
                let need1: bool = !self.bucket_cache[1].contains_key(&loc.bucket_seg1);
                let idx0: usize = if need0 {
                    loc.bucket_seg0
                } else {
                    self.pick_decoy_index(0, &empty_avoid, rng)?
                };
                let idx1: usize = if need1 {
                    loc.bucket_seg1
                } else {
                    self.pick_decoy_index(1, &empty_avoid, rng)?
                };
                self.prepare_two_segment_query(
                    idx0,
                    idx1,
                    LookupResolution::Buckets {
                        bucket_seg0: loc.bucket_seg0,
                        bucket_seg1: loc.bucket_seg1,
                        tag: loc.tag,
                        wire_is_real: [need0, need1],
                    },
                    rng,
                )
            }
            CachePolicy::Off => self.prepare_two_segment_query(
                loc.bucket_seg0,
                loc.bucket_seg1,
                LookupResolution::Buckets {
                    bucket_seg0: loc.bucket_seg0,
                    bucket_seg1: loc.bucket_seg1,
                    tag: loc.tag,
                    wire_is_real: [true, true],
                },
                rng,
            ),
        }
    }

    fn prepare_two_segment_query<R: Rng + ?Sized>(
        &mut self,
        idx0: usize,
        idx1: usize,
        resolution: LookupResolution,
        rng: &mut R,
    ) -> PrepareLookupResult<B> {
        let p0: B::Pending = self.backends[0]
            .prepare_index_query(idx0, rng)
            .map_err(|error| KeywordClientError::Backend { segment: 0, error })?;
        let p1: B::Pending = match self.backends[1].prepare_index_query(idx1, rng) {
            Ok(p) => p,
            Err(error) => {
                let _ = self.backends[0].cancel_unsent_index_query(p0);
                return Err(KeywordClientError::Backend { segment: 1, error });
            }
        };
        let query: KeywordLookupQuery<B::Query> = KeywordLookupQuery {
            table_epoch: self.spec.table_epoch(),
            state_epoch: self.spec.state_epoch(),
            segments: [B::pending_query(&p0).clone(), B::pending_query(&p1).clone()],
        };
        let hint_ids: [B::HintId; NUM_SEGMENTS] =
            [B::pending_hint_id(&p0), B::pending_hint_id(&p1)];
        let pending: PendingKeywordLookup<B> = PendingKeywordLookup {
            session_id: self.session_id,
            table_epoch: self.spec.table_epoch(),
            state_epoch: self.spec.state_epoch(),
            resolution,
            hint_ids,
            segments: [Some(p0), Some(p1)],
        };
        Ok(LookupPlan::Query { query, pending })
    }

    /// Finishes an in-flight single keyword lookup from the server's answer.
    ///
    /// Validates `table_epoch`, `state_epoch`, session binding, and both segment
    /// backend answers **before** mutating either backend. On a validation error
    /// (such as a stale server snapshot epoch), neither backend is mutated and the
    /// caller retains `pending` to retry or call [`KeywordClient::abort_lookup`].
    pub fn finish_lookup<R: Rng + ?Sized>(
        &mut self,
        pending: &PendingKeywordLookup<B>,
        answer: &KeywordLookupAnswer<B::Answer>,
        rng: &mut R,
    ) -> Result<Option<Vec<u8>>, KeywordClientError<B::Error>> {
        if pending.session_id != self.session_id {
            return Err(KeywordClientError::SessionMismatch);
        }
        if pending.table_epoch != self.spec.table_epoch() {
            return Err(KeywordClientError::TableEpochMismatch {
                expected: self.spec.table_epoch(),
                got: pending.table_epoch,
            });
        }
        if answer.table_epoch != self.spec.table_epoch() {
            return Err(KeywordClientError::TableEpochMismatch {
                expected: self.spec.table_epoch(),
                got: answer.table_epoch,
            });
        }
        if pending.state_epoch != self.spec.state_epoch() {
            return Err(KeywordClientError::StateEpochMismatch {
                expected: self.spec.state_epoch(),
                got: pending.state_epoch,
            });
        }
        if answer.state_epoch != self.spec.state_epoch() {
            return Err(KeywordClientError::StateEpochMismatch {
                expected: self.spec.state_epoch(),
                got: answer.state_epoch,
            });
        }
        let Some(p0) = pending.segments[0].as_ref() else {
            return Err(KeywordClientError::SessionMismatch);
        };
        let Some(p1) = pending.segments[1].as_ref() else {
            return Err(KeywordClientError::SessionMismatch);
        };
        // Plan and validate both segments (including all refresh draws) before committing either.
        let plan0: B::FinishPlan = self.backends[0]
            .plan_finish_batch(&[p0], std::slice::from_ref(&answer.segments[0]), rng)
            .map_err(|error| KeywordClientError::Backend { segment: 0, error })?;
        let plan1: B::FinishPlan = self.backends[1]
            .plan_finish_batch(&[p1], std::slice::from_ref(&answer.segments[1]), rng)
            .map_err(|error| KeywordClientError::Backend { segment: 1, error })?;

        // Validate reconstructed bucket tags BEFORE committing either backend or writing to
        // `bucket_cache`, so `DuplicateTagMatch` leaves backend state and cache unchanged and
        // `pending` remains usable for `abort_lookup`.
        let found: Option<Vec<u8>> = {
            let vals0: &[Vec<u8>] = B::plan_finish_values(&plan0);
            let vals1: &[Vec<u8>] = B::plan_finish_values(&plan1);
            if vals0.len() != 1 {
                return Err(KeywordClientError::AnswerCountMismatch {
                    segment: 0,
                    expected: 1,
                    actual: vals0.len(),
                });
            }
            if vals1.len() != 1 {
                return Err(KeywordClientError::AnswerCountMismatch {
                    segment: 1,
                    expected: 1,
                    actual: vals1.len(),
                });
            }
            match &pending.resolution {
                LookupResolution::StashHit(val) => Some(val.clone()),
                LookupResolution::Buckets {
                    bucket_seg0,
                    bucket_seg1,
                    tag,
                    wire_is_real,
                } => {
                    let b0: &[u8] = if wire_is_real[0] {
                        &vals0[0]
                    } else {
                        self.bucket_cache[0]
                            .get(bucket_seg0)
                            .map(Vec::as_slice)
                            .ok_or(KeywordClientError::StateEpochMismatch {
                                expected: self.spec.state_epoch(),
                                got: pending.state_epoch,
                            })?
                    };
                    let b1: &[u8] = if wire_is_real[1] {
                        &vals1[0]
                    } else {
                        self.bucket_cache[1]
                            .get(bucket_seg1)
                            .map(Vec::as_slice)
                            .ok_or(KeywordClientError::StateEpochMismatch {
                                expected: self.spec.state_epoch(),
                                got: pending.state_epoch,
                            })?
                    };
                    let stash_val: Option<&[u8]> = self.stash_lookup(*bucket_seg0, tag);
                    self.spec
                        .scan_buckets(b0, b1, tag, stash_val)?
                        .map(ToOwned::to_owned)
                }
            }
        };

        // All scans succeeded; commit both backends and publish real wire buckets to the cache.
        let b0_wire: Vec<u8> = self.backends[0]
            .commit_finish_batch(plan0)
            .pop()
            .expect("single-query finish plan returns one record");
        let b1_wire: Vec<u8> = self.backends[1]
            .commit_finish_batch(plan1)
            .pop()
            .expect("single-query finish plan returns one record");

        if self.cache_policy != CachePolicy::Off {
            if let LookupResolution::Buckets {
                bucket_seg0,
                bucket_seg1,
                wire_is_real,
                ..
            } = &pending.resolution
            {
                if wire_is_real[0] {
                    self.bucket_cache[0].insert(*bucket_seg0, b0_wire);
                }
                if wire_is_real[1] {
                    self.bucket_cache[1].insert(*bucket_seg1, b1_wire);
                }
            }
        }

        Ok(found)
    }

    /// Aborts an in-flight single keyword lookup, permanently retiring both segment
    /// hints so neither exposed hint can be reused.
    ///
    /// Borrows `pending` mutably so a [`KeywordClientError::SessionMismatch`] leaves
    /// `pending` intact for the caller to abort on its owning client.
    pub fn abort_lookup(
        &mut self,
        pending: &mut PendingKeywordLookup<B>,
    ) -> Result<(), KeywordClientError<B::Error>> {
        if pending.session_id != self.session_id {
            return Err(KeywordClientError::SessionMismatch);
        }
        let r0 = pending.segments[0]
            .take()
            .map(|p0| self.backends[0].abort_index_query(p0));
        let r1 = pending.segments[1]
            .take()
            .map(|p1| self.backends[1].abort_index_query(p1));
        if r0.is_none() && r1.is_none() {
            return Err(KeywordClientError::SessionMismatch);
        }
        if let Some(Err(error)) = r0 {
            return Err(KeywordClientError::Backend { segment: 0, error });
        }
        if let Some(Err(error)) = r1 {
            return Err(KeywordClientError::Backend { segment: 1, error });
        }
        Ok(())
    }

    /// Prepares a 1-RTT batch of keyword lookups for `keys` (`&[(kind, key)]`).
    ///
    /// Uses two-pass per-segment scheduling under [`CachePolicy::Off`] and
    /// [`CachePolicy::ConstantTraffic`] (Pass 1: uncached real buckets; Pass 2:
    /// backend-cached buckets and decoys for duplicates/shared buckets/cache hits)
    /// so internal backend decoys never collide with real batch buckets and every
    /// input key emits exactly one query pair on the wire.
    ///
    /// On any error (including insufficient decoys or backup hints), rolls back all
    /// tentative reservations cleanly via [`IndexPirClient::cancel_unsent_index_query`].
    pub fn prepare_batch<R: Rng + ?Sized>(
        &mut self,
        keys: &[(u8, &[u8])],
        rng: &mut R,
    ) -> PrepareBatchResult<B> {
        if keys.is_empty() {
            return Ok(BatchLookupPlan::Local(Vec::new()));
        }

        let locs = keys
            .iter()
            .map(|&(kind, key)| self.spec.locate(kind, key))
            .collect::<Result<Vec<_>, _>>()?;

        match self.cache_policy {
            CachePolicy::LocalHit => self.prepare_batch_local_hit(&locs, rng),
            CachePolicy::Off | CachePolicy::ConstantTraffic => {
                self.prepare_batch_constant_pair(&locs, rng)
            }
        }
    }

    fn prepare_batch_local_hit<R: Rng + ?Sized>(
        &mut self,
        locs: &[super::KeyLocation],
        rng: &mut R,
    ) -> PrepareBatchResult<B> {
        let mut entries: Vec<BatchEntryPlan> = Vec::with_capacity(locs.len());
        let mut seg_needed: [Vec<usize>; NUM_SEGMENTS] = [Vec::new(), Vec::new()];
        let mut seg_slot_map: [HashMap<usize, usize>; NUM_SEGMENTS] =
            [HashMap::new(), HashMap::new()];
        let mut all_local: bool = true;

        for loc in locs {
            if let Some(val) = self.stash_lookup(loc.bucket_seg0, &loc.tag) {
                entries.push(BatchEntryPlan::Local(Some(val.to_vec())));
                continue;
            }
            if let (Some(b0), Some(b1)) = (
                self.bucket_cache[0].get(&loc.bucket_seg0),
                self.bucket_cache[1].get(&loc.bucket_seg1),
            ) {
                let found: Option<Vec<u8>> = self
                    .spec
                    .scan_buckets(b0, b1, &loc.tag, None)?
                    .map(ToOwned::to_owned);
                entries.push(BatchEntryPlan::Local(found));
                continue;
            }

            all_local = false;
            let source0: BatchBucketSource = if self.bucket_cache[0].contains_key(&loc.bucket_seg0)
            {
                BatchBucketSource::Cached(loc.bucket_seg0)
            } else {
                let wire_idx: usize =
                    *seg_slot_map[0].entry(loc.bucket_seg0).or_insert_with(|| {
                        let idx: usize = seg_needed[0].len();
                        seg_needed[0].push(loc.bucket_seg0);
                        idx
                    });
                BatchBucketSource::WireSlot(wire_idx)
            };
            let source1: BatchBucketSource = if self.bucket_cache[1].contains_key(&loc.bucket_seg1)
            {
                BatchBucketSource::Cached(loc.bucket_seg1)
            } else {
                let wire_idx: usize =
                    *seg_slot_map[1].entry(loc.bucket_seg1).or_insert_with(|| {
                        let idx: usize = seg_needed[1].len();
                        seg_needed[1].push(loc.bucket_seg1);
                        idx
                    });
                BatchBucketSource::WireSlot(wire_idx)
            };

            entries.push(BatchEntryPlan::Scan {
                bucket_seg0: loc.bucket_seg0,
                tag: loc.tag.clone(),
                source0,
                source1,
            });
        }

        if all_local {
            let local_vals: Vec<Option<Vec<u8>>> = entries
                .into_iter()
                .map(|e| match e {
                    BatchEntryPlan::Local(v) => v,
                    BatchEntryPlan::Scan { .. } => unreachable!("all_local invariant"),
                })
                .collect();
            return Ok(BatchLookupPlan::Local(local_vals));
        }

        let mut prepared: [Vec<Option<B::Pending>>; NUM_SEGMENTS] = [
            (0..seg_needed[0].len()).map(|_| None).collect(),
            (0..seg_needed[1].len()).map(|_| None).collect(),
        ];

        for seg in 0..NUM_SEGMENTS {
            // Pass 1: uncached real buckets first so internal backend decoys in Pass 2
            // cannot collide with any real bucket needed by this batch.
            for (slot_idx, &bucket_idx) in seg_needed[seg].iter().enumerate() {
                if !self.backends[seg].is_index_cached(bucket_idx) {
                    match self.backends[seg].prepare_index_query(bucket_idx, rng) {
                        Ok(p) => prepared[seg][slot_idx] = Some(p),
                        Err(error) => {
                            self.rollback_prepared(&mut prepared);
                            return Err(KeywordClientError::Backend {
                                segment: seg,
                                error,
                            });
                        }
                    }
                }
            }
            // Pass 2: backend-cached buckets.
            for (slot_idx, &bucket_idx) in seg_needed[seg].iter().enumerate() {
                if prepared[seg][slot_idx].is_none() {
                    match self.backends[seg].prepare_index_query(bucket_idx, rng) {
                        Ok(p) => prepared[seg][slot_idx] = Some(p),
                        Err(error) => {
                            self.rollback_prepared(&mut prepared);
                            return Err(KeywordClientError::Backend {
                                segment: seg,
                                error,
                            });
                        }
                    }
                }
            }
        }

        let wire_real_buckets: [Vec<Option<usize>>; NUM_SEGMENTS] = [
            seg_needed[0].iter().copied().map(Some).collect(),
            seg_needed[1].iter().copied().map(Some).collect(),
        ];
        Ok(self.finalize_batch_plan(prepared, wire_real_buckets, entries))
    }

    fn prepare_batch_constant_pair<R: Rng + ?Sized>(
        &mut self,
        locs: &[super::KeyLocation],
        rng: &mut R,
    ) -> PrepareBatchResult<B> {
        let n: usize = locs.len();
        let use_kw_cache: bool = self.cache_policy == CachePolicy::ConstantTraffic;

        #[derive(Clone, Copy)]
        enum SlotTask {
            RealUncached(usize),
            RealBackendCached(usize),
            Decoy,
        }

        let mut entries: Vec<BatchEntryPlan> = Vec::with_capacity(n);
        let mut tasks: [Vec<SlotTask>; NUM_SEGMENTS] =
            [vec![SlotTask::Decoy; n], vec![SlotTask::Decoy; n]];
        let mut wire_real_buckets: [Vec<Option<usize>>; NUM_SEGMENTS] =
            [vec![None; n], vec![None; n]];
        let mut first_real_slot: [HashMap<usize, usize>; NUM_SEGMENTS] =
            [HashMap::new(), HashMap::new()];

        for (i, loc) in locs.iter().enumerate() {
            if use_kw_cache {
                if let Some(val) = self.stash_lookup(loc.bucket_seg0, &loc.tag) {
                    entries.push(BatchEntryPlan::Local(Some(val.to_vec())));
                    continue;
                }
            }

            let mut sources: [BatchBucketSource; NUM_SEGMENTS] =
                [BatchBucketSource::WireSlot(0); NUM_SEGMENTS];
            let buckets: [usize; NUM_SEGMENTS] = [loc.bucket_seg0, loc.bucket_seg1];

            for seg in 0..NUM_SEGMENTS {
                let b_idx: usize = buckets[seg];
                if use_kw_cache && self.bucket_cache[seg].contains_key(&b_idx) {
                    sources[seg] = BatchBucketSource::Cached(b_idx);
                } else if let Some(&existing_slot) = first_real_slot[seg].get(&b_idx) {
                    sources[seg] = BatchBucketSource::WireSlot(existing_slot);
                } else {
                    first_real_slot[seg].insert(b_idx, i);
                    sources[seg] = BatchBucketSource::WireSlot(i);
                    wire_real_buckets[seg][i] = Some(b_idx);
                    tasks[seg][i] = if self.backends[seg].is_index_cached(b_idx) {
                        SlotTask::RealBackendCached(b_idx)
                    } else {
                        SlotTask::RealUncached(b_idx)
                    };
                }
            }

            entries.push(BatchEntryPlan::Scan {
                bucket_seg0: loc.bucket_seg0,
                tag: loc.tag.clone(),
                source0: sources[0],
                source1: sources[1],
            });
        }

        let mut prepared: [Vec<Option<B::Pending>>; NUM_SEGMENTS] = [
            (0..n).map(|_| None).collect(),
            (0..n).map(|_| None).collect(),
        ];

        for seg in 0..NUM_SEGMENTS {
            let mut used_wire_indices: HashSet<usize> = HashSet::with_capacity(n);

            // Pass 1: schedule every distinct real bucket not cached in `backend[seg]`.
            for i in 0..n {
                if let SlotTask::RealUncached(b_idx) = tasks[seg][i] {
                    match self.backends[seg].prepare_index_query(b_idx, rng) {
                        Ok(p) => {
                            used_wire_indices.insert(B::pending_queried_index(&p));
                            prepared[seg][i] = Some(p);
                        }
                        Err(error) => {
                            self.rollback_prepared(&mut prepared);
                            return Err(KeywordClientError::Backend {
                                segment: seg,
                                error,
                            });
                        }
                    }
                }
            }

            // Pass 2a: schedule backend-cached real buckets (internal backend decoys
            // cannot collide with Pass 1 indices because Pass 1 is already in flight).
            for i in 0..n {
                if let SlotTask::RealBackendCached(b_idx) = tasks[seg][i] {
                    match self.backends[seg].prepare_index_query(b_idx, rng) {
                        Ok(p) => {
                            used_wire_indices.insert(B::pending_queried_index(&p));
                            prepared[seg][i] = Some(p);
                        }
                        Err(error) => {
                            self.rollback_prepared(&mut prepared);
                            return Err(KeywordClientError::Backend {
                                segment: seg,
                                error,
                            });
                        }
                    }
                }
            }

            // Pass 2b: schedule decoy queries for duplicate keys, shared buckets,
            // keyword-cached buckets, and stash hits.
            for i in 0..n {
                if matches!(tasks[seg][i], SlotTask::Decoy) {
                    let decoy_idx: usize = match self.pick_decoy_index(seg, &used_wire_indices, rng)
                    {
                        Ok(idx) => idx,
                        Err(err) => {
                            self.rollback_prepared(&mut prepared);
                            return Err(err);
                        }
                    };
                    match self.backends[seg].prepare_index_query(decoy_idx, rng) {
                        Ok(p) => {
                            used_wire_indices.insert(B::pending_queried_index(&p));
                            prepared[seg][i] = Some(p);
                        }
                        Err(error) => {
                            self.rollback_prepared(&mut prepared);
                            return Err(KeywordClientError::Backend {
                                segment: seg,
                                error,
                            });
                        }
                    }
                }
            }
        }

        Ok(self.finalize_batch_plan(prepared, wire_real_buckets, entries))
    }

    fn rollback_prepared(&mut self, prepared: &mut [Vec<Option<B::Pending>>; NUM_SEGMENTS]) {
        for (seg, slots) in prepared.iter_mut().enumerate() {
            for slot in slots.iter_mut() {
                if let Some(p) = slot.take() {
                    let _ = self.backends[seg].cancel_unsent_index_query(p);
                }
            }
        }
    }

    fn finalize_batch_plan(
        &self,
        prepared: [Vec<Option<B::Pending>>; NUM_SEGMENTS],
        wire_real_buckets: [Vec<Option<usize>>; NUM_SEGMENTS],
        entries: Vec<BatchEntryPlan>,
    ) -> BatchLookupPlan<B::Query, PendingBatchLookup<B>> {
        let [p0_opt, p1_opt] = prepared;
        let seg0_queries: Vec<B::Query> = p0_opt
            .iter()
            .map(|o| B::pending_query(o.as_ref().expect("all batch slots prepared")).clone())
            .collect();
        let seg1_queries: Vec<B::Query> = p1_opt
            .iter()
            .map(|o| B::pending_query(o.as_ref().expect("all batch slots prepared")).clone())
            .collect();
        let seg0_hint_ids: Vec<B::HintId> = p0_opt
            .iter()
            .map(|o| B::pending_hint_id(o.as_ref().expect("all batch slots prepared")))
            .collect();
        let seg1_hint_ids: Vec<B::HintId> = p1_opt
            .iter()
            .map(|o| B::pending_hint_id(o.as_ref().expect("all batch slots prepared")))
            .collect();
        let query: KeywordBatchQuery<B::Query> = KeywordBatchQuery {
            table_epoch: self.spec.table_epoch(),
            state_epoch: self.spec.state_epoch(),
            segments: [seg0_queries, seg1_queries],
        };
        let pending: PendingBatchLookup<B> = PendingBatchLookup {
            session_id: self.session_id,
            table_epoch: self.spec.table_epoch(),
            state_epoch: self.spec.state_epoch(),
            hint_ids: [seg0_hint_ids, seg1_hint_ids],
            segments: [p0_opt, p1_opt],
            wire_real_buckets,
            entries,
        };
        BatchLookupPlan::Query { query, pending }
    }

    /// Finishes an in-flight batch lookup from the server's batch answer.
    ///
    /// Validates `table_epoch`, `state_epoch`, session binding, answer counts, and
    /// every segment backend answer **before** mutating either backend.
    pub fn finish_batch<R: Rng + ?Sized>(
        &mut self,
        pending: &PendingBatchLookup<B>,
        answer: &KeywordBatchAnswer<B::Answer>,
        rng: &mut R,
    ) -> Result<Vec<Option<Vec<u8>>>, KeywordClientError<B::Error>> {
        if pending.session_id != self.session_id {
            return Err(KeywordClientError::SessionMismatch);
        }
        if pending.table_epoch != self.spec.table_epoch() {
            return Err(KeywordClientError::TableEpochMismatch {
                expected: self.spec.table_epoch(),
                got: pending.table_epoch,
            });
        }
        if answer.table_epoch != self.spec.table_epoch() {
            return Err(KeywordClientError::TableEpochMismatch {
                expected: self.spec.table_epoch(),
                got: answer.table_epoch,
            });
        }
        if pending.state_epoch != self.spec.state_epoch() {
            return Err(KeywordClientError::StateEpochMismatch {
                expected: self.spec.state_epoch(),
                got: pending.state_epoch,
            });
        }
        if answer.state_epoch != self.spec.state_epoch() {
            return Err(KeywordClientError::StateEpochMismatch {
                expected: self.spec.state_epoch(),
                got: answer.state_epoch,
            });
        }

        let mut seg_refs: [Vec<&B::Pending>; NUM_SEGMENTS] = [
            Vec::with_capacity(pending.segments[0].len()),
            Vec::with_capacity(pending.segments[1].len()),
        ];
        for (seg, refs) in seg_refs.iter_mut().enumerate() {
            if answer.segments[seg].len() != pending.segments[seg].len() {
                return Err(KeywordClientError::AnswerCountMismatch {
                    segment: seg,
                    expected: pending.segments[seg].len(),
                    actual: answer.segments[seg].len(),
                });
            }
            for slot in &pending.segments[seg] {
                let Some(p) = slot.as_ref() else {
                    return Err(KeywordClientError::SessionMismatch);
                };
                refs.push(p);
            }
        }

        // Plan both segments against staged state before committing either.
        let plan0: B::FinishPlan = self.backends[0]
            .plan_finish_batch(&seg_refs[0], &answer.segments[0], rng)
            .map_err(|error| KeywordClientError::Backend { segment: 0, error })?;
        let plan1: B::FinishPlan = self.backends[1]
            .plan_finish_batch(&seg_refs[1], &answer.segments[1], rng)
            .map_err(|error| KeywordClientError::Backend { segment: 1, error })?;

        // Validate every batch entry's reconstructed bucket tags BEFORE committing either backend
        // or writing to `bucket_cache`, so `DuplicateTagMatch` leaves backend state and cache
        // unchanged and `pending` remains usable for `abort_batch`.
        let out: Vec<Option<Vec<u8>>> = {
            let staged_vals: [&[Vec<u8>]; NUM_SEGMENTS] =
                [B::plan_finish_values(&plan0), B::plan_finish_values(&plan1)];
            for (seg, vals) in staged_vals.iter().enumerate() {
                if vals.len() != seg_refs[seg].len() {
                    return Err(KeywordClientError::AnswerCountMismatch {
                        segment: seg,
                        expected: seg_refs[seg].len(),
                        actual: vals.len(),
                    });
                }
            }
            let mut out: Vec<Option<Vec<u8>>> = Vec::with_capacity(pending.entries.len());
            for entry in &pending.entries {
                match entry {
                    BatchEntryPlan::Local(val) => out.push(val.clone()),
                    BatchEntryPlan::Scan {
                        bucket_seg0,
                        tag,
                        source0,
                        source1,
                    } => {
                        let b0: &[u8] = match source0 {
                            BatchBucketSource::WireSlot(idx) => &staged_vals[0][*idx],
                            BatchBucketSource::Cached(b_idx) => self.bucket_cache[0]
                                .get(b_idx)
                                .map(Vec::as_slice)
                                .ok_or(KeywordClientError::StateEpochMismatch {
                                    expected: self.spec.state_epoch(),
                                    got: pending.state_epoch,
                                })?,
                        };
                        let b1: &[u8] = match source1 {
                            BatchBucketSource::WireSlot(idx) => &staged_vals[1][*idx],
                            BatchBucketSource::Cached(b_idx) => self.bucket_cache[1]
                                .get(b_idx)
                                .map(Vec::as_slice)
                                .ok_or(KeywordClientError::StateEpochMismatch {
                                    expected: self.spec.state_epoch(),
                                    got: pending.state_epoch,
                                })?,
                        };
                        let stash_val: Option<&[u8]> = self.stash_lookup(*bucket_seg0, tag);
                        let val: Option<Vec<u8>> = self
                            .spec
                            .scan_buckets(b0, b1, tag, stash_val)?
                            .map(ToOwned::to_owned);
                        out.push(val);
                    }
                }
            }
            out
        };

        // All scans succeeded; commit both backends and publish real wire buckets to the cache.
        let recovered: [Vec<Vec<u8>>; NUM_SEGMENTS] = [
            self.backends[0].commit_finish_batch(plan0),
            self.backends[1].commit_finish_batch(plan1),
        ];

        if self.cache_policy != CachePolicy::Off {
            for (seg, seg_recovered) in recovered.into_iter().enumerate() {
                for (bucket_bytes, maybe_bucket) in seg_recovered
                    .into_iter()
                    .zip(&pending.wire_real_buckets[seg])
                {
                    if let Some(bucket_idx) = maybe_bucket {
                        self.bucket_cache[seg].insert(*bucket_idx, bucket_bytes);
                    }
                }
            }
        }

        Ok(out)
    }

    /// Aborts an in-flight batch lookup, permanently retiring every segment hint in
    /// the batch so no exposed hint can be reused.
    ///
    /// Borrows `pending` mutably so a [`KeywordClientError::SessionMismatch`] leaves
    /// `pending` intact for the caller to abort on its owning client.
    pub fn abort_batch(
        &mut self,
        pending: &mut PendingBatchLookup<B>,
    ) -> Result<(), KeywordClientError<B::Error>> {
        if pending.session_id != self.session_id {
            return Err(KeywordClientError::SessionMismatch);
        }
        let mut any_taken: bool = false;
        let mut first_err: Option<KeywordClientError<B::Error>> = None;
        for seg in 0..NUM_SEGMENTS {
            for slot in &mut pending.segments[seg] {
                if let Some(p) = slot.take() {
                    any_taken = true;
                    if let Err(error) = self.backends[seg].abort_index_query(p) {
                        if first_err.is_none() {
                            first_err = Some(KeywordClientError::Backend {
                                segment: seg,
                                error,
                            });
                        }
                    }
                }
            }
        }
        if !any_taken {
            return Err(KeywordClientError::SessionMismatch);
        }
        match first_err {
            Some(err) => Err(err),
            None => Ok(()),
        }
    }

    /// Applies a synchronized [`KeywordEpochUpdate`] atomically across both segment
    /// backends, the keyword bucket cache, and the public runtime-stash.
    ///
    /// Validates `table_epoch`, `from_state_epoch`, `to_state_epoch > from_state_epoch`,
    /// all delta bounds/lengths, and all `stash_ops` before mutating either backend.
    /// An empty `segment_deltas[j]` is skipped so `backend[j].epoch()` remains unchanged.
    pub fn apply_epoch_update(
        &mut self,
        update: &KeywordEpochUpdate,
        threads: usize,
    ) -> Result<(), KeywordClientError<B::Error>> {
        if update.table_epoch != self.spec.table_epoch() {
            return Err(KeywordClientError::TableEpochMismatch {
                expected: self.spec.table_epoch(),
                got: update.table_epoch,
            });
        }
        if update.from_state_epoch != self.spec.state_epoch() {
            return Err(KeywordClientError::StateEpochMismatch {
                expected: self.spec.state_epoch(),
                got: update.from_state_epoch,
            });
        }
        if update.to_state_epoch <= update.from_state_epoch {
            return Err(KeywordClientError::InvalidEpochUpdate(
                "to_state_epoch must be strictly greater than from_state_epoch",
            ));
        }

        let s: usize = self.spec.buckets_per_segment();
        let rec: usize = self.spec.record_size();
        for deltas in &update.segment_deltas {
            for (bucket_idx, xor_delta) in deltas {
                if *bucket_idx >= s {
                    return Err(KeywordClientError::InvalidEpochUpdate(
                        "bucket_idx out of range for segment",
                    ));
                }
                if xor_delta.len() != rec {
                    return Err(KeywordClientError::InvalidEpochUpdate(
                        "xor_delta length does not match record_size",
                    ));
                }
            }
        }

        // Stage and validate all stash ops before touching either PIR backend.
        let mut staged_stash: HashMap<(usize, Vec<u8>), Vec<u8>> = self.stash.clone();
        for op in &update.stash_ops {
            match op {
                StashOp::Insert(entry) => {
                    validate_stash_fields(
                        &self.spec,
                        entry.bucket_seg0,
                        &entry.tag,
                        Some(&entry.value),
                    )?;
                    let key: (usize, Vec<u8>) = (entry.bucket_seg0, entry.tag.clone());
                    if staged_stash.contains_key(&key) {
                        return Err(KeywordClientError::InvalidEpochUpdate(
                            "duplicate StashOp::Insert for existing (bucket_seg0, tag)",
                        ));
                    }
                    staged_stash.insert(key, entry.value.clone());
                }
                StashOp::Update {
                    bucket_seg0,
                    tag,
                    value,
                } => {
                    validate_stash_fields(&self.spec, *bucket_seg0, tag, Some(value))?;
                    let key: (usize, Vec<u8>) = (*bucket_seg0, tag.clone());
                    let Some(slot) = staged_stash.get_mut(&key) else {
                        return Err(KeywordClientError::InvalidEpochUpdate(
                            "StashOp::Update target not present in stash",
                        ));
                    };
                    *slot = value.clone();
                }
                StashOp::Delete { bucket_seg0, tag } => {
                    validate_stash_fields(&self.spec, *bucket_seg0, tag, None)?;
                    let key: (usize, Vec<u8>) = (*bucket_seg0, tag.clone());
                    if staged_stash.remove(&key).is_none() {
                        return Err(KeywordClientError::InvalidEpochUpdate(
                            "StashOp::Delete target not present in stash",
                        ));
                    }
                }
            }
        }

        // Validate and stage both segment backends before committing either so a
        // segment-1 rejection can never leave segment 0 advanced.
        let plan0: B::DeltaPlan = self.backends[0]
            .prepare_index_deltas(&update.segment_deltas[0])
            .map_err(|error| KeywordClientError::Backend { segment: 0, error })?;
        let plan1: B::DeltaPlan = self.backends[1]
            .prepare_index_deltas(&update.segment_deltas[1])
            .map_err(|error| KeywordClientError::Backend { segment: 1, error })?;

        // Commit both staged backend delta plans infallibly.
        self.backends[0].commit_index_deltas(plan0, threads);
        self.backends[1].commit_index_deltas(plan1, threads);

        // Maintain keyword bucket cache via XOR in place.
        if self.cache_policy != CachePolicy::Off {
            for seg in 0..NUM_SEGMENTS {
                for (bucket_idx, xor_delta) in &update.segment_deltas[seg] {
                    if let Some(cached) = self.bucket_cache[seg].get_mut(bucket_idx) {
                        xor_bytes_in_place(cached, xor_delta);
                    }
                }
            }
        }

        self.stash = staged_stash;
        self.spec.state_epoch = update.to_state_epoch;
        Ok(())
    }

    /// Retires WAL-logged hint identifiers across both segment backends after
    /// restoring a checkpoint so no previously transmitted hint can be reused.
    ///
    /// Validates both lists before mutating either backend.
    pub fn retire_hints(
        &mut self,
        seg0_hints: &[B::HintId],
        seg1_hints: &[B::HintId],
    ) -> Result<(), KeywordClientError<B::Error>> {
        self.backends[0]
            .validate_retire_hints(seg0_hints)
            .map_err(|error| KeywordClientError::Backend { segment: 0, error })?;
        self.backends[1]
            .validate_retire_hints(seg1_hints)
            .map_err(|error| KeywordClientError::Backend { segment: 1, error })?;
        self.backends[0]
            .retire_index_hints(seg0_hints)
            .map_err(|error| KeywordClientError::Backend { segment: 0, error })?;
        self.backends[1]
            .retire_index_hints(seg1_hints)
            .map_err(|error| KeywordClientError::Backend { segment: 1, error })?;
        Ok(())
    }

    /// Serializes this [`KeywordClient`] (spec, epochs, cache policy, runtime-stash,
    /// keyword bucket cache, and both segment backend checkpoints) with a trailing
    /// 32-byte BLAKE3 checksum.
    ///
    /// Fails with [`KeywordClientError::QueryInFlight`] if any query is in flight.
    pub fn write_checkpoint_with<W, F>(
        &self,
        writer: W,
        mut write_backend: F,
    ) -> Result<(), KeywordClientError<B::Error>>
    where
        W: Write,
        F: FnMut(usize, &B, &mut Vec<u8>) -> Result<(), KeywordClientError<B::Error>>,
    {
        if self.queries_in_flight() > 0 {
            return Err(KeywordClientError::QueryInFlight);
        }

        let bucket_size_u32: u32 = u32::try_from(self.spec.bucket_size())
            .map_err(|_| KeywordClientError::Checkpoint("bucket_size exceeds u32::MAX"))?;
        let tag_size_u32: u32 = u32::try_from(self.spec.tag_size())
            .map_err(|_| KeywordClientError::Checkpoint("tag_size exceeds u32::MAX"))?;
        let value_size_u32: u32 = u32::try_from(self.spec.value_size())
            .map_err(|_| KeywordClientError::Checkpoint("value_size exceeds u32::MAX"))?;

        let mut seg_blobs: [Vec<u8>; NUM_SEGMENTS] = [Vec::new(), Vec::new()];
        for (seg, blob) in seg_blobs.iter_mut().enumerate() {
            write_backend(seg, &self.backends[seg], blob)?;
        }

        let mut out: HashingWriter<W> = HashingWriter::new(writer);
        out.write_hashed(&KEYWORD_CHECKPOINT_MAGIC)?;
        out.write_hashed(&KEYWORD_CHECKPOINT_VERSION.to_le_bytes())?;

        out.write_hashed(self.spec.table_seed())?;
        out.write_hashed(&self.spec.table_epoch().to_le_bytes())?;
        out.write_hashed(&self.spec.state_epoch().to_le_bytes())?;
        out.write_hashed(&(self.spec.buckets_per_segment() as u64).to_le_bytes())?;
        out.write_hashed(&bucket_size_u32.to_le_bytes())?;
        out.write_hashed(&tag_size_u32.to_le_bytes())?;
        out.write_hashed(&value_size_u32.to_le_bytes())?;
        let (has_key_size, key_size_val): (u8, u64) = match self.spec.key_size() {
            Some(k) => (1, k as u64),
            None => (0, 0),
        };
        out.write_hashed(&[has_key_size, self.cache_policy.to_byte(), 0, 0])?;
        out.write_hashed(&key_size_val.to_le_bytes())?;

        // Runtime-stash sorted deterministically by (bucket_seg0, tag).
        let mut stash_sorted: Vec<(usize, &[u8], &[u8])> = self
            .stash
            .iter()
            .map(|((a0, tag), val)| (*a0, tag.as_slice(), val.as_slice()))
            .collect();
        stash_sorted.sort_by_key(|&(a0, tag, _)| (a0, tag));
        let stash_len: u32 = u32::try_from(stash_sorted.len())
            .map_err(|_| KeywordClientError::Checkpoint("stash length exceeds u32::MAX"))?;
        out.write_hashed(&stash_len.to_le_bytes())?;
        for (bucket_seg0, tag, value) in stash_sorted {
            out.write_hashed(&(bucket_seg0 as u64).to_le_bytes())?;
            out.write_hashed(tag)?;
            out.write_hashed(value)?;
        }

        // Keyword bucket caches sorted deterministically by bucket_idx.
        for cache in &self.bucket_cache {
            let mut entries: Vec<(usize, &[u8])> =
                cache.iter().map(|(&idx, b)| (idx, b.as_slice())).collect();
            entries.sort_by_key(|&(idx, _)| idx);
            let cache_len: u32 = u32::try_from(entries.len())
                .map_err(|_| KeywordClientError::Checkpoint("bucket cache exceeds u32::MAX"))?;
            out.write_hashed(&cache_len.to_le_bytes())?;
            for (bucket_idx, bytes) in entries {
                out.write_hashed(&(bucket_idx as u64).to_le_bytes())?;
                out.write_hashed(bytes)?;
            }
        }

        // Length-prefixed backend blobs.
        for blob in &seg_blobs {
            out.write_hashed(&(blob.len() as u64).to_le_bytes())?;
            out.write_hashed(blob)?;
        }

        out.finish()?;
        Ok(())
    }

    /// Deserializes a [`KeywordClient`] checkpoint written by
    /// [`KeywordClient::write_checkpoint_with`], enforcing a bounded byte budget and
    /// verifying the trailing 32-byte BLAKE3 checksum.
    ///
    /// `read_backend` receives `(segment, blob, remaining_budget)` and returns
    /// `(decoded_backend, retained_memory_bytes)` so segment 0's retained memory is
    /// deducted before decoding segment 1.
    pub fn read_checkpoint_with_limit<R, F>(
        reader: R,
        max_bytes: usize,
        mut read_backend: F,
    ) -> Result<Self, KeywordClientError<B::Error>>
    where
        R: Read,
        F: FnMut(usize, &[u8], usize) -> Result<(B, usize), KeywordClientError<B::Error>>,
    {
        let mut remaining_budget: usize = max_bytes;
        let mut charge = |n: usize| -> Result<(), KeywordClientError<B::Error>> {
            if n > remaining_budget {
                return Err(KeywordClientError::Checkpoint(
                    "checkpoint exceeds memory budget",
                ));
            }
            remaining_budget -= n;
            Ok(())
        };

        let mut inp: HashingReader<R> = HashingReader::new(reader);
        let mut prefix: [u8; 12] = [0u8; 12];
        inp.read_hashed(&mut prefix)?;
        if prefix[0..8] != KEYWORD_CHECKPOINT_MAGIC {
            return Err(KeywordClientError::Checkpoint("bad checkpoint magic"));
        }
        let version: u32 = u32_at(&prefix, 8);
        if version != KEYWORD_CHECKPOINT_VERSION {
            return Err(KeywordClientError::Checkpoint(
                "unsupported checkpoint version",
            ));
        }

        let mut spec_buf: [u8; 80] = [0u8; 80];
        inp.read_hashed(&mut spec_buf)?;
        let mut table_seed: [u8; 32] = [0u8; 32];
        table_seed.copy_from_slice(&spec_buf[0..32]);
        let table_epoch: u64 = u64_at(&spec_buf, 32);
        let state_epoch: u64 = u64_at(&spec_buf, 40);
        let buckets_per_segment: usize = usize::try_from(u64_at(&spec_buf, 48))
            .map_err(|_| KeywordClientError::Checkpoint("buckets_per_segment overflow"))?;
        let bucket_size: usize = u32_at(&spec_buf, 56) as usize;
        let tag_size: usize = u32_at(&spec_buf, 60) as usize;
        let value_size: usize = u32_at(&spec_buf, 64) as usize;
        let has_key_size: u8 = spec_buf[68];
        let policy_byte: u8 = spec_buf[69];
        if spec_buf[70] != 0 || spec_buf[71] != 0 {
            return Err(KeywordClientError::Checkpoint("non-zero reserved bytes"));
        }
        let key_size_val: u64 = u64_at(&spec_buf, 72);
        let key_size: Option<usize> = match has_key_size {
            0 => {
                if key_size_val != 0 {
                    return Err(KeywordClientError::Checkpoint(
                        "non-zero key_size when has_key_size is 0",
                    ));
                }
                None
            }
            1 => Some(
                usize::try_from(key_size_val)
                    .map_err(|_| KeywordClientError::Checkpoint("key_size overflow"))?,
            ),
            _ => return Err(KeywordClientError::Checkpoint("invalid has_key_size flag")),
        };
        let cache_policy: CachePolicy = CachePolicy::from_byte(policy_byte)
            .ok_or(KeywordClientError::Checkpoint("invalid cache policy byte"))?;

        let spec: KeywordTableSpec = KeywordTableSpec::new(
            buckets_per_segment,
            bucket_size,
            tag_size,
            value_size,
            key_size,
            table_seed,
            table_epoch,
            state_epoch,
        )?;

        // Read runtime-stash (charge payload + HashMap/Vec per-entry overhead).
        let mut len_buf: [u8; 4] = [0u8; 4];
        inp.read_hashed(&mut len_buf)?;
        let stash_len: usize = u32::from_le_bytes(len_buf) as usize;
        let stash_entry_bytes: usize = std::mem::size_of::<((usize, Vec<u8>), Vec<u8>)>()
            .checked_add(16)
            .and_then(|v| v.checked_add(tag_size))
            .and_then(|v| v.checked_add(value_size))
            .ok_or(KeywordClientError::Checkpoint("stash entry size overflow"))?;
        let stash_total: usize = stash_len
            .checked_mul(stash_entry_bytes)
            .ok_or(KeywordClientError::Checkpoint("stash size overflow"))?;
        charge(stash_total)?;

        let mut stash: HashMap<(usize, Vec<u8>), Vec<u8>> = HashMap::new();
        stash
            .try_reserve(stash_len)
            .map_err(|_| KeywordClientError::Checkpoint("stash allocation failed"))?;
        for _ in 0..stash_len {
            let mut idx_buf: [u8; 8] = [0u8; 8];
            inp.read_hashed(&mut idx_buf)?;
            let bucket_seg0: usize = usize::try_from(u64::from_le_bytes(idx_buf))
                .map_err(|_| KeywordClientError::Checkpoint("stash bucket_seg0 overflow"))?;
            let mut tag: Vec<u8> = vec![0u8; tag_size];
            inp.read_hashed(&mut tag)?;
            let mut value: Vec<u8> = vec![0u8; value_size];
            inp.read_hashed(&mut value)?;
            validate_stash_fields(&spec, bucket_seg0, &tag, Some(&value))?;
            if stash.insert((bucket_seg0, tag), value).is_some() {
                return Err(KeywordClientError::Checkpoint(
                    "duplicate (bucket_seg0, tag) in checkpoint stash",
                ));
            }
        }

        // Read keyword bucket caches (charge payload + HashMap/Vec per-entry overhead).
        let rec: usize = spec.record_size();
        let mut bucket_cache: [HashMap<usize, Vec<u8>>; NUM_SEGMENTS] =
            [HashMap::new(), HashMap::new()];
        for cache in &mut bucket_cache {
            inp.read_hashed(&mut len_buf)?;
            let cache_len: usize = u32::from_le_bytes(len_buf) as usize;
            if cache_policy == CachePolicy::Off && cache_len > 0 {
                return Err(KeywordClientError::Checkpoint(
                    "non-empty bucket cache with CachePolicy::Off",
                ));
            }
            let entry_bytes: usize = std::mem::size_of::<(usize, Vec<u8>)>()
                .checked_add(16)
                .and_then(|v| v.checked_add(rec))
                .ok_or(KeywordClientError::Checkpoint("cache entry size overflow"))?;
            let total_bytes: usize = cache_len
                .checked_mul(entry_bytes)
                .ok_or(KeywordClientError::Checkpoint("cache size overflow"))?;
            charge(total_bytes)?;
            cache
                .try_reserve(cache_len)
                .map_err(|_| KeywordClientError::Checkpoint("cache allocation failed"))?;
            for _ in 0..cache_len {
                let mut idx_buf: [u8; 8] = [0u8; 8];
                inp.read_hashed(&mut idx_buf)?;
                let bucket_idx: usize = usize::try_from(u64::from_le_bytes(idx_buf))
                    .map_err(|_| KeywordClientError::Checkpoint("cache bucket_idx overflow"))?;
                if bucket_idx >= buckets_per_segment {
                    return Err(KeywordClientError::Checkpoint(
                        "cached bucket_idx out of range",
                    ));
                }
                let mut bytes: Vec<u8> = vec![0u8; rec];
                inp.read_hashed(&mut bytes)?;
                if cache.insert(bucket_idx, bytes).is_some() {
                    return Err(KeywordClientError::Checkpoint(
                        "duplicate bucket_idx in checkpoint cache",
                    ));
                }
            }
        }

        // Read raw backend blobs first so we can verify the outer checksum before decoding.
        let mut seg_blobs: [Vec<u8>; NUM_SEGMENTS] = [Vec::new(), Vec::new()];
        for blob in &mut seg_blobs {
            let mut blob_len_buf: [u8; 8] = [0u8; 8];
            inp.read_hashed(&mut blob_len_buf)?;
            let blob_len: usize = usize::try_from(u64::from_le_bytes(blob_len_buf))
                .map_err(|_| KeywordClientError::Checkpoint("backend blob length overflow"))?;
            charge(blob_len)?;
            blob.try_reserve_exact(blob_len)
                .map_err(|_| KeywordClientError::Checkpoint("backend blob allocation failed"))?;
            blob.resize(blob_len, 0u8);
            inp.read_hashed(blob)?;
        }

        match inp.finish() {
            Ok(()) => {}
            Err(TrailerError::Io(err)) => return Err(KeywordClientError::Io(err)),
            Err(TrailerError::ChecksumMismatch) => {
                return Err(KeywordClientError::Checkpoint("checksum mismatch"))
            }
            Err(TrailerError::TrailingData) => {
                return Err(KeywordClientError::Checkpoint(
                    "trailing data after checksum",
                ))
            }
        }

        let (b0, b0_retained): (B, usize) = read_backend(0, &seg_blobs[0], remaining_budget)?;
        remaining_budget =
            remaining_budget
                .checked_sub(b0_retained)
                .ok_or(KeywordClientError::Checkpoint(
                    "checkpoint exceeds memory budget",
                ))?;
        let (b1, b1_retained): (B, usize) = read_backend(1, &seg_blobs[1], remaining_budget)?;
        remaining_budget
            .checked_sub(b1_retained)
            .ok_or(KeywordClientError::Checkpoint(
                "checkpoint exceeds memory budget",
            ))?;
        let backends: [B; NUM_SEGMENTS] = [b0, b1];
        let expected: (usize, usize) = (spec.buckets_per_segment(), spec.record_size());
        for (seg, backend) in backends.iter().enumerate() {
            let actual: (usize, usize) = (backend.num_entries(), backend.record_size());
            if actual != expected {
                return Err(KeywordClientError::BackendGeometryMismatch {
                    segment: seg,
                    expected,
                    actual,
                });
            }
            if backend.queries_in_flight() > 0 {
                return Err(KeywordClientError::Checkpoint(
                    "restored segment backend contains in-flight queries",
                ));
            }
        }

        Ok(Self {
            session_id: next_keyword_session_id(),
            spec,
            backends,
            cache_policy,
            stash,
            bucket_cache,
        })
    }
}

impl KeywordClient<MultiLevelClient> {
    /// Serializes a 2-segment [`MultiLevelClient`]-backed [`KeywordClient`] checkpoint.
    ///
    /// Master seeds are never written; supply the same `[seed_seg0, seed_seg1]` to
    /// [`KeywordClient::read_checkpoint`] or [`KeywordClient::from_bytes`].
    pub fn write_checkpoint<W: Write>(
        &self,
        writer: W,
    ) -> Result<(), KeywordClientError<MultiLevelError>> {
        self.write_checkpoint_with(writer, |seg, backend, buf| {
            backend
                .write_hints(buf)
                .map_err(|error| KeywordClientError::Backend {
                    segment: seg,
                    error,
                })
        })
    }

    /// Serializes this checkpoint to a `Vec<u8>`.
    pub fn to_bytes(&self) -> Result<Vec<u8>, KeywordClientError<MultiLevelError>> {
        let mut buf: Vec<u8> = Vec::new();
        self.write_checkpoint(&mut buf)?;
        Ok(buf)
    }

    /// Restores a 2-segment [`MultiLevelClient`]-backed [`KeywordClient`] checkpoint
    /// using the default decoded memory budget (`512 MiB`).
    pub fn read_checkpoint<R: Read>(
        reader: R,
        seg_seeds: &[[u8; 32]; NUM_SEGMENTS],
    ) -> Result<Self, KeywordClientError<MultiLevelError>> {
        Self::read_checkpoint_with_budget(reader, seg_seeds, MAX_KEYWORD_CHECKPOINT_BYTES)
    }

    /// Restores a 2-segment [`MultiLevelClient`]-backed [`KeywordClient`] checkpoint
    /// with an explicit decoded memory budget.
    pub fn read_checkpoint_with_budget<R: Read>(
        reader: R,
        seg_seeds: &[[u8; 32]; NUM_SEGMENTS],
        max_bytes: usize,
    ) -> Result<Self, KeywordClientError<MultiLevelError>> {
        Self::read_checkpoint_with_limit(reader, max_bytes, |seg, blob, limit| {
            let (client, retained): (MultiLevelClient, usize) =
                MultiLevelClient::from_bytes_with_limit_accounted(blob, &seg_seeds[seg], limit)
                    .map_err(|error| KeywordClientError::Backend {
                        segment: seg,
                        error,
                    })?;
            Ok((client, retained))
        })
    }

    /// Restores a 2-segment [`MultiLevelClient`]-backed [`KeywordClient`] from bytes.
    pub fn from_bytes(
        bytes: &[u8],
        seg_seeds: &[[u8; 32]; NUM_SEGMENTS],
    ) -> Result<Self, KeywordClientError<MultiLevelError>> {
        Self::read_checkpoint(bytes, seg_seeds)
    }
}

impl KeywordClient<ProtocolClient> {
    /// Serializes a 2-segment [`crate::protocol::Client`]-backed [`KeywordClient`] checkpoint.
    ///
    /// Master seeds and query seeds are never written; supply the same master seeds
    /// and fresh random query seeds when restoring.
    pub fn write_checkpoint<W: Write>(
        &self,
        writer: W,
    ) -> Result<(), KeywordClientError<ProtocolError>> {
        self.write_checkpoint_with(writer, |_seg, backend, buf| {
            backend.write_state(buf).map_err(KeywordClientError::Io)
        })
    }

    /// Serializes this checkpoint to a `Vec<u8>`.
    pub fn to_bytes(&self) -> Result<Vec<u8>, KeywordClientError<ProtocolError>> {
        let mut buf: Vec<u8> = Vec::new();
        self.write_checkpoint(&mut buf)?;
        Ok(buf)
    }

    /// Restores a 2-segment [`crate::protocol::Client`]-backed [`KeywordClient`] checkpoint
    /// using the default decoded memory budget (`512 MiB`).
    pub fn read_checkpoint<R: Read>(
        reader: R,
        seg_seeds: &[[u8; 32]; NUM_SEGMENTS],
        query_seeds: &[[u8; 32]; NUM_SEGMENTS],
    ) -> Result<Self, KeywordClientError<ProtocolError>> {
        Self::read_checkpoint_with_budget(
            reader,
            seg_seeds,
            query_seeds,
            MAX_KEYWORD_CHECKPOINT_BYTES,
        )
    }

    /// Restores a 2-segment [`crate::protocol::Client`]-backed [`KeywordClient`] checkpoint
    /// with an explicit decoded memory budget.
    pub fn read_checkpoint_with_budget<R: Read>(
        reader: R,
        seg_seeds: &[[u8; 32]; NUM_SEGMENTS],
        query_seeds: &[[u8; 32]; NUM_SEGMENTS],
        max_bytes: usize,
    ) -> Result<Self, KeywordClientError<ProtocolError>> {
        Self::read_checkpoint_with_limit(reader, max_bytes, |seg, blob, limit| {
            let (client, in_flight, retained) = ProtocolClient::read_state_with_limit(
                seg_seeds[seg],
                query_seeds[seg],
                blob,
                limit,
            )
            .map_err(|err| match err {
                ClientStateError::Mismatch(p) => KeywordClientError::Backend {
                    segment: seg,
                    error: p,
                },
                ClientStateError::Io(io_err) => KeywordClientError::Io(io_err),
                ClientStateError::Invalid(msg) => KeywordClientError::Checkpoint(msg),
                _ => KeywordClientError::Checkpoint("invalid protocol client state blob"),
            })?;
            if !in_flight.is_empty() {
                return Err(KeywordClientError::Checkpoint(
                    "restored protocol backend contains in-flight queries",
                ));
            }
            Ok((client, retained))
        })
    }

    /// Restores a 2-segment [`crate::protocol::Client`]-backed [`KeywordClient`] from bytes.
    pub fn from_bytes(
        bytes: &[u8],
        seg_seeds: &[[u8; 32]; NUM_SEGMENTS],
        query_seeds: &[[u8; 32]; NUM_SEGMENTS],
    ) -> Result<Self, KeywordClientError<ProtocolError>> {
        Self::read_checkpoint(bytes, seg_seeds, query_seeds)
    }
}

fn validate_stash_fields<E>(
    spec: &KeywordTableSpec,
    bucket_seg0: usize,
    tag: &[u8],
    value: Option<&[u8]>,
) -> Result<(), KeywordClientError<E>> {
    if bucket_seg0 >= spec.buckets_per_segment() {
        return Err(KeywordClientError::InvalidEpochUpdate(
            "stash bucket_seg0 out of range",
        ));
    }
    if tag.len() != spec.tag_size() || is_all_zero(tag) {
        return Err(KeywordClientError::Keyword(KeywordError::InvalidTag));
    }
    if let Some(val) = value {
        if val.len() != spec.value_size() {
            return Err(KeywordClientError::Keyword(
                KeywordError::ValueSizeMismatch {
                    expected: spec.value_size(),
                    actual: val.len(),
                },
            ));
        }
    }
    Ok(())
}

/// Synchronous transport abstraction for [`KeywordSession`].
pub trait KeywordTransport<Q, A> {
    /// Transport error type.
    type Error: Error + Send + Sync + 'static;

    /// Sends a single 2-segment lookup query and returns the server's answer.
    fn send_lookup(
        &mut self,
        query: KeywordLookupQuery<Q>,
    ) -> Result<KeywordLookupAnswer<A>, Self::Error>;

    /// Sends a 1-RTT batch lookup query and returns the server's batch answer.
    fn send_batch(
        &mut self,
        query: KeywordBatchQuery<Q>,
    ) -> Result<KeywordBatchAnswer<A>, Self::Error>;
}

/// Error returned by [`KeywordSession::get`] or [`KeywordSession::get_batch`].
#[derive(Debug)]
pub enum KeywordSessionError<BE, TE> {
    /// Error from [`KeywordClient`] (any in-flight hints were aborted cleanly).
    Client(KeywordClientError<BE>),
    /// Error from [`KeywordTransport`] (in-flight hints were aborted cleanly).
    Transport(TE),
    /// Transport failed and aborting the exposed in-flight hints also failed;
    /// hints may still remain in flight.
    TransportAndAbort {
        /// Underlying transport failure.
        transport: TE,
        /// Failure returned while aborting the exposed hints.
        abort: KeywordClientError<BE>,
    },
    /// Completing the lookup failed and aborting the remaining in-flight hints
    /// also failed; hints may still remain in flight.
    FinishAndAbort {
        /// Failure returned by `finish_lookup` / `finish_batch`.
        finish: KeywordClientError<BE>,
        /// Failure returned while aborting the exposed hints.
        abort: KeywordClientError<BE>,
    },
}

impl<BE: fmt::Display, TE: fmt::Display> fmt::Display for KeywordSessionError<BE, TE> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Client(err) => write!(f, "{err}"),
            Self::Transport(err) => write!(f, "keyword transport error: {err}"),
            Self::TransportAndAbort { transport, abort } => write!(
                f,
                "keyword transport error ({transport}) and hint abort failed ({abort})"
            ),
            Self::FinishAndAbort { finish, abort } => write!(
                f,
                "keyword finish error ({finish}) and hint abort failed ({abort})"
            ),
        }
    }
}

impl<BE: Error + 'static, TE: Error + 'static> Error for KeywordSessionError<BE, TE> {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Client(err) => Some(err),
            Self::Transport(err) => Some(err),
            Self::TransportAndAbort { transport, .. } => Some(transport),
            Self::FinishAndAbort { finish, .. } => Some(finish),
        }
    }
}

/// Thin synchronous convenience wrapper around [`KeywordClient`] + [`KeywordTransport`].
///
/// Automatically aborts in-flight hints on transport or answer-validation failure
/// so exposed hints are permanently retired and never left dangling.
pub struct KeywordSession<B: IndexPirClient, T, R> {
    client: KeywordClient<B>,
    transport: T,
    rng: R,
}

impl<B, T, R> KeywordSession<B, T, R>
where
    B: IndexPirClient,
    T: KeywordTransport<B::Query, B::Answer>,
    R: Rng,
{
    /// Creates a new [`KeywordSession`].
    pub fn new(client: KeywordClient<B>, transport: T, rng: R) -> Self {
        Self {
            client,
            transport,
            rng,
        }
    }

    /// Performs a private lookup for `(kind, key)`, aborting in-flight hints
    /// automatically on transport or completion error.
    ///
    /// Note: this convenience method does not expose `pending.hint_ids()` before
    /// sending, so it cannot perform the WAL-before-send step required for
    /// checkpoint-and-WAL crash recovery. Callers that need WAL recovery must use
    /// [`Self::client_mut`] to call [`KeywordClient::prepare_lookup`], durably log
    /// `pending.hint_ids()` before sending, and finish via
    /// [`KeywordClient::finish_lookup`] or [`KeywordClient::abort_lookup`].
    pub fn get(&mut self, kind: u8, key: &[u8]) -> SessionResult<Option<Vec<u8>>, B, T> {
        let in_flight_before: usize = self.client.queries_in_flight();
        match self
            .client
            .prepare_lookup(kind, key, &mut self.rng)
            .map_err(KeywordSessionError::Client)?
        {
            LookupPlan::Local(val) => Ok(val),
            LookupPlan::Query { query, mut pending } => {
                let answer: KeywordLookupAnswer<B::Answer> = match self.transport.send_lookup(query)
                {
                    Ok(a) => a,
                    Err(transport) => {
                        return match self.client.abort_lookup(&mut pending) {
                            Ok(()) => Err(KeywordSessionError::Transport(transport)),
                            Err(abort) => {
                                Err(KeywordSessionError::TransportAndAbort { transport, abort })
                            }
                        };
                    }
                };
                match self.client.finish_lookup(&pending, &answer, &mut self.rng) {
                    Ok(val) => Ok(val),
                    Err(finish) => {
                        if self.client.queries_in_flight() > in_flight_before {
                            if let Err(abort) = self.client.abort_lookup(&mut pending) {
                                return Err(KeywordSessionError::FinishAndAbort { finish, abort });
                            }
                        }
                        Err(KeywordSessionError::Client(finish))
                    }
                }
            }
        }
    }

    /// Performs a 1-RTT batch of private lookups for `keys`, aborting in-flight
    /// hints automatically on transport or completion error.
    ///
    /// Note: like [`Self::get`], this convenience method does not expose
    /// `pending.hint_ids()` before sending; callers needing checkpoint-and-WAL
    /// crash recovery must drive [`KeywordClient::prepare_batch`], WAL logging,
    /// and [`KeywordClient::finish_batch`] / [`KeywordClient::abort_batch`]
    /// directly via [`Self::client_mut`].
    pub fn get_batch(&mut self, keys: &[(u8, &[u8])]) -> SessionResult<Vec<Option<Vec<u8>>>, B, T> {
        let in_flight_before: usize = self.client.queries_in_flight();
        match self
            .client
            .prepare_batch(keys, &mut self.rng)
            .map_err(KeywordSessionError::Client)?
        {
            BatchLookupPlan::Local(vals) => Ok(vals),
            BatchLookupPlan::Query { query, mut pending } => {
                let answer: KeywordBatchAnswer<B::Answer> = match self.transport.send_batch(query) {
                    Ok(a) => a,
                    Err(transport) => {
                        return match self.client.abort_batch(&mut pending) {
                            Ok(()) => Err(KeywordSessionError::Transport(transport)),
                            Err(abort) => {
                                Err(KeywordSessionError::TransportAndAbort { transport, abort })
                            }
                        };
                    }
                };
                match self.client.finish_batch(&pending, &answer, &mut self.rng) {
                    Ok(vals) => Ok(vals),
                    Err(finish) => {
                        if self.client.queries_in_flight() > in_flight_before {
                            if let Err(abort) = self.client.abort_batch(&mut pending) {
                                return Err(KeywordSessionError::FinishAndAbort { finish, abort });
                            }
                        }
                        Err(KeywordSessionError::Client(finish))
                    }
                }
            }
        }
    }

    /// Returns a reference to the underlying [`KeywordClient`].
    pub fn client(&self) -> &KeywordClient<B> {
        &self.client
    }

    /// Returns a mutable reference to the underlying [`KeywordClient`].
    pub fn client_mut(&mut self) -> &mut KeywordClient<B> {
        &mut self.client
    }

    /// Returns a reference to the underlying transport.
    pub fn transport(&self) -> &T {
        &self.transport
    }

    /// Returns a mutable reference to the underlying transport.
    pub fn transport_mut(&mut self) -> &mut T {
        &mut self.transport
    }

    /// Consumes the session and returns `(client, transport, rng)`.
    pub fn into_parts(self) -> (KeywordClient<B>, T, R) {
        (self.client, self.transport, self.rng)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::keyword::KvScfTable;
    use crate::multilevel::{LevelBackend, LevelSpec, MultiLevelParams, MultiLevelServer};
    use crate::protocol::{ProtocolParams, Server as ProtocolServer};
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;

    struct InMemoryMlTransport {
        table: KvScfTable,
        params: MultiLevelParams,
        seg_epochs: [u64; NUM_SEGMENTS],
        fail_next: bool,
    }

    #[derive(Debug)]
    struct MockTransportError(&'static str);

    impl fmt::Display for MockTransportError {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "{}", self.0)
        }
    }

    impl Error for MockTransportError {}

    impl InMemoryMlTransport {
        fn answer_lookup(
            &self,
            q: &KeywordLookupQuery<MultiLevelQuery>,
        ) -> KeywordLookupAnswer<MultiLevelAnswer> {
            let a0 = MultiLevelServer::answer(
                self.table.segment_bytes(0).unwrap(),
                &self.params,
                &q.segments[0],
                self.seg_epochs[0],
            )
            .unwrap();
            let a1 = MultiLevelServer::answer(
                self.table.segment_bytes(1).unwrap(),
                &self.params,
                &q.segments[1],
                self.seg_epochs[1],
            )
            .unwrap();
            KeywordLookupAnswer {
                table_epoch: self.table.spec().table_epoch(),
                state_epoch: self.table.spec().state_epoch(),
                segments: [a0, a1],
            }
        }

        fn answer_batch(
            &self,
            q: &KeywordBatchQuery<MultiLevelQuery>,
        ) -> KeywordBatchAnswer<MultiLevelAnswer> {
            let a0: Vec<MultiLevelAnswer> = q.segments[0]
                .iter()
                .map(|sq| {
                    MultiLevelServer::answer(
                        self.table.segment_bytes(0).unwrap(),
                        &self.params,
                        sq,
                        self.seg_epochs[0],
                    )
                    .unwrap()
                })
                .collect();
            let a1: Vec<MultiLevelAnswer> = q.segments[1]
                .iter()
                .map(|sq| {
                    MultiLevelServer::answer(
                        self.table.segment_bytes(1).unwrap(),
                        &self.params,
                        sq,
                        self.seg_epochs[1],
                    )
                    .unwrap()
                })
                .collect();
            KeywordBatchAnswer {
                table_epoch: self.table.spec().table_epoch(),
                state_epoch: self.table.spec().state_epoch(),
                segments: [a0, a1],
            }
        }

        fn publish_outcomes(&mut self, outcomes: &[MutationOutcome]) -> KeywordEpochUpdate {
            let from_epoch: u64 = self.table.spec().state_epoch();
            let to_epoch: u64 = self.table.advance_state_epoch().unwrap();
            let update: KeywordEpochUpdate = KeywordEpochUpdate::from_outcomes(
                self.table.spec().table_epoch(),
                from_epoch,
                to_epoch,
                outcomes,
            );
            for seg in 0..NUM_SEGMENTS {
                if !update.segment_deltas[seg].is_empty() {
                    self.seg_epochs[seg] += 1;
                }
            }
            update
        }
    }

    impl KeywordTransport<MultiLevelQuery, MultiLevelAnswer> for InMemoryMlTransport {
        type Error = MockTransportError;

        fn send_lookup(
            &mut self,
            query: KeywordLookupQuery<MultiLevelQuery>,
        ) -> Result<KeywordLookupAnswer<MultiLevelAnswer>, Self::Error> {
            if self.fail_next {
                self.fail_next = false;
                return Err(MockTransportError("simulated lookup transport failure"));
            }
            Ok(self.answer_lookup(&query))
        }

        fn send_batch(
            &mut self,
            query: KeywordBatchQuery<MultiLevelQuery>,
        ) -> Result<KeywordBatchAnswer<MultiLevelAnswer>, Self::Error> {
            if self.fail_next {
                self.fail_next = false;
                return Err(MockTransportError("simulated batch transport failure"));
            }
            Ok(self.answer_batch(&query))
        }
    }

    fn make_ml_params(s: usize, record_size: usize, query_budget: usize) -> MultiLevelParams {
        let block_size: usize = ((s as f64).sqrt().ceil() as usize)
            .max(4)
            .next_power_of_two();
        let num_blocks: usize = s.div_ceil(block_size);
        let num_primary: usize = (16 * block_size).max(64);
        MultiLevelParams::custom(
            s,
            record_size,
            vec![LevelSpec {
                block_size,
                num_blocks,
                num_primary,
                num_backup: query_budget,
                backend: LevelBackend::FlatAes,
            }],
        )
        .unwrap()
    }

    fn make_kv_pair(id: u64, value_size: usize) -> (Vec<u8>, Vec<u8>) {
        let key: Vec<u8> = format!("key-{id:06}").into_bytes();
        let val: Vec<u8> = (0..value_size)
            .map(|j| ((id as u8).wrapping_mul(37) ^ (j as u8).wrapping_mul(13)) | 1)
            .collect();
        (key, val)
    }

    fn init_ml_keyword_client(
        table: &KvScfTable,
        params: &MultiLevelParams,
        seeds: &[[u8; 32]; 2],
        policy: CachePolicy,
    ) -> KeywordClient<MultiLevelClient> {
        let c0: MultiLevelClient = MultiLevelClient::hint_init(
            params.clone(),
            &seeds[0],
            table.segment_bytes(0).unwrap(),
            1,
        )
        .unwrap();
        let c1: MultiLevelClient = MultiLevelClient::hint_init(
            params.clone(),
            &seeds[1],
            table.segment_bytes(1).unwrap(),
            1,
        )
        .unwrap();
        KeywordClient::with_stash(table.spec().clone(), [c0, c1], policy, table.stash()).unwrap()
    }

    #[test]
    fn multilevel_e2e_policies_single_segment_delta_and_stash() {
        let value_size: usize = 19;
        let num_keys: usize = 120;
        let spec: KeywordTableSpec =
            KeywordTableSpec::for_capacity(num_keys, value_size, 0.80, [0x42; 32]).unwrap();
        let entries: Vec<(u8, Vec<u8>, Vec<u8>)> = (0..num_keys as u64)
            .map(|i| {
                let (k, v) = make_kv_pair(i, value_size);
                (0x01, k, v)
            })
            .collect();
        let (table, _) = KvScfTable::build(
            spec.clone(),
            entries
                .iter()
                .map(|(knd, k, v)| (*knd, k.as_slice(), v.as_slice())),
        )
        .unwrap();
        let table = table.with_max_stash(8);

        let params: MultiLevelParams =
            make_ml_params(spec.buckets_per_segment(), spec.record_size(), 64);
        let seeds: [[u8; 32]; 2] = [[0x11; 32], [0x22; 32]];

        for policy in [
            CachePolicy::Off,
            CachePolicy::LocalHit,
            CachePolicy::ConstantTraffic,
        ] {
            let mut transport = InMemoryMlTransport {
                table: KvScfTable::from_database_bytes(spec.clone(), table.as_bytes().to_vec())
                    .unwrap()
                    .with_max_stash(8),
                params: params.clone(),
                seg_epochs: [0, 0],
                fail_next: false,
            };
            let mut client = init_ml_keyword_client(&transport.table, &params, &seeds, policy);
            let mut rng = ChaCha20Rng::seed_from_u64(777);

            // 1) Initial lookup of key 0 (hit) and absent key (miss).
            let (k0, v0) = (&entries[0].1, &entries[0].2);
            let plan = client.prepare_lookup(0x01, k0, &mut rng).unwrap();
            let got = match plan {
                LookupPlan::Query { query, pending } => {
                    let ans = transport.answer_lookup(&query);
                    client.finish_lookup(&pending, &ans, &mut rng).unwrap()
                }
                LookupPlan::Local(_) => panic!("first lookup must hit network"),
            };
            assert_eq!(got.as_deref(), Some(v0.as_slice()));

            // 2) Repeat lookup of key 0 checks CachePolicy behavior.
            let plan2 = client.prepare_lookup(0x01, k0, &mut rng).unwrap();
            match (policy, plan2) {
                (CachePolicy::LocalHit, LookupPlan::Local(val)) => {
                    assert_eq!(val.as_deref(), Some(v0.as_slice()));
                }
                (
                    CachePolicy::Off | CachePolicy::ConstantTraffic,
                    LookupPlan::Query { query, pending },
                ) => {
                    let ans = transport.answer_lookup(&query);
                    let val = client.finish_lookup(&pending, &ans, &mut rng).unwrap();
                    assert_eq!(val.as_deref(), Some(v0.as_slice()));
                }
                _ => panic!("unexpected LookupPlan for policy {policy:?}"),
            }

            // 3) Single-segment update: modify key 0 so ONLY one segment's epoch advances!
            let new_v0: Vec<u8> = vec![0xEE; value_size];
            let upd_out = transport.table.update(0x01, k0, &new_v0).unwrap();
            assert_eq!(upd_out.deltas.len(), 1);
            let touched_seg: usize = upd_out.deltas[0].segment;
            let epoch_upd = transport.publish_outcomes(&[upd_out]);
            assert_eq!(transport.seg_epochs[touched_seg], 1);
            assert_eq!(transport.seg_epochs[1 - touched_seg], 0);
            client.apply_epoch_update(&epoch_upd, 1).unwrap();
            assert_eq!(client.segment_epochs(), transport.seg_epochs);

            // Cached bucket must have been XOR-updated in place under LocalHit/ConstantTraffic.
            let plan3 = client.prepare_lookup(0x01, k0, &mut rng).unwrap();
            let got3 = match plan3 {
                LookupPlan::Local(val) => {
                    assert_eq!(policy, CachePolicy::LocalHit);
                    val
                }
                LookupPlan::Query { query, pending } => {
                    let ans = transport.answer_lookup(&query);
                    client.finish_lookup(&pending, &ans, &mut rng).unwrap()
                }
            };
            assert_eq!(got3.as_deref(), Some(new_v0.as_slice()));

            // 4) Batch lookup with duplicate keys + cached key + absent key.
            let (k1, v1) = (&entries[1].1, &entries[1].2);
            let (k2, v2) = (&entries[2].1, &entries[2].2);
            let missing_k = b"absent-key-99999";
            let batch_keys: Vec<(u8, &[u8])> = vec![
                (0x01, k0.as_slice()),
                (0x01, k1.as_slice()),
                (0x01, k1.as_slice()),
                (0x01, k2.as_slice()),
                (0x01, missing_k.as_slice()),
            ];
            let batch_plan = client.prepare_batch(&batch_keys, &mut rng).unwrap();
            let batch_vals = match batch_plan {
                BatchLookupPlan::Query { query, pending } => {
                    match policy {
                        CachePolicy::Off | CachePolicy::ConstantTraffic => {
                            assert_eq!(pending.wire_query_counts(), [5, 5]);
                        }
                        CachePolicy::LocalHit => {
                            // k0 is cached and k1 is duplicated, so fewer than 5 queries per seg.
                            let counts = pending.wire_query_counts();
                            assert!(counts[0] < 5 && counts[1] < 5);
                        }
                    }
                    let ans = transport.answer_batch(&query);
                    client.finish_batch(&pending, &ans, &mut rng).unwrap()
                }
                BatchLookupPlan::Local(_) => panic!("uncached keys must issue wire queries"),
            };
            assert_eq!(batch_vals[0].as_deref(), Some(new_v0.as_slice()));
            assert_eq!(batch_vals[1].as_deref(), Some(v1.as_slice()));
            assert_eq!(batch_vals[2].as_deref(), Some(v1.as_slice()));
            assert_eq!(batch_vals[3].as_deref(), Some(v2.as_slice()));
            assert_eq!(batch_vals[4], None);

            // 5) Runtime-stash Insert, Update, and Delete via KeywordEpochUpdate.
            let stash_key = b"stashed-overflow-key";
            let stash_loc = client.spec().locate(0x02, stash_key).unwrap();
            let stash_v1: Vec<u8> = vec![0x5A; value_size];
            let stash_v2: Vec<u8> = vec![0xA5; value_size];
            let ins_epoch = KeywordEpochUpdate {
                table_epoch: client.spec().table_epoch(),
                from_state_epoch: client.spec().state_epoch(),
                to_state_epoch: client.spec().state_epoch() + 1,
                segment_deltas: [Vec::new(), Vec::new()],
                stash_ops: vec![StashOp::Insert(StashEntry {
                    bucket_seg0: stash_loc.bucket_seg0,
                    tag: stash_loc.tag.clone(),
                    value: stash_v1.clone(),
                })],
            };
            transport.table.advance_state_epoch();
            client.apply_epoch_update(&ins_epoch, 1).unwrap();
            assert_eq!(client.stash_len(), 1);

            let stash_plan = client.prepare_lookup(0x02, stash_key, &mut rng).unwrap();
            let got_stash = match (policy, stash_plan) {
                (CachePolicy::LocalHit, LookupPlan::Local(v)) => v,
                (
                    CachePolicy::Off | CachePolicy::ConstantTraffic,
                    LookupPlan::Query { query, pending },
                ) => {
                    let ans = transport.answer_lookup(&query);
                    client.finish_lookup(&pending, &ans, &mut rng).unwrap()
                }
                _ => panic!("unexpected stash plan for {policy:?}"),
            };
            assert_eq!(got_stash.as_deref(), Some(stash_v1.as_slice()));

            // Update stash entry then delete it.
            let upd_stash = KeywordEpochUpdate {
                table_epoch: client.spec().table_epoch(),
                from_state_epoch: client.spec().state_epoch(),
                to_state_epoch: client.spec().state_epoch() + 1,
                segment_deltas: [Vec::new(), Vec::new()],
                stash_ops: vec![StashOp::Update {
                    bucket_seg0: stash_loc.bucket_seg0,
                    tag: stash_loc.tag.clone(),
                    value: stash_v2.clone(),
                }],
            };
            transport.table.advance_state_epoch();
            client.apply_epoch_update(&upd_stash, 1).unwrap();
            let got_stash2 = match client.prepare_lookup(0x02, stash_key, &mut rng).unwrap() {
                LookupPlan::Local(v) => v,
                LookupPlan::Query { query, pending } => {
                    let ans = transport.answer_lookup(&query);
                    client.finish_lookup(&pending, &ans, &mut rng).unwrap()
                }
            };
            assert_eq!(got_stash2.as_deref(), Some(stash_v2.as_slice()));

            let del_stash = KeywordEpochUpdate {
                table_epoch: client.spec().table_epoch(),
                from_state_epoch: client.spec().state_epoch(),
                to_state_epoch: client.spec().state_epoch() + 1,
                segment_deltas: [Vec::new(), Vec::new()],
                stash_ops: vec![StashOp::Delete {
                    bucket_seg0: stash_loc.bucket_seg0,
                    tag: stash_loc.tag.clone(),
                }],
            };
            transport.table.advance_state_epoch();
            client.apply_epoch_update(&del_stash, 1).unwrap();
            assert_eq!(client.stash_len(), 0);
            let got_deleted = match client.prepare_lookup(0x02, stash_key, &mut rng).unwrap() {
                LookupPlan::Local(v) => v,
                LookupPlan::Query { query, pending } => {
                    let ans = transport.answer_lookup(&query);
                    client.finish_lookup(&pending, &ans, &mut rng).unwrap()
                }
            };
            assert_eq!(got_deleted, None);
        }
    }

    #[test]
    fn protocol_client_two_pass_batch_handles_cached_duplicates_and_rollback() {
        let value_size: usize = 16;
        let spec: KeywordTableSpec =
            KeywordTableSpec::new(12, 4, 8, value_size, None, [0x55; 32], 0, 0).unwrap();
        let entries: Vec<(u8, Vec<u8>, Vec<u8>)> = (0..40u64)
            .map(|i| {
                let (k, v) = make_kv_pair(i, value_size);
                (0x01, k, v)
            })
            .collect();
        let (table, _) = KvScfTable::build(
            spec.clone(),
            entries
                .iter()
                .map(|(knd, k, v)| (*knd, k.as_slice(), v.as_slice())),
        )
        .unwrap();

        let p_params = ProtocolParams::new(
            spec.buckets_per_segment(),
            spec.record_size(),
            Some(4),
            16,
            Some(32),
        )
        .unwrap();
        let s0 = ProtocolServer::new(p_params.clone(), 0, table.segment_bytes(0).unwrap()).unwrap();
        let s1 = ProtocolServer::new(p_params.clone(), 0, table.segment_bytes(1).unwrap()).unwrap();
        let c0 = ProtocolClient::offline_init(
            p_params.clone(),
            [0x10; 32],
            [0x20; 32],
            0,
            table.segment_bytes(0).unwrap(),
        )
        .unwrap();
        let c1 = ProtocolClient::offline_init(
            p_params,
            [0x30; 32],
            [0x40; 32],
            0,
            table.segment_bytes(1).unwrap(),
        )
        .unwrap();

        let mut kw = KeywordClient::new(spec.clone(), [c0, c1], CachePolicy::Off).unwrap();
        let mut rng = ChaCha20Rng::seed_from_u64(909);

        // Warm protocol::Client's internal cache with keys 0 and 1.
        for (_, key, expected_val) in entries.iter().take(2) {
            let plan = kw.prepare_lookup(0x01, key, &mut rng).unwrap();
            let LookupPlan::Query { query, pending } = plan else {
                unreachable!()
            };
            let ans = KeywordLookupAnswer {
                table_epoch: 0,
                state_epoch: 0,
                segments: [
                    s0.answer(&query.segments[0]).unwrap(),
                    s1.answer(&query.segments[1]).unwrap(),
                ],
            };
            let got = kw.finish_lookup(&pending, &ans, &mut rng).unwrap();
            assert_eq!(got.as_deref(), Some(expected_val.as_slice()));
        }

        // Batch mixes backend-cached keys (0, 1), duplicate keys (0, 2), and uncached keys (2..6).
        // Two-pass scheduling must schedule uncached buckets first so internal backend decoys
        // never collide with real batch buckets.
        let batch_keys: Vec<(u8, &[u8])> = vec![
            (0x01, entries[0].1.as_slice()),
            (0x01, entries[0].1.as_slice()),
            (0x01, entries[1].1.as_slice()),
            (0x01, entries[2].1.as_slice()),
            (0x01, entries[2].1.as_slice()),
            (0x01, entries[3].1.as_slice()),
            (0x01, entries[4].1.as_slice()),
            (0x01, b"missing-protocol-key"),
        ];
        let plan = kw.prepare_batch(&batch_keys, &mut rng).unwrap();
        let BatchLookupPlan::Query { query, pending } = plan else {
            unreachable!()
        };
        assert_eq!(pending.wire_query_counts(), [8, 8]);
        let ans = KeywordBatchAnswer {
            table_epoch: 0,
            state_epoch: 0,
            segments: [
                query.segments[0]
                    .iter()
                    .map(|q| s0.answer(q).unwrap())
                    .collect(),
                query.segments[1]
                    .iter()
                    .map(|q| s1.answer(q).unwrap())
                    .collect(),
            ],
        };
        let vals = kw.finish_batch(&pending, &ans, &mut rng).unwrap();
        assert_eq!(vals[0].as_deref(), Some(entries[0].2.as_slice()));
        assert_eq!(vals[1].as_deref(), Some(entries[0].2.as_slice()));
        assert_eq!(vals[2].as_deref(), Some(entries[1].2.as_slice()));
        assert_eq!(vals[3].as_deref(), Some(entries[2].2.as_slice()));
        assert_eq!(vals[4].as_deref(), Some(entries[2].2.as_slice()));
        assert_eq!(vals[5].as_deref(), Some(entries[3].2.as_slice()));
        assert_eq!(vals[6].as_deref(), Some(entries[4].2.as_slice()));
        assert_eq!(vals[7], None);

        // Requesting batch_len > s on protocol::Client must fail with NoDecoyIndexAvailable
        // and roll back every tentative reservation without leaking hints.
        let remaining_before: usize = kw.remaining_queries();
        let oversized: Vec<(u8, &[u8])> =
            (0..13).map(|_| (0x01, entries[0].1.as_slice())).collect();
        let err = kw.prepare_batch(&oversized, &mut rng).unwrap_err();
        assert!(matches!(
            err,
            KeywordClientError::NoDecoyIndexAvailable { .. }
                | KeywordClientError::Backend {
                    error: ProtocolError::NoBackupHintsLeft,
                    ..
                }
        ));
        assert_eq!(kw.queries_in_flight(), 0);
        assert_eq!(kw.remaining_queries(), remaining_before);
    }

    #[test]
    fn epoch_mismatch_is_atomic_and_session_aborts_on_transport_error() {
        let value_size: usize = 16;
        let spec: KeywordTableSpec =
            KeywordTableSpec::for_capacity(80, value_size, 0.80, [0x77; 32]).unwrap();
        let entries: Vec<(u8, Vec<u8>, Vec<u8>)> = (0..80u64)
            .map(|i| {
                let (k, v) = make_kv_pair(i, value_size);
                (0x01, k, v)
            })
            .collect();
        let (table, _) = KvScfTable::build(
            spec.clone(),
            entries
                .iter()
                .map(|(knd, k, v)| (*knd, k.as_slice(), v.as_slice())),
        )
        .unwrap();
        let params: MultiLevelParams =
            make_ml_params(spec.buckets_per_segment(), spec.record_size(), 32);
        let seeds: [[u8; 32]; 2] = [[0x31; 32], [0x32; 32]];

        let mut transport = InMemoryMlTransport {
            table,
            params: params.clone(),
            seg_epochs: [0, 0],
            fail_next: false,
        };
        let mut client =
            init_ml_keyword_client(&transport.table, &params, &seeds, CachePolicy::Off);
        let mut rng = ChaCha20Rng::seed_from_u64(55);

        // 1) Stale segment-1 epoch must fail finish_lookup WITHOUT mutating segment 0,
        //    so retrying with the correct answer succeeds.
        let plan = client
            .prepare_lookup(0x01, &entries[5].1, &mut rng)
            .unwrap();
        let LookupPlan::Query { query, pending } = plan else {
            unreachable!()
        };
        let mut bad_ans = transport.answer_lookup(&query);
        bad_ans.segments[1].snapshot_epoch = 999;
        let err = client
            .finish_lookup(&pending, &bad_ans, &mut rng)
            .unwrap_err();
        assert!(matches!(
            err,
            KeywordClientError::Backend { segment: 1, .. }
        ));
        assert_eq!(client.queries_in_flight(), 2);

        let good_ans = transport.answer_lookup(&query);
        let val = client.finish_lookup(&pending, &good_ans, &mut rng).unwrap();
        assert_eq!(val.as_deref(), Some(entries[5].2.as_slice()));
        assert_eq!(client.queries_in_flight(), 0);

        // 2) KeywordSession automatically aborts on transport failure, leaving 0 in-flight.
        transport.fail_next = true;
        let mut session = KeywordSession::new(client, transport, rng);
        let err = session.get(0x01, &entries[6].1).unwrap_err();
        assert!(matches!(err, KeywordSessionError::Transport(_)));
        assert_eq!(session.client().queries_in_flight(), 0);

        // Subsequent session get succeeds cleanly.
        let val6 = session.get(0x01, &entries[6].1).unwrap();
        assert_eq!(val6.as_deref(), Some(entries[6].2.as_slice()));

        // Batch transport failure also auto-aborts all segment reservations.
        session.transport_mut().fail_next = true;
        let b_err = session
            .get_batch(&[
                (0x01, entries[7].1.as_slice()),
                (0x01, entries[8].1.as_slice()),
            ])
            .unwrap_err();
        assert!(matches!(b_err, KeywordSessionError::Transport(_)));
        assert_eq!(session.client().queries_in_flight(), 0);
    }

    #[test]
    fn checkpoint_wal_retire_hints_and_corruption_multilevel_and_protocol() {
        let value_size: usize = 16;
        let spec: KeywordTableSpec =
            KeywordTableSpec::with_layout(64, Some(10), value_size, 8, 4, 0.80, [0x88; 32])
                .unwrap();
        let entries: Vec<(u8, Vec<u8>, Vec<u8>)> = (0..64u64)
            .map(|i| {
                let (k, v) = make_kv_pair(i, value_size);
                (0x01, k, v)
            })
            .collect();
        let (table, _) = KvScfTable::build(
            spec.clone(),
            entries
                .iter()
                .map(|(knd, k, v)| (*knd, k.as_slice(), v.as_slice())),
        )
        .unwrap();

        // --- MultiLevelClient checkpoint + WAL retire_hints ---
        let ml_params: MultiLevelParams =
            make_ml_params(spec.buckets_per_segment(), spec.record_size(), 32);
        let ml_seeds: [[u8; 32]; 2] = [[0x61; 32], [0x62; 32]];
        let transport = InMemoryMlTransport {
            table: KvScfTable::from_database_bytes(spec.clone(), table.as_bytes().to_vec())
                .unwrap(),
            params: ml_params.clone(),
            seg_epochs: [0, 0],
            fail_next: false,
        };
        let mut ml_client = init_ml_keyword_client(
            &transport.table,
            &ml_params,
            &ml_seeds,
            CachePolicy::LocalHit,
        );
        let mut rng = ChaCha20Rng::seed_from_u64(4242);

        // Warm one bucket into keyword cache and verify write_checkpoint refuses in-flight.
        let p_inflight = ml_client
            .prepare_lookup(0x01, &entries[0].1, &mut rng)
            .unwrap();
        let LookupPlan::Query { query, pending } = p_inflight else {
            unreachable!()
        };
        assert!(matches!(
            ml_client.to_bytes().unwrap_err(),
            KeywordClientError::QueryInFlight
        ));
        let ans = transport.answer_lookup(&query);
        assert_eq!(
            ml_client
                .finish_lookup(&pending, &ans, &mut rng)
                .unwrap()
                .as_deref(),
            Some(entries[0].2.as_slice())
        );

        // Save idle checkpoint, then run post-checkpoint queries logging WAL hint IDs.
        let ckpt_bytes: Vec<u8> = ml_client.to_bytes().unwrap();
        let mut wal0: Vec<(usize, u32)> = Vec::new();
        let mut wal1: Vec<(usize, u32)> = Vec::new();
        for (_, key, _) in entries.iter().take(6).skip(1) {
            let plan = ml_client.prepare_lookup(0x01, key, &mut rng).unwrap();
            if let LookupPlan::Query { query, pending } = plan {
                let [h0, h1] = pending.hint_ids();
                wal0.push(h0);
                wal1.push(h1);
                let ans = transport.answer_lookup(&query);
                ml_client.finish_lookup(&pending, &ans, &mut rng).unwrap();
            }
        }
        // Also simulate a promoted backup hint_id logged after the checkpoint.
        let promoted_backup_id: (usize, u32) = (0, ml_params.levels[0].num_primary as u32 + 2);
        wal0.push(promoted_backup_id);

        // Restore checkpoint and verify cross-session pending rejection.
        let mut restored =
            KeywordClient::<MultiLevelClient>::from_bytes(&ckpt_bytes, &ml_seeds).unwrap();
        assert_eq!(
            restored.cached_bucket_counts(),
            ml_client.cached_bucket_counts().min([1, 1])
        );
        let stale_p = ml_client
            .prepare_lookup(0x01, &entries[10].1, &mut rng)
            .unwrap();
        if let LookupPlan::Query { mut pending, .. } = stale_p {
            // Wrong session rejects without dropping `pending`, so original owner can still abort it.
            assert!(matches!(
                restored.abort_lookup(&mut pending).unwrap_err(),
                KeywordClientError::SessionMismatch
            ));
            ml_client.abort_lookup(&mut pending).unwrap();
            assert_eq!(ml_client.queries_in_flight(), 0);
        }

        // Invalid retire_hints must fail atomically without mutating either segment.
        assert!(restored.retire_hints(&[(99, 0)], &[]).is_err());

        restored.retire_hints(&wal0, &wal1).unwrap();
        let wal0_set: HashSet<(usize, u32)> = wal0.iter().copied().collect();
        let wal1_set: HashSet<(usize, u32)> = wal1.iter().copied().collect();

        // Future queries on restored client must NEVER reuse any WAL-logged hint_id.
        for (_, key, expected_val) in entries.iter().take(20).skip(11) {
            if let LookupPlan::Query { query, pending } =
                restored.prepare_lookup(0x01, key, &mut rng).unwrap()
            {
                let [h0, h1] = pending.hint_ids();
                assert!(
                    !wal0_set.contains(&h0),
                    "reused WAL-retired seg0 hint {h0:?}"
                );
                assert!(
                    !wal1_set.contains(&h1),
                    "reused WAL-retired seg1 hint {h1:?}"
                );
                let ans = transport.answer_lookup(&query);
                assert_eq!(
                    restored
                        .finish_lookup(&pending, &ans, &mut rng)
                        .unwrap()
                        .as_deref(),
                    Some(expected_val.as_slice())
                );
            }
        }

        // Corrupted checksum and tiny memory budget must be rejected.
        let mut corrupted: Vec<u8> = ckpt_bytes.clone();
        let mid: usize = corrupted.len() / 2;
        corrupted[mid] ^= 0xFF;
        assert!(KeywordClient::<MultiLevelClient>::from_bytes(&corrupted, &ml_seeds).is_err());
        assert!(
            KeywordClient::<MultiLevelClient>::read_checkpoint_with_budget(
                ckpt_bytes.as_slice(),
                &ml_seeds,
                64
            )
            .is_err()
        );

        // --- protocol::Client checkpoint + WAL retire_hints ---
        let p_params = ProtocolParams::new(
            spec.buckets_per_segment(),
            spec.record_size(),
            Some(4),
            16,
            Some(32),
        )
        .unwrap();
        let p_seeds: [[u8; 32]; 2] = [[0x71; 32], [0x72; 32]];
        let q_seeds: [[u8; 32]; 2] = [[0x81; 32], [0x82; 32]];
        let s0 = ProtocolServer::new(p_params.clone(), 0, table.segment_bytes(0).unwrap()).unwrap();
        let s1 = ProtocolServer::new(p_params.clone(), 0, table.segment_bytes(1).unwrap()).unwrap();
        let c0 = ProtocolClient::offline_init(
            p_params.clone(),
            p_seeds[0],
            q_seeds[0],
            0,
            table.segment_bytes(0).unwrap(),
        )
        .unwrap();
        let c1 = ProtocolClient::offline_init(
            p_params,
            p_seeds[1],
            q_seeds[1],
            0,
            table.segment_bytes(1).unwrap(),
        )
        .unwrap();
        let mut p_kw = KeywordClient::new(spec, [c0, c1], CachePolicy::ConstantTraffic).unwrap();
        let p_ckpt: Vec<u8> = p_kw.to_bytes().unwrap();

        let mut p_wal0: Vec<usize> = Vec::new();
        let mut p_wal1: Vec<usize> = Vec::new();
        for (_, key, _) in entries.iter().take(3) {
            let LookupPlan::Query { query, pending } =
                p_kw.prepare_lookup(0x01, key, &mut rng).unwrap()
            else {
                unreachable!()
            };
            let [h0, h1] = pending.hint_ids();
            p_wal0.push(h0);
            p_wal1.push(h1);
            let ans = KeywordLookupAnswer {
                table_epoch: 0,
                state_epoch: 0,
                segments: [
                    s0.answer(&query.segments[0]).unwrap(),
                    s1.answer(&query.segments[1]).unwrap(),
                ],
            };
            p_kw.finish_lookup(&pending, &ans, &mut rng).unwrap();
        }

        let mut p_restored = KeywordClient::<ProtocolClient>::from_bytes(
            &p_ckpt,
            &p_seeds,
            &[[0x91; 32], [0x92; 32]],
        )
        .unwrap();
        p_restored.retire_hints(&p_wal0, &p_wal1).unwrap();
        let p_wal0_set: HashSet<usize> = p_wal0.iter().copied().collect();
        let p_wal1_set: HashSet<usize> = p_wal1.iter().copied().collect();
        for (_, key, expected_val) in entries.iter().take(6).skip(3) {
            let LookupPlan::Query { query, pending } =
                p_restored.prepare_lookup(0x01, key, &mut rng).unwrap()
            else {
                unreachable!()
            };
            let [h0, h1] = pending.hint_ids();
            assert!(!p_wal0_set.contains(&h0));
            assert!(!p_wal1_set.contains(&h1));
            let ans = KeywordLookupAnswer {
                table_epoch: 0,
                state_epoch: 0,
                segments: [
                    s0.answer(&query.segments[0]).unwrap(),
                    s1.answer(&query.segments[1]).unwrap(),
                ],
            };
            assert_eq!(
                p_restored
                    .finish_lookup(&pending, &ans, &mut rng)
                    .unwrap()
                    .as_deref(),
                Some(expected_val.as_slice())
            );
        }
    }

    #[test]
    fn multilevel_failed_level_refresh_and_promote_out_batch_atomicity() {
        use crate::multilevel::{LevelBackend, LevelSpec};
        use rand_chacha::ChaCha20Rng;

        // Part A: single lookup where segment 1 answers on level 1 while segment 1's level 0 has
        // zero backups left and active unreserved slots. Must fail atomically (leaving segment 0
        // untouched) when level 0 draws >= 1 replacement, and succeed when level 0 draws 0.
        let spec: KeywordTableSpec =
            KeywordTableSpec::new(8, 2, 8, 8, Some(8), [0x61; 32], 0, 0).unwrap();
        let rec: usize = spec.record_size();
        let key_a: [u8; 8] = 1u64.to_le_bytes();
        let val_a: [u8; 8] = [0xA1; 8];
        let key_b: [u8; 8] = 2u64.to_le_bytes();
        let val_b: [u8; 8] = [0xB2; 8];
        let (table, _) = KvScfTable::build(
            spec.clone(),
            [
                (0x01, key_a.as_slice(), val_a.as_slice()),
                (0x01, key_b.as_slice(), val_b.as_slice()),
            ],
        )
        .unwrap();

        let ml_params = MultiLevelParams::custom(
            8,
            rec,
            vec![
                LevelSpec {
                    block_size: 2,
                    num_blocks: 4,
                    num_primary: 4,
                    num_backup: 1,
                    backend: LevelBackend::FlatAes,
                },
                LevelSpec {
                    block_size: 2,
                    num_blocks: 4,
                    num_primary: 32,
                    num_backup: 8,
                    backend: LevelBackend::FlatAes,
                },
            ],
        )
        .unwrap();
        let seg_seeds: [[u8; 32]; 2] = [[0xA5; 32], [0x5A; 32]];
        let c0 = MultiLevelClient::hint_init(
            ml_params.clone(),
            &seg_seeds[0],
            table.segment_bytes(0).unwrap(),
            1,
        )
        .unwrap();
        let mut c1 = MultiLevelClient::hint_init(
            ml_params.clone(),
            &seg_seeds[1],
            table.segment_bytes(1).unwrap(),
            1,
        )
        .unwrap();

        // Exhaust segment 1's level 0 backup budget (hint_id = num_primary + 0 = 4) and retire
        // only the level-0 primary hint(s) that match `key_a`'s bucket_seg1, leaving active
        // unreserved slots on level 0 so segment 1 must fall through to level 1.
        let loc_a = spec.locate(0x01, &key_a).unwrap();
        c1.retire_hints(&[(0, 4)]).unwrap();
        let mut prep_rng = ChaCha20Rng::seed_from_u64(42);
        let mut retired_l0: usize = 0;
        loop {
            let p = c1.prepare_query(loc_a.bucket_seg1, &mut prep_rng).unwrap();
            let (lvl, hid) = p.hint_id();
            c1.cancel_unsent_query(&p).unwrap();
            if lvl == 0 {
                c1.retire_hints(&[(0, hid)]).unwrap();
                retired_l0 += 1;
            } else {
                break;
            }
        }
        assert!(
            retired_l0 < 4,
            "at least one active level-0 slot must remain on segment 1"
        );
        assert_eq!(c1.remaining_backups()[0], 0);

        let mut kw = KeywordClient::new(spec.clone(), [c0, c1], CachePolicy::Off).unwrap();
        let LookupPlan::Query { query, mut pending } =
            kw.prepare_lookup(0x01, &key_a, &mut prep_rng).unwrap()
        else {
            unreachable!();
        };
        assert_eq!(pending.hint_ids()[1].0, 1, "segment 1 must match level 1");
        let answer = KeywordLookupAnswer {
            table_epoch: 0,
            state_epoch: 0,
            segments: [
                MultiLevelServer::answer(
                    table.segment_bytes(0).unwrap(),
                    &ml_params,
                    &query.segments[0],
                    0,
                )
                .unwrap(),
                MultiLevelServer::answer(
                    table.segment_bytes(1).unwrap(),
                    &ml_params,
                    &query.segments[1],
                    0,
                )
                .unwrap(),
            ],
        };
        let seg0_backups_before = kw.backends()[0].remaining_backups();

        // Probe read-only plans to pick a seed where segment 1 fails on level-0 refresh and a
        // seed where level 0 draws zero replacements (both tested against the exact same state).
        let p0_ref = pending.segments[0].as_ref().unwrap();
        let p1_ref = pending.segments[1].as_ref().unwrap();
        let mut failing_seed: Option<u64> = None;
        let mut succeeding_seed: Option<u64> = None;
        for seed in 0u64..1024 {
            let mut probe_rng = ChaCha20Rng::seed_from_u64(seed);
            if kw.backends()[0]
                .plan_finish_batch(
                    &[p0_ref],
                    std::slice::from_ref(&answer.segments[0]),
                    &mut probe_rng,
                )
                .is_err()
            {
                continue;
            }
            match kw.backends()[1].plan_finish_batch(
                &[p1_ref],
                std::slice::from_ref(&answer.segments[1]),
                &mut probe_rng,
            ) {
                Err(MultiLevelError::BackupExhausted { level: 0, .. })
                    if failing_seed.is_none() =>
                {
                    failing_seed = Some(seed);
                }
                Ok(_) if succeeding_seed.is_none() => {
                    succeeding_seed = Some(seed);
                }
                _ => {}
            }
            if failing_seed.is_some() && succeeding_seed.is_some() {
                break;
            }
        }
        let fail_seed: u64 =
            failing_seed.expect("must find seed hitting failed-level BackupExhausted");
        let ok_seed: u64 =
            succeeding_seed.expect("must find seed with zero failed-level replacements");

        let mut fail_rng = ChaCha20Rng::seed_from_u64(fail_seed);
        let err = kw
            .finish_lookup(&pending, &answer, &mut fail_rng)
            .unwrap_err();
        assert!(matches!(
            err,
            KeywordClientError::Backend {
                segment: 1,
                error: MultiLevelError::BackupExhausted { level: 0, .. }
            }
        ));
        assert_eq!(kw.backends()[0].queries_in_flight(), 1);
        assert_eq!(kw.backends()[1].queries_in_flight(), 1);
        assert_eq!(kw.backends()[0].remaining_backups(), seg0_backups_before);

        // Zero-replacement failed-level refresh succeeds on the same untouched pending lookup.
        let mut ok_rng = ChaCha20Rng::seed_from_u64(ok_seed);
        let val = kw.finish_lookup(&pending, &answer, &mut ok_rng).unwrap();
        assert_eq!(val.as_deref(), Some(val_a.as_slice()));
        assert_eq!(kw.queries_in_flight(), 0);
        let _ = kw.abort_lookup(&mut pending);

        // Part B: 2-query batch where segment 1 has only 2 backups left on level 0 and query 0
        // triggers a PromoteOut sweep consuming >= 2 backups -> query 1 fails before segment 0 commits.
        let batch_params = MultiLevelParams::custom(
            8,
            rec,
            vec![
                LevelSpec {
                    block_size: 2,
                    num_blocks: 4,
                    num_primary: 4,
                    num_backup: 2,
                    backend: LevelBackend::FlatAes,
                },
                LevelSpec {
                    block_size: 2,
                    num_blocks: 4,
                    num_primary: 32,
                    num_backup: 8,
                    backend: LevelBackend::FlatAes,
                },
            ],
        )
        .unwrap();
        let bc0 = MultiLevelClient::hint_init(
            batch_params.clone(),
            &seg_seeds[0],
            table.segment_bytes(0).unwrap(),
            1,
        )
        .unwrap();
        let bc1 = MultiLevelClient::hint_init(
            batch_params.clone(),
            &seg_seeds[1],
            table.segment_bytes(1).unwrap(),
            1,
        )
        .unwrap();
        let mut bkw = KeywordClient::new(spec, [bc0, bc1], CachePolicy::Off).unwrap();
        let mut b_rng = ChaCha20Rng::seed_from_u64(99);
        let BatchLookupPlan::Query {
            query: bq,
            pending: mut bp,
        } = bkw
            .prepare_batch(
                &[(0x01, key_a.as_slice()), (0x01, key_b.as_slice())],
                &mut b_rng,
            )
            .unwrap()
        else {
            unreachable!();
        };
        let b_ans = KeywordBatchAnswer {
            table_epoch: 0,
            state_epoch: 0,
            segments: [
                bq.segments[0]
                    .iter()
                    .map(|q| {
                        MultiLevelServer::answer(
                            table.segment_bytes(0).unwrap(),
                            &batch_params,
                            q,
                            0,
                        )
                        .unwrap()
                    })
                    .collect(),
                bq.segments[1]
                    .iter()
                    .map(|q| {
                        MultiLevelServer::answer(
                            table.segment_bytes(1).unwrap(),
                            &batch_params,
                            q,
                            0,
                        )
                        .unwrap()
                    })
                    .collect(),
            ],
        };
        let b_seg0_backups_before = bkw.backends()[0].remaining_backups();
        let b_seg1_backups_before = bkw.backends()[1].remaining_backups();

        // Find a seed where segment 0 succeeds and segment 1 exhausts level-0 backups mid-batch
        // (due to PromoteOut sweep on query 0 or failed-level refresh).
        let bp0_refs: Vec<&PendingMultiLevelQuery> =
            bp.segments[0].iter().map(|o| o.as_ref().unwrap()).collect();
        let bp1_refs: Vec<&PendingMultiLevelQuery> =
            bp.segments[1].iter().map(|o| o.as_ref().unwrap()).collect();
        let mut batch_fail_seed: Option<u64> = None;
        for seed in 0u64..1024 {
            let mut probe_rng = ChaCha20Rng::seed_from_u64(seed);
            if bkw.backends()[0]
                .plan_finish_batch(&bp0_refs, &b_ans.segments[0], &mut probe_rng)
                .is_err()
            {
                continue;
            }
            let mut seg1_prefix_rng = probe_rng.clone();
            if bkw.backends()[1]
                .plan_finish_batch(
                    &bp1_refs[..1],
                    &b_ans.segments[1][..1],
                    &mut seg1_prefix_rng,
                )
                .is_err()
            {
                continue;
            }
            if matches!(
                bkw.backends()[1].plan_finish_batch(&bp1_refs, &b_ans.segments[1], &mut probe_rng),
                Err(MultiLevelError::BackupExhausted { .. })
            ) {
                batch_fail_seed = Some(seed);
                break;
            }
        }
        let batch_fail_seed: u64 =
            batch_fail_seed.expect("must find seed exhausting segment 1 backups mid-batch");
        let mut trial_rng = ChaCha20Rng::seed_from_u64(batch_fail_seed);
        let batch_err = bkw.finish_batch(&bp, &b_ans, &mut trial_rng).unwrap_err();
        assert!(matches!(
            batch_err,
            KeywordClientError::Backend {
                segment: 1,
                error: MultiLevelError::BackupExhausted { .. }
            }
        ));
        assert_eq!(bkw.backends()[0].queries_in_flight(), 2);
        assert_eq!(bkw.backends()[1].queries_in_flight(), 2);
        assert_eq!(bkw.backends()[0].remaining_backups(), b_seg0_backups_before);
        assert_eq!(bkw.backends()[1].remaining_backups(), b_seg1_backups_before);
        bkw.abort_batch(&mut bp).unwrap();
        assert_eq!(bkw.queries_in_flight(), 0);
    }

    #[test]
    fn checkpoint_budgets_charge_multilevel_retained_and_protocol_per_block_state() {
        use crate::checked_io::HashingWriter;
        use crate::hints::{seed_check, DatabaseSnapshot, HintGeometry, HintSet};
        use crate::multilevel::{LevelBackend, LevelSpec};
        use crate::protocol::{CLIENT_STATE_MAGIC, CLIENT_STATE_VERSION};

        // 1) MultiLevelClient: retained decoder allocations exceed `hint_memory_bytes()` because
        // of `hint_id_to_slot` and `PrimarySlot` arrays; a budget sized using `hint_memory_bytes()`
        // must be rejected while the exact retained + decode budget succeeds.
        let spec = KeywordTableSpec::new(16, 1, 4, 4, Some(8), [0x33; 32], 0, 0).unwrap();
        let (table, _) = KvScfTable::build(
            spec.clone(),
            [(0x01, &1u64.to_le_bytes()[..], &[7u8; 4][..])],
        )
        .unwrap();
        let ml_params = MultiLevelParams::custom(
            16,
            spec.record_size(),
            vec![LevelSpec {
                block_size: 4,
                num_blocks: 4,
                num_primary: 32,
                num_backup: 16,
                backend: LevelBackend::FlatAes,
            }],
        )
        .unwrap();
        let seeds: [[u8; 32]; 2] = [[0x12; 32], [0x34; 32]];
        let c0 = MultiLevelClient::hint_init(
            ml_params.clone(),
            &seeds[0],
            table.segment_bytes(0).unwrap(),
            1,
        )
        .unwrap();
        let c1 =
            MultiLevelClient::hint_init(ml_params, &seeds[1], table.segment_bytes(1).unwrap(), 1)
                .unwrap();
        let b0_raw: Vec<u8> = c0.to_bytes().unwrap();
        let b1_raw: Vec<u8> = c1.to_bytes().unwrap();
        let (_, retained0): (MultiLevelClient, usize) =
            MultiLevelClient::from_bytes_with_limit_accounted(&b0_raw, &seeds[0], usize::MAX)
                .unwrap();
        let estimate0: usize = c0.hint_memory_bytes();
        assert!(
            retained0 > estimate0,
            "retained0 ({retained0}) must strictly exceed hint_memory_bytes() ({estimate0})"
        );
        let kw = KeywordClient::new(spec, [c0, c1], CachePolicy::Off).unwrap();
        let ckpt: Vec<u8> = kw.to_bytes().unwrap();

        // A budget equal to raw blobs + estimate0 + retained0 leaves less than retained0 for seg1
        // once seg0's true `retained0` is deducted, so segment 1 decode must fail.
        let insufficient_budget: usize = b0_raw.len() + b1_raw.len() + estimate0 + retained0;
        assert!(
            KeywordClient::<MultiLevelClient>::read_checkpoint_with_budget(
                ckpt.as_slice(),
                &seeds,
                insufficient_budget,
            )
            .is_err()
        );

        // 2) ProtocolClient: tiny state blob with huge `num_blocks` must be rejected by
        // `read_state_with_limit` BEFORE allocating millions of per-block `Iprf` structs.
        let master_seed: [u8; 32] = [0x77; 32];
        let query_seed: [u8; 32] = [0x88; 32];
        let huge_num_entries: u64 = 1u64 << 24;
        let small_block_size: u64 = 2;
        let huge_num_blocks: u64 = huge_num_entries / small_block_size;
        let tiny_hints = HintSet::new(
            HintGeometry {
                num_entries: huge_num_entries,
                record_size: 8,
                block_size: small_block_size,
                num_blocks: huge_num_blocks,
            },
            seed_check(&master_seed),
            DatabaseSnapshot {
                version: 0,
                hash: [0u8; 32],
            },
            vec![1u32; 2], // lambda = 1, num_regular = 2 -> tiny 24-byte hint payload!
            vec![1u32; 1], // num_backup = 1 (required > 0) -> tiny 20-byte backup payload!
        )
        .unwrap();
        let mut tiny_state: Vec<u8> = Vec::new();
        {
            let mut out = HashingWriter::new(&mut tiny_state);
            out.write_hashed(&CLIENT_STATE_MAGIC).unwrap();
            out.write_hashed(&CLIENT_STATE_VERSION.to_le_bytes())
                .unwrap();
            tiny_hints.write_body(&mut out).unwrap();
            out.write_hashed(&[0u8; 1]).unwrap(); // sent bitset (2 regular hints -> 1 byte)
            out.write_hashed(&0u64.to_le_bytes()).unwrap(); // num_answered
            out.write_hashed(&0u64.to_le_bytes()).unwrap(); // num_in_flight
            out.write_hashed(&0u64.to_le_bytes()).unwrap(); // num_retired
            out.finish().unwrap();
        }
        assert!(
            tiny_state.len() < 512,
            "crafted state blob must be tiny ({} bytes)",
            tiny_state.len()
        );
        let err = ProtocolClient::read_state_with_limit(
            master_seed,
            query_seed,
            tiny_state.as_slice(),
            1024 * 1024, // 1 MiB limit, >> tiny_state.len(), << 8.39M * size_of::<Iprf>()
        )
        .err()
        .expect("huge num_blocks must be rejected before per-block Iprf allocation");
        assert!(
            matches!(
                err,
                ClientStateError::Invalid("client state exceeds memory budget")
            ),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn duplicate_tag_match_rejects_before_backend_commit_or_cache_poisoning() {
        use crate::multilevel::{LevelBackend, LevelSpec};
        use rand_chacha::ChaCha20Rng;

        let spec = KeywordTableSpec::new(8, 2, 8, 8, Some(8), [0x51; 32], 0, 0).unwrap();
        let rec: usize = spec.record_size();
        let key: [u8; 8] = 7u64.to_le_bytes();
        let val: [u8; 8] = [0x42; 8];
        let (table, _) =
            KvScfTable::build(spec.clone(), [(0x01, key.as_slice(), val.as_slice())]).unwrap();
        let loc = spec.locate(0x01, &key).unwrap();

        let ml_params = MultiLevelParams::custom(
            8,
            rec,
            vec![LevelSpec {
                block_size: 2,
                num_blocks: 4,
                num_primary: 16,
                num_backup: 8,
                backend: LevelBackend::FlatAes,
            }],
        )
        .unwrap();
        let seg_seeds: [[u8; 32]; 2] = [[0x21; 32], [0x22; 32]];
        let c0 = MultiLevelClient::hint_init(
            ml_params.clone(),
            &seg_seeds[0],
            table.segment_bytes(0).unwrap(),
            1,
        )
        .unwrap();
        let c1 = MultiLevelClient::hint_init(
            ml_params.clone(),
            &seg_seeds[1],
            table.segment_bytes(1).unwrap(),
            1,
        )
        .unwrap();
        let mut kw = KeywordClient::new(spec, [c0, c1], CachePolicy::LocalHit).unwrap();
        let mut rng = ChaCha20Rng::seed_from_u64(123);

        // Craft a malicious server parity delta that turns the honest bucket at `loc.bucket_seg0`
        // into one where both slots carry `loc.tag` (reconstructing a `DuplicateTagMatch`).
        let base: usize = loc.bucket_seg0 * rec;
        let honest_bucket: &[u8] = &table.segment_bytes(0).unwrap()[base..base + rec];
        let mut evil_bucket: Vec<u8> = vec![0u8; rec];
        evil_bucket[0..8].copy_from_slice(&loc.tag);
        evil_bucket[8..16].copy_from_slice(&[0xEE; 8]);
        evil_bucket[16..24].copy_from_slice(&loc.tag);
        evil_bucket[24..32].copy_from_slice(&[0xFF; 8]);
        let tamper_delta: Vec<u8> = honest_bucket
            .iter()
            .zip(evil_bucket.iter())
            .map(|(a, b)| a ^ b)
            .collect();
        let tamper_answer = |mut ans: MultiLevelAnswer| -> MultiLevelAnswer {
            for (p, d) in ans.parity_0.iter_mut().zip(tamper_delta.iter()) {
                *p ^= *d;
            }
            for (p, d) in ans.parity_1.iter_mut().zip(tamper_delta.iter()) {
                *p ^= *d;
            }
            ans
        };

        // 1) Single lookup: DuplicateTagMatch must reject before commit and before caching.
        let backups0_before = kw.backends()[0].remaining_backups();
        let backups1_before = kw.backends()[1].remaining_backups();
        let LookupPlan::Query { query, mut pending } =
            kw.prepare_lookup(0x01, &key, &mut rng).unwrap()
        else {
            unreachable!();
        };
        let evil_ans = KeywordLookupAnswer {
            table_epoch: 0,
            state_epoch: 0,
            segments: [
                tamper_answer(
                    MultiLevelServer::answer(
                        table.segment_bytes(0).unwrap(),
                        &ml_params,
                        &query.segments[0],
                        0,
                    )
                    .unwrap(),
                ),
                MultiLevelServer::answer(
                    table.segment_bytes(1).unwrap(),
                    &ml_params,
                    &query.segments[1],
                    0,
                )
                .unwrap(),
            ],
        };
        assert!(matches!(
            kw.finish_lookup(&pending, &evil_ans, &mut rng),
            Err(KeywordClientError::Keyword(KeywordError::DuplicateTagMatch))
        ));
        assert_eq!(kw.backends()[0].queries_in_flight(), 1);
        assert_eq!(kw.backends()[1].queries_in_flight(), 1);
        assert_eq!(kw.backends()[0].remaining_backups(), backups0_before);
        assert_eq!(kw.backends()[1].remaining_backups(), backups1_before);
        assert_eq!(kw.cached_bucket_counts(), [0, 0]);
        kw.abort_lookup(&mut pending).unwrap();
        assert_eq!(kw.queries_in_flight(), 0);

        // 2) Batch lookup: DuplicateTagMatch must also reject before commit and before caching.
        let BatchLookupPlan::Query {
            query: bq,
            pending: mut bp,
        } = kw
            .prepare_batch(&[(0x01, key.as_slice())], &mut rng)
            .unwrap()
        else {
            unreachable!();
        };
        let evil_batch_ans = KeywordBatchAnswer {
            table_epoch: 0,
            state_epoch: 0,
            segments: [
                vec![tamper_answer(
                    MultiLevelServer::answer(
                        table.segment_bytes(0).unwrap(),
                        &ml_params,
                        &bq.segments[0][0],
                        0,
                    )
                    .unwrap(),
                )],
                vec![MultiLevelServer::answer(
                    table.segment_bytes(1).unwrap(),
                    &ml_params,
                    &bq.segments[1][0],
                    0,
                )
                .unwrap()],
            ],
        };
        assert!(matches!(
            kw.finish_batch(&bp, &evil_batch_ans, &mut rng),
            Err(KeywordClientError::Keyword(KeywordError::DuplicateTagMatch))
        ));
        assert_eq!(kw.backends()[0].queries_in_flight(), 1);
        assert_eq!(kw.backends()[1].queries_in_flight(), 1);
        assert_eq!(kw.cached_bucket_counts(), [0, 0]);
        kw.abort_batch(&mut bp).unwrap();
        assert_eq!(kw.queries_in_flight(), 0);

        // Next lookup under LocalHit must still issue a wire query (cache was never poisoned)
        // and return the honest value when answered honestly.
        let LookupPlan::Query {
            query: ok_q,
            pending: ok_p,
        } = kw.prepare_lookup(0x01, &key, &mut rng).unwrap()
        else {
            panic!("cache must not have been poisoned by rejected DuplicateTagMatch");
        };
        let ok_ans = KeywordLookupAnswer {
            table_epoch: 0,
            state_epoch: 0,
            segments: [
                MultiLevelServer::answer(
                    table.segment_bytes(0).unwrap(),
                    &ml_params,
                    &ok_q.segments[0],
                    0,
                )
                .unwrap(),
                MultiLevelServer::answer(
                    table.segment_bytes(1).unwrap(),
                    &ml_params,
                    &ok_q.segments[1],
                    0,
                )
                .unwrap(),
            ],
        };
        assert_eq!(
            kw.finish_lookup(&ok_p, &ok_ans, &mut rng)
                .unwrap()
                .as_deref(),
            Some(val.as_slice())
        );
    }
}
