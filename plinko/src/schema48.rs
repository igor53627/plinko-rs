//! 48-byte PIR entry schema for GPU-optimized memory coalescing.
//!
//! This module defines the new 48-byte (384-bit) entry format that provides:
//! - Uniform row size for both accounts and storage (simplifies PIR geometry)
//! - 3×16B alignment for optimal GPU memory coalescing
//! - Cuckoo TAG for client-side collision resolution (2-server PIR compatibility)
//! - Code ID indirection for bytecode deduplication
//!
//! # Layout
//!
//! ## Account Entry (48 bytes)
//! ```text
//! ┌─────────────────┬──────────┬────────┬─────────┬─────────────┐
//! │    Balance      │  Nonce   │ CodeID │   TAG   │   Padding   │
//! │    16 bytes     │  8 bytes │ 4 bytes│ 8 bytes │  12 bytes   │
//! └─────────────────┴──────────┴────────┴─────────┴─────────────┘
//! ```
//!
//! ## Storage Entry (48 bytes)
//! ```text
//! ┌────────────────────────────────────┬─────────┬─────────────┐
//! │           Storage Value            │   TAG   │   Padding   │
//! │             32 bytes               │ 8 bytes │  8 bytes    │
//! └────────────────────────────────────┴─────────┴─────────────┘
//! ```


/// Entry size in bytes (48 = 3 × 16 for GPU alignment)
pub const ENTRY_SIZE: usize = 48;

/// Entry size as u64 count (48 / 8 = 6)
pub const ENTRY_U64_COUNT: usize = 6;

/// Balance field size (128-bit, clamped from 256-bit)
pub const BALANCE_SIZE: usize = 16;

/// Nonce field size
pub const NONCE_SIZE: usize = 8;

/// Code ID field size (index into bytecode store)
pub const CODE_ID_SIZE: usize = 4;

/// TAG field size (Cuckoo fingerprint)
pub const TAG_SIZE: usize = 8;

/// Account padding size
pub const ACCOUNT_PADDING_SIZE: usize = 12;

/// Storage value size (full 256-bit word)
pub const STORAGE_VALUE_SIZE: usize = 32;

/// Storage padding size
pub const STORAGE_PADDING_SIZE: usize = 8;

/// Special Code ID indicating no bytecode (EOA account)
pub const CODE_ID_NONE: u32 = 0;

/// 8-byte TAG for Cuckoo collision resolution.
///
/// Derived from truncated hash of the address (for accounts) or
/// (address || slot_key) for storage entries.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(transparent)]
pub struct Tag(pub [u8; TAG_SIZE]);

impl Tag {
    /// Create TAG from address bytes (for account entries).
    ///
    /// Uses first 8 bytes of keccak256(address).
    pub fn from_address(address: &[u8; 20]) -> Self {
        use tiny_keccak::{Hasher as KeccakHasher, Keccak};
        let mut hasher = Keccak::v256();
        hasher.update(address);
        let mut hash = [0u8; 32];
        hasher.finalize(&mut hash);
        let mut tag = [0u8; TAG_SIZE];
        tag.copy_from_slice(&hash[..TAG_SIZE]);
        Tag(tag)
    }

    /// Create TAG from address and storage slot key (for storage entries).
    ///
    /// Uses first 8 bytes of keccak256(address || slot_key).
    pub fn from_address_slot(address: &[u8; 20], slot_key: &[u8; 32]) -> Self {
        use tiny_keccak::{Hasher as KeccakHasher, Keccak};
        let mut hasher = Keccak::v256();
        hasher.update(address);
        hasher.update(slot_key);
        let mut hash = [0u8; 32];
        hasher.finalize(&mut hash);
        let mut tag = [0u8; TAG_SIZE];
        tag.copy_from_slice(&hash[..TAG_SIZE]);
        Tag(tag)
    }

    /// Create TAG from raw bytes.
    pub fn from_bytes(bytes: [u8; TAG_SIZE]) -> Self {
        Tag(bytes)
    }

    /// Get TAG as bytes.
    pub fn as_bytes(&self) -> &[u8; TAG_SIZE] {
        &self.0
    }
}

/// 4-byte Code ID referencing bytecode in external store.
///
/// Code ID 0 is reserved for EOA accounts (no bytecode).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct CodeId(pub u32);

impl CodeId {
    /// Create a new Code ID.
    pub fn new(id: u32) -> Self {
        CodeId(id)
    }

    /// Check if this is an EOA (no bytecode).
    pub fn is_eoa(&self) -> bool {
        self.0 == CODE_ID_NONE
    }

    /// Get the raw ID value.
    pub fn as_u32(&self) -> u32 {
        self.0
    }

    /// Convert to little-endian bytes.
    pub fn to_le_bytes(&self) -> [u8; CODE_ID_SIZE] {
        self.0.to_le_bytes()
    }

    /// Create from little-endian bytes.
    pub fn from_le_bytes(bytes: [u8; CODE_ID_SIZE]) -> Self {
        CodeId(u32::from_le_bytes(bytes))
    }
}

/// 48-byte Account Entry.
///
/// Layout:
/// - bytes 0..16:  Balance (128-bit LE, clamped from 256-bit)
/// - bytes 16..24: Nonce (64-bit LE)
/// - bytes 24..28: Code ID (32-bit LE, index into bytecode store)
/// - bytes 28..36: TAG (8 bytes, Cuckoo fingerprint)
/// - bytes 36..48: Padding (12 bytes, zeroed)
#[derive(Clone, Copy, Debug)]
#[repr(C, align(8))]
pub struct AccountEntry48 {
    /// Account balance, clamped to 128-bit.
    /// Ethereum balances practically fit in 128 bits (max ~120 million ETH × 10^18).
    pub balance: [u8; BALANCE_SIZE],
    /// Account nonce.
    pub nonce: u64,
    /// Index into bytecode store. 0 = EOA (no bytecode).
    pub code_id: CodeId,
    /// Cuckoo TAG for collision resolution.
    pub tag: Tag,
    /// Padding for 48-byte alignment.
    pub _padding: [u8; ACCOUNT_PADDING_SIZE],
}

impl Default for AccountEntry48 {
    fn default() -> Self {
        Self {
            balance: [0u8; BALANCE_SIZE],
            nonce: 0,
            code_id: CodeId(CODE_ID_NONE),
            tag: Tag::default(),
            _padding: [0u8; ACCOUNT_PADDING_SIZE],
        }
    }
}

impl AccountEntry48 {
    /// Create a new account entry.
    ///
    /// # Arguments
    /// * `balance_256` - Full 256-bit balance (will be clamped to 128-bit)
    /// * `nonce` - Account nonce
    /// * `code_id` - Index into bytecode store (0 for EOA)
    /// * `address` - Account address (used to derive TAG)
    pub fn new(balance_256: &[u8; 32], nonce: u64, code_id: CodeId, address: &[u8; 20]) -> Self {
        let mut balance = [0u8; BALANCE_SIZE];
        // Take lower 128 bits (little-endian, so first 16 bytes)
        // For balances that fit in 128 bits, this is lossless
        balance.copy_from_slice(&balance_256[..BALANCE_SIZE]);

        Self {
            balance,
            nonce,
            code_id,
            tag: Tag::from_address(address),
            _padding: [0u8; ACCOUNT_PADDING_SIZE],
        }
    }

    /// Serialize to 48-byte array.
    pub fn to_bytes(&self) -> [u8; ENTRY_SIZE] {
        let mut bytes = [0u8; ENTRY_SIZE];
        bytes[0..16].copy_from_slice(&self.balance);
        bytes[16..24].copy_from_slice(&self.nonce.to_le_bytes());
        bytes[24..28].copy_from_slice(&self.code_id.to_le_bytes());
        bytes[28..36].copy_from_slice(self.tag.as_bytes());
        // bytes[36..48] remain zero (padding)
        bytes
    }

    /// Deserialize from 48-byte array.
    pub fn from_bytes(bytes: &[u8; ENTRY_SIZE]) -> Self {
        let mut balance = [0u8; BALANCE_SIZE];
        balance.copy_from_slice(&bytes[0..16]);

        let nonce = u64::from_le_bytes(bytes[16..24].try_into().unwrap());

        let mut code_id_bytes = [0u8; CODE_ID_SIZE];
        code_id_bytes.copy_from_slice(&bytes[24..28]);
        let code_id = CodeId::from_le_bytes(code_id_bytes);

        let mut tag_bytes = [0u8; TAG_SIZE];
        tag_bytes.copy_from_slice(&bytes[28..36]);
        let tag = Tag::from_bytes(tag_bytes);

        Self {
            balance,
            nonce,
            code_id,
            tag,
            _padding: [0u8; ACCOUNT_PADDING_SIZE],
        }
    }
}

/// 48-byte Storage Entry.
///
/// Layout:
/// - bytes 0..32:  Storage Value (256-bit LE)
/// - bytes 32..40: TAG (8 bytes, Cuckoo fingerprint)
/// - bytes 40..48: Padding (8 bytes, zeroed)
#[derive(Clone, Copy, Debug)]
#[repr(C, align(8))]
pub struct StorageEntry48 {
    /// Storage value (full 256-bit word).
    pub value: [u8; STORAGE_VALUE_SIZE],
    /// Cuckoo TAG for collision resolution.
    pub tag: Tag,
    /// Padding for 48-byte alignment.
    pub _padding: [u8; STORAGE_PADDING_SIZE],
}

impl Default for StorageEntry48 {
    fn default() -> Self {
        Self {
            value: [0u8; STORAGE_VALUE_SIZE],
            tag: Tag::default(),
            _padding: [0u8; STORAGE_PADDING_SIZE],
        }
    }
}

impl StorageEntry48 {
    /// Create a new storage entry.
    ///
    /// # Arguments
    /// * `value` - 256-bit storage value
    /// * `address` - Account address
    /// * `slot_key` - Storage slot key
    pub fn new(value: &[u8; 32], address: &[u8; 20], slot_key: &[u8; 32]) -> Self {
        Self {
            value: *value,
            tag: Tag::from_address_slot(address, slot_key),
            _padding: [0u8; STORAGE_PADDING_SIZE],
        }
    }

    /// Serialize to 48-byte array.
    pub fn to_bytes(&self) -> [u8; ENTRY_SIZE] {
        let mut bytes = [0u8; ENTRY_SIZE];
        bytes[0..32].copy_from_slice(&self.value);
        bytes[32..40].copy_from_slice(self.tag.as_bytes());
        // bytes[40..48] remain zero (padding)
        bytes
    }

    /// Deserialize from 48-byte array.
    pub fn from_bytes(bytes: &[u8; ENTRY_SIZE]) -> Self {
        let mut value = [0u8; STORAGE_VALUE_SIZE];
        value.copy_from_slice(&bytes[0..32]);

        let mut tag_bytes = [0u8; TAG_SIZE];
        tag_bytes.copy_from_slice(&bytes[32..40]);
        let tag = Tag::from_bytes(tag_bytes);

        Self {
            value,
            tag,
            _padding: [0u8; STORAGE_PADDING_SIZE],
        }
    }
}

/// Code ID store for bytecode hash deduplication.
///
/// Maps Code ID (u32) to bytecode hash (32 bytes).
/// Code ID 0 is reserved and should not appear in the store.
#[derive(Debug, Default)]
pub struct CodeStore {
    /// Bytecode hashes indexed by Code ID - 1 (since ID 0 is reserved).
    hashes: Vec<[u8; 32]>,
    /// Reverse lookup: hash -> Code ID.
    hash_to_id: std::collections::HashMap<[u8; 32], CodeId>,
}

impl CodeStore {
    /// Create a new empty code store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Get or create a Code ID for the given bytecode hash.
    ///
    /// Returns `CodeId(0)` if `hash` is None (EOA account).
    pub fn get_or_insert(&mut self, hash: Option<&[u8; 32]>) -> CodeId {
        let hash = match hash {
            Some(h) if *h != [0u8; 32] => h,
            _ => return CodeId(CODE_ID_NONE),
        };

        if let Some(&id) = self.hash_to_id.get(hash) {
            return id;
        }

        // Allocate new ID (1-indexed since 0 is reserved)
        let id = CodeId((self.hashes.len() + 1) as u32);
        self.hashes.push(*hash);
        self.hash_to_id.insert(*hash, id);
        id
    }

    /// Look up bytecode hash by Code ID.
    ///
    /// Returns `None` for Code ID 0 (EOA) or invalid IDs.
    pub fn get(&self, id: CodeId) -> Option<&[u8; 32]> {
        if id.is_eoa() {
            return None;
        }
        self.hashes.get((id.0 - 1) as usize)
    }

    /// Number of unique bytecode hashes stored.
    pub fn len(&self) -> usize {
        self.hashes.len()
    }

    /// Check if store is empty.
    pub fn is_empty(&self) -> bool {
        self.hashes.is_empty()
    }

    /// Serialize code store to binary format.
    ///
    /// Format: [count: u32][hash0: 32B][hash1: 32B]...
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(4 + self.hashes.len() * 32);
        bytes.extend_from_slice(&(self.hashes.len() as u32).to_le_bytes());
        for hash in &self.hashes {
            bytes.extend_from_slice(hash);
        }
        bytes
    }

    /// Deserialize code store from binary format.
    pub fn from_bytes(bytes: &[u8]) -> eyre::Result<Self> {
        use eyre::ensure;

        ensure!(bytes.len() >= 4, "Code store too short");
        let count = u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
        ensure!(
            bytes.len() == 4 + count * 32,
            "Code store size mismatch: expected {}, got {}",
            4 + count * 32,
            bytes.len()
        );

        let mut store = Self::new();
        for i in 0..count {
            let offset = 4 + i * 32;
            let mut hash = [0u8; 32];
            hash.copy_from_slice(&bytes[offset..offset + 32]);
            store.hashes.push(hash);
            store.hash_to_id.insert(hash, CodeId((i + 1) as u32));
        }
        Ok(store)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entry_sizes() {
        assert_eq!(ENTRY_SIZE, 48);
        assert_eq!(
            BALANCE_SIZE + NONCE_SIZE + CODE_ID_SIZE + TAG_SIZE + ACCOUNT_PADDING_SIZE,
            ENTRY_SIZE
        );
        assert_eq!(
            STORAGE_VALUE_SIZE + TAG_SIZE + STORAGE_PADDING_SIZE,
            ENTRY_SIZE
        );
        assert_eq!(std::mem::size_of::<AccountEntry48>(), ENTRY_SIZE);
        assert_eq!(std::mem::size_of::<StorageEntry48>(), ENTRY_SIZE);
    }

    #[test]
    fn test_account_entry_roundtrip() {
        let balance = [0x42u8; 32];
        let nonce = 12345u64;
        let address = [0xABu8; 20];
        let code_id = CodeId::new(999);

        let entry = AccountEntry48::new(&balance, nonce, code_id, &address);
        let bytes = entry.to_bytes();
        let recovered = AccountEntry48::from_bytes(&bytes);

        assert_eq!(entry.balance, recovered.balance);
        assert_eq!(entry.nonce, recovered.nonce);
        assert_eq!(entry.code_id, recovered.code_id);
        assert_eq!(entry.tag, recovered.tag);
    }

    #[test]
    fn test_storage_entry_roundtrip() {
        let value = [0x77u8; 32];
        let address = [0xABu8; 20];
        let slot_key = [0xCDu8; 32];

        let entry = StorageEntry48::new(&value, &address, &slot_key);
        let bytes = entry.to_bytes();
        let recovered = StorageEntry48::from_bytes(&bytes);

        assert_eq!(entry.value, recovered.value);
        assert_eq!(entry.tag, recovered.tag);
    }

    #[test]
    fn test_tag_deterministic() {
        let address = [0x12u8; 20];
        let tag1 = Tag::from_address(&address);
        let tag2 = Tag::from_address(&address);
        assert_eq!(tag1, tag2);

        let slot = [0x34u8; 32];
        let tag3 = Tag::from_address_slot(&address, &slot);
        let tag4 = Tag::from_address_slot(&address, &slot);
        assert_eq!(tag3, tag4);

        // Different inputs produce different tags
        let other_address = [0x56u8; 20];
        let tag5 = Tag::from_address(&other_address);
        assert_ne!(tag1, tag5);
    }

    #[test]
    fn test_code_store() {
        let mut store = CodeStore::new();

        // EOA returns 0
        assert_eq!(store.get_or_insert(None), CodeId(0));
        assert_eq!(store.get_or_insert(Some(&[0u8; 32])), CodeId(0));

        // First real hash gets ID 1
        let hash1 = [0x11u8; 32];
        let id1 = store.get_or_insert(Some(&hash1));
        assert_eq!(id1, CodeId(1));

        // Same hash returns same ID
        let id1_again = store.get_or_insert(Some(&hash1));
        assert_eq!(id1_again, CodeId(1));

        // Different hash gets ID 2
        let hash2 = [0x22u8; 32];
        let id2 = store.get_or_insert(Some(&hash2));
        assert_eq!(id2, CodeId(2));

        // Lookup works
        assert_eq!(store.get(CodeId(0)), None);
        assert_eq!(store.get(CodeId(1)), Some(&hash1));
        assert_eq!(store.get(CodeId(2)), Some(&hash2));
        assert_eq!(store.get(CodeId(999)), None);

        assert_eq!(store.len(), 2);
    }

    #[test]
    fn test_code_store_serialization() {
        let mut store = CodeStore::new();
        store.get_or_insert(Some(&[0x11u8; 32]));
        store.get_or_insert(Some(&[0x22u8; 32]));
        store.get_or_insert(Some(&[0x33u8; 32]));

        let bytes = store.to_bytes();
        let recovered = CodeStore::from_bytes(&bytes).unwrap();

        assert_eq!(store.len(), recovered.len());
        assert_eq!(store.get(CodeId(1)), recovered.get(CodeId(1)));
        assert_eq!(store.get(CodeId(2)), recovered.get(CodeId(2)));
        assert_eq!(store.get(CodeId(3)), recovered.get(CodeId(3)));
    }

    #[test]
    fn test_balance_clamping() {
        // Balance that fits in 128 bits
        let mut balance_small = [0u8; 32];
        balance_small[0] = 0xFF;
        balance_small[15] = 0xFF;

        let entry = AccountEntry48::new(&balance_small, 0, CodeId(0), &[0u8; 20]);
        assert_eq!(&entry.balance[..], &balance_small[..16]);

        // Large balance (upper 128 bits set) - gets truncated
        let balance_large = [0xFFu8; 32];
        let entry2 = AccountEntry48::new(&balance_large, 0, CodeId(0), &[0u8; 20]);
        // Only lower 16 bytes preserved
        assert_eq!(&entry2.balance[..], &[0xFFu8; 16]);
    }
}
