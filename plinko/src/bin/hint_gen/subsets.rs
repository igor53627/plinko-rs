//! Parity helpers for hint generation.
//!
//! Block subsets are defined in the library module `plinko::subsets`.

/// XORs `src` into `dst` in place (32-byte hint parity words).
pub fn xor_32(dst: &mut [u8; 32], src: &[u8; 32]) {
    for i in 0..32 {
        dst[i] ^= src[i];
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hint_gen::types::{parity_word_at, DB_ENTRY_SIZE, PARITY_WORD_SIZE};
    use plinko::schema40::AccountEntry40;

    #[test]
    fn parity_word_reads_first_32_bytes_of_v3_entry() {
        let address = [0x11u8; 20];
        let balance = [0u8; 32];
        let entry = AccountEntry40::new(&balance, 7, plinko::schema40::CodeId(0), &address);
        let bytes = entry.to_bytes();
        assert_eq!(bytes.len(), DB_ENTRY_SIZE);

        let mut db = vec![0u8; DB_ENTRY_SIZE * 3];
        db[DB_ENTRY_SIZE..DB_ENTRY_SIZE * 2].copy_from_slice(&bytes);

        assert_eq!(parity_word_at(&db, 1), bytes[..PARITY_WORD_SIZE]);
    }

    #[test]
    fn test_xor_32_identity() {
        let mut a = [0xABu8; 32];
        let b = [0u8; 32];
        let original = a;
        xor_32(&mut a, &b);
        assert_eq!(a, original);
    }

    #[test]
    fn test_xor_32_inverse() {
        let mut a = [0x12u8; 32];
        let b = [0x34u8; 32];
        let original = a;
        xor_32(&mut a, &b);
        xor_32(&mut a, &b);
        assert_eq!(a, original);
    }
}
