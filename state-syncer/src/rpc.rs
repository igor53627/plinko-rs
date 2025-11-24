use eyre::Result;
use reqwest::blocking::Client;
use serde_json::json;
use serde::Deserialize;
use crate::update_manager::DBUpdate;
use crate::db::DB_ENTRY_U64_COUNT;
use std::collections::HashSet;

pub struct EthClient {
    url: String,
    client: Client,
}

#[derive(Deserialize)]
struct RpcResponse<T> {
    result: Option<T>,
    error: Option<serde_json::Value>,
}

#[derive(Deserialize)]
struct Block {
    transactions: Vec<Transaction>,
}

#[derive(Deserialize)]
struct Transaction {
    from: String,
    to: Option<String>,
}

impl EthClient {
    pub fn new(url: String) -> Self {
        Self {
            url,
            client: Client::new(),
        }
    }

    fn call<T: for<'de> Deserialize<'de>>(&self, method: &str, params: serde_json::Value) -> Result<T> {
        let body = json!({
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
            "id": 1
        });

        let resp: RpcResponse<T> = self.client.post(&self.url)
            .json(&body)
            .send()?
            .json()?;

        if let Some(err) = resp.error {
            return Err(eyre::eyre!("RPC Error: {:?}", err));
        }
        
        resp.result.ok_or_else(|| eyre::eyre!("RPC returned null result"))
    }

    pub fn block_number(&self) -> Result<u64> {
        let hex: String = self.call("eth_blockNumber", json!([]))?;
        u64::from_str_radix(hex.trim_start_matches("0x"), 16).map_err(|e| e.into())
    }

    pub fn get_block_transactions(&self, block_number: u64) -> Result<Vec<Transaction>> {
        let hex_num = format!("0x{:x}", block_number);
        let block: Block = self.call("eth_getBlockByNumber", json!([hex_num, true]))?;
        Ok(block.transactions)
    }

    pub fn get_balance(&self, address: &str, block_number: u64) -> Result<u128> {
        let hex_num = format!("0x{:x}", block_number);
        let hex: String = self.call("eth_getBalance", json!([address, hex_num]))?;
        u128::from_str_radix(hex.trim_start_matches("0x"), 16).map_err(|e| e.into())
    }
}

pub fn fetch_updates_rpc(client: &EthClient, block_number: u64, manager: &crate::update_manager::UpdateManager, address_mapping: &std::collections::HashMap<String, u64>) -> Result<Vec<DBUpdate>> {
    // 1. Get block transactions to find touched addresses
    let txs = client.get_block_transactions(block_number)?;
    
    let mut addresses = HashSet::new();
    for tx in txs {
        addresses.insert(tx.from.to_lowercase());
        if let Some(to) = tx.to {
            addresses.insert(to.to_lowercase());
        }
    }

    if addresses.is_empty() {
        return Ok(vec![]);
    }

    let mut updates = Vec::new();
    
    // 2. For each touched address, check if it's in our DB and if balance changed
    for addr in addresses {
        if let Some(&index) = address_mapping.get(&addr) {
            // Fetch new balance
            let balance = client.get_balance(&addr, block_number)?;
            
            // Get old value from DB
            let old_value = manager.get_value(index).unwrap_or([0; 4]);
            
            // New value: [Nonce (skip), Balance, CodeHash (skip), Padding]
            // Currently we only update Balance for this PoC
            // Word 1 is Balance.
            // Note: `database.bin` layout:
            // Word 0: Nonce
            // Word 1: Balance
            // Word 2: CodeHash
            // Word 3: Padding
            
            // We need to preserve other words.
            let mut new_value = old_value;
            
            // Encode u128 balance into u256 (32 bytes)
            // Our DB stores u64 words.
            // Balance is Word 1 (index + 1 in flattened, but `old_value` is [u64; 4])
            // Wait, Balance is U256. It takes 32 bytes.
            // In `plinko-extractor` main.rs:
            //   writer.write_all(&account.balance.to_le_bytes::<32>())?;
            // This writes 4 u64s? No, it writes 32 bytes.
            
            // In `database.bin` (flat u64s):
            // Entry = 4 * u64 = 32 bytes.
            // Wait.
            // In Extractor:
            // Accounts occupy 4 consecutive ENTRIES? No.
            // "Accounts: occupy 4 consecutive entries (128 bytes)."
            // "Word 0: Nonce", "Word 1: Balance", "Word 2: CodeHash", "Word 3: Padding"
            
            // In `state-syncer` DB:
            // `DB_ENTRY_SIZE` = 32 bytes.
            // `DB_ENTRY_U64_COUNT` = 4.
            // So `DBEntry` = 32 bytes.
            
            // Is an Account ONE entry (32 bytes) or FOUR entries (128 bytes)?
            // README says: "Accounts: occupy 4 consecutive entries (128 bytes)."
            
            // So an Account takes indices `i, i+1, i+2, i+3`.
            // The `address-mapping.bin` points to the START index.
            
            // My `UpdateManager` logic works on `DBUpdate` which has `[u64; 4]`.
            // This is 32 bytes.
            // So `DBUpdate` updates ONE entry.
            
            // If I want to update Balance, I need to update the Entry at `index + 1`.
            
            // Wait, `balance` is u256 (32 bytes).
            // It occupies ONE entire entry (32 bytes).
            
            // So:
            // Index = Account Base Index
            // Index + 0 = Nonce (32 bytes)
            // Index + 1 = Balance (32 bytes)
            // Index + 2 = CodeHash (32 bytes)
            // Index + 3 = Padding (32 bytes)
            
            // So to update balance, I generate an update for `index + 1`.
            
            let balance_idx = index + 1;
            let old_balance_entry = manager.get_value(balance_idx).unwrap_or([0; 4]);
            
            // Convert new balance (u128) to [u64; 4] (u256 LE)
            let mut new_balance_entry = [0u64; 4];
            new_balance_entry[0] = balance as u64;
            new_balance_entry[1] = (balance >> 64) as u64;
            // [2] and [3] are 0 because u128 fits in 2 u64s.
            
            if old_balance_entry != new_balance_entry {
                updates.push(DBUpdate {
                    index: balance_idx,
                    old_value: old_balance_entry,
                    new_value: new_balance_entry,
                });
            }
        }
    }
    
    Ok(updates)
}
