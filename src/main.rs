use clap::Parser;
use eyre::Result;
use indicatif::{ProgressBar, ProgressStyle};
use plinko::schema48::{AccountEntry48, CodeStore, StorageEntry48, ENTRY_SIZE};
use reth_db::{
    cursor::{DbCursorRO, DbDupCursorRO},
    database::Database,
    open_db_read_only, tables,
    transaction::DbTx,
};
use std::{
    fs::File,
    io::{BufWriter, Write},
    path::PathBuf,
};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the Reth database folder (e.g. /home/user/.local/share/reth/mainnet/db)
    #[arg(long, default_value = "/var/lib/reth/mainnet/db")]
    db_path: PathBuf,

    /// Output directory for artifacts
    #[arg(long, default_value = "data")]
    output_dir: PathBuf,

    /// Limit the number of accounts/slots to extract (useful for testing)
    #[arg(long)]
    limit: Option<usize>,

    /// Count items only, do not write artifacts
    #[arg(long, default_value_t = false)]
    count_only: bool,

    /// Batch size for transactions (items per transaction)
    #[arg(long, default_value_t = 100_000)]
    batch_size: usize,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let args = Args::parse();

    let now = || chrono::Local::now().format("%Y-%m-%d %H:%M:%S");

    println!("[{}] Opening database at {:?}", now(), args.db_path);
    println!("[{}] Using 48-byte schema (v2)", now());

    // We keep the DB open, but we will open/close TXs
    let db = open_db_read_only(&args.db_path, Default::default())?;

    // Code store for bytecode hash deduplication
    let mut code_store = CodeStore::new();

    let (mut db_writer, mut acc_map_writer, mut sto_map_writer) = if !args.count_only {
        // Create output directory
        std::fs::create_dir_all(&args.output_dir)?;
        let db_file_path = args.output_dir.join("database.bin");
        let acc_map_path = args.output_dir.join("account-mapping.bin");
        let sto_map_path = args.output_dir.join("storage-mapping.bin");

        println!("[{}] Writing outputs to:", now());
        println!("  Database: {:?}", db_file_path);
        println!("  Acc Map:  {:?}", acc_map_path);
        println!("  Sto Map:  {:?}", sto_map_path);

        (
            Some(BufWriter::new(File::create(&db_file_path)?)),
            Some(BufWriter::new(File::create(&acc_map_path)?)),
            Some(BufWriter::new(File::create(&sto_map_path)?)),
        )
    } else {
        println!(
            "[{}] Running in COUNT-ONLY mode. No files will be written.",
            now()
        );
        (None, None, None)
    };

    // Setup progress bar
    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.green} [{elapsed_precise}] {msg}")
            .unwrap(),
    );

    let limit = args.limit.unwrap_or(usize::MAX);
    let batch_size = args.batch_size;

    // --- READ CHAIN INFO ---
    println!("[{}] Reading Chain Info...", now());
    let tx = db.tx()?;
    let last_block = tx
        .cursor_read::<tables::CanonicalHeaders>()?
        .last()?
        .map(|(num, _hash)| num)
        .unwrap_or(0);
    println!("[{}] Database Tip: Block #{}", now(), last_block);
    drop(tx);

    // --- PROCESS ACCOUNTS ---
    println!("[{}] Processing Accounts (48-byte entries)...", now());
    let mut count_acc = 0;
    let mut total_entries = 0u64; // Now each account/storage is 1 entry (not 3)
    let mut last_acc_key = None;

    loop {
        if count_acc >= limit {
            break;
        }

        let tx = db.tx()?;
        let mut cursor = tx.cursor_read::<tables::PlainAccountState>()?;

        // Resume from last key
        let walker = if let Some(key) = last_acc_key {
            let mut w = cursor.walk(Some(key))?;
            // Skip the first element if it matches the last key we processed
            if let Some(Ok((addr, _))) = w.next() {
                if addr != key {
                    // If it's different, we shouldn't have skipped it.
                    drop(w);
                    let w2 = cursor.walk(Some(key))?;
                    w2
                } else {
                    w
                }
            } else {
                w
            }
        } else {
            cursor.walk(None)?
        };

        let mut batch_count = 0;
        let mut current_acc_key = None;

        for entry in walker {
            let (address, account) = entry?;

            // Double check skipping
            if let Some(last) = &last_acc_key {
                if &address == last {
                    continue;
                }
            }

            if count_acc >= limit {
                break;
            }

            // --- GET OR CREATE CODE ID ---
            let bytecode_hash = account.bytecode_hash.map(|h| {
                let mut arr = [0u8; 32];
                arr.copy_from_slice(h.as_slice());
                arr
            });
            let code_id = code_store.get_or_insert(bytecode_hash.as_ref());

            // --- WRITE DATABASE ENTRY (48 bytes) ---
            if let Some(writer) = db_writer.as_mut() {
                // Convert address to [u8; 20]
                let addr_bytes: [u8; 20] = address.0 .0;

                // Convert balance to [u8; 32] (little-endian)
                let balance_bytes = account.balance.to_le_bytes::<32>();

                // Create 48-byte account entry
                let entry =
                    AccountEntry48::new(&balance_bytes, account.nonce, code_id, &addr_bytes);
                writer.write_all(&entry.to_bytes())?;
            }

            // --- WRITE MAPPING ---
            if let Some(writer) = acc_map_writer.as_mut() {
                // Address (20) + Index (4)
                writer.write_all(address.as_slice())?;

                let entry_index = u32::try_from(total_entries)
                    .map_err(|_| eyre::eyre!("total_entries {} exceeds u32::MAX", total_entries))?;
                writer.write_all(&entry_index.to_le_bytes())?;
            }

            total_entries += 1; // 1 entry per account now (was 3)
            count_acc += 1;
            batch_count += 1;
            current_acc_key = Some(address);

            if count_acc % 10000 == 0 {
                pb.set_message(format!(
                    "Acc: {}, Sto: 0, CodeIDs: {}",
                    count_acc,
                    code_store.len()
                ));
                pb.inc(1);
            }

            if batch_count >= batch_size {
                break;
            }
        }

        // End of batch
        if batch_count == 0 {
            break;
        }

        last_acc_key = current_acc_key;
        drop(tx); // Close transaction
    }

    let account_entries = total_entries;
    println!(
        "[{}] Processed {} accounts ({} entries). Unique bytecode hashes: {}",
        now(),
        count_acc,
        account_entries,
        code_store.len()
    );

    // --- PROCESS STORAGE ---
    println!("[{}] Processing Storage (48-byte entries)...", now());
    let mut count_sto = 0;
    let mut last_sto_addr = None;

    loop {
        if count_sto >= limit {
            break;
        }

        let tx = db.tx()?;
        let mut cursor = tx.cursor_dup_read::<tables::PlainStorageState>()?;

        // Resume Logic for Storage
        let walker = if let Some(addr) = last_sto_addr {
            // Try to seek to the address
            if let Some((_k, _)) = cursor.seek_exact(addr)? {
                // If found, jump to next unique key
                if let Some((next_k, _)) = cursor.next_no_dup()? {
                    cursor.walk(Some(next_k))?
                } else {
                    // End of database
                    break;
                }
            } else {
                // If `addr` not found, just walk from `addr` and filtering will handle it
                cursor.walk(Some(addr))?
            }
        } else {
            cursor.walk(None)?
        };

        let mut batch_count = 0;
        let mut current_addr = None;

        for entry in walker {
            let (address, storage_entry) = entry?;

            // Robust skipping: if we are still seeing the old address, skip it
            if let Some(last) = &last_sto_addr {
                if &address == last {
                    continue;
                }
            }

            // Check if we crossed a boundary and should stop for this batch
            if let Some(curr) = current_addr {
                if address != curr && batch_count >= batch_size {
                    break;
                }
            }

            current_addr = Some(address);

            if count_sto >= limit {
                break;
            }

            // --- WRITE DATABASE ENTRY (48 bytes) ---
            if let Some(writer) = db_writer.as_mut() {
                // Convert types
                let addr_bytes: [u8; 20] = address.0 .0;
                let value_bytes = storage_entry.value.to_le_bytes::<32>();
                let mut slot_key = [0u8; 32];
                slot_key.copy_from_slice(storage_entry.key.as_slice());

                // Create 48-byte storage entry
                let entry = StorageEntry48::new(&value_bytes, &addr_bytes, &slot_key);
                writer.write_all(&entry.to_bytes())?;
            }

            // --- WRITE MAPPING ---
            if let Some(writer) = sto_map_writer.as_mut() {
                writer.write_all(address.as_slice())?;
                writer.write_all(storage_entry.key.as_slice())?;

                let entry_index = u32::try_from(total_entries)
                    .map_err(|_| eyre::eyre!("total_entries {} exceeds u32::MAX", total_entries))?;
                writer.write_all(&entry_index.to_le_bytes())?;
            }

            total_entries += 1;
            count_sto += 1;
            batch_count += 1;

            if count_sto % 10000 == 0 {
                pb.set_message(format!("Acc: {}, Sto: {}", count_acc, count_sto));
                pb.inc(1);
            }
        }

        if batch_count == 0 {
            break;
        }

        last_sto_addr = current_addr;
        drop(tx);
    }

    pb.finish_and_clear();
    println!(
        "[{}] Done! Acc: {}, Sto: {}, Total Entries: {}",
        now(),
        count_acc,
        count_sto,
        total_entries
    );

    // Calculate sizes
    let db_size_bytes = total_entries * ENTRY_SIZE as u64;
    let db_size_gb = db_size_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
    println!(
        "[{}] Database size: {} entries × {} bytes = {:.2} GB",
        now(),
        total_entries,
        ENTRY_SIZE,
        db_size_gb
    );

    if let Some(mut writer) = db_writer {
        writer.flush()?;
    }
    if let Some(mut writer) = acc_map_writer {
        writer.flush()?;
    }
    if let Some(mut writer) = sto_map_writer {
        writer.flush()?;
    }

    // --- WRITE CODE STORE ---
    if !args.count_only && !code_store.is_empty() {
        let code_store_path = args.output_dir.join("code_store.bin");
        println!("[{}] Writing code store: {:?}", now(), code_store_path);
        std::fs::write(&code_store_path, code_store.to_bytes())?;
        println!(
            "[{}] Code store: {} unique bytecode hashes ({} bytes)",
            now(),
            code_store.len(),
            4 + code_store.len() * 32
        );
    }

    // --- WRITE METADATA ---
    if !args.count_only {
        let meta_path = args.output_dir.join("metadata.json");
        let json = format!(
            r#"{{
  "schema_version": 2,
  "entry_size_bytes": {},
  "block": {},
  "accounts": {},
  "storage_slots": {},
  "total_entries": {},
  "unique_bytecode_hashes": {},
  "generated_at": "{}"
}}"#,
            ENTRY_SIZE,
            last_block,
            count_acc,
            count_sto,
            total_entries,
            code_store.len(),
            now()
        );
        std::fs::write(meta_path, json)?;
    }

    println!("[{}] Extraction complete.", now());

    Ok(())
}
