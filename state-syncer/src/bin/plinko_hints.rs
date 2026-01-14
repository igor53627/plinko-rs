//! Plinko PIR Hint Generator - see docs/hint_generation.md for details.

mod hint_gen;

use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use memmap2::MmapOptions;
use state_syncer::iprf::{Iprf, IprfTee};
use std::cell::Cell;
use std::fs::File;
use std::time::Instant;

use hint_gen::*;

fn format_eta(seconds: f64) -> String {
    if !seconds.is_finite() || seconds < 0.0 {
        return "unknown".to_string();
    }
    let secs = seconds.round() as u64;
    let hours = secs / 3600;
    let mins = (secs % 3600) / 60;
    let secs = secs % 60;
    if hours > 0 {
        format!("{hours}h{mins:02}m{secs:02}s")
    } else if mins > 0 {
        format!("{mins}m{secs:02}s")
    } else {
        format!("{secs}s")
    }
}

fn main() -> eyre::Result<()> {
    let args = Args::parse();
    validate_args(&args)?;

    println!("Plinko PIR Hint Generator (Paper-compliant)");
    println!("============================================");
    println!("Database: {:?}", args.db_path);

    let file = File::open(&args.db_path)?;
    let file_len = file.metadata()?.len() as usize;
    println!(
        "DB Size: {:.2} GB",
        file_len as f64 / 1024.0 / 1024.0 / 1024.0
    );

    let mmap = unsafe { MmapOptions::new().map(&file)? };
    let db_bytes: &[u8] = &mmap;

    let geom = compute_geometry(db_bytes.len(), &args)?;
    println!("Total Entries (N): {}", geom.n_entries);

    println!("\nPlinko Parameters:");
    println!("  Entries per block (w): {}", geom.w);
    println!("  Number of blocks (c): {}", geom.c);
    println!("  Lambda: {}", args.lambda);

    let params = HintParams::from_args(&args, geom.w);
    if args.constant_time {
        validate_hint_params(&params, geom.w)?;
    }
    println!("\nHint Structure:");
    println!("  Regular hints: {}", params.num_regular);
    println!("  Backup hints: {}", params.num_backup);

    let master_seed = parse_or_generate_seed(&args)?;
    let start = Instant::now();

    println!("\n[1/4] Generating {} iPRF keys...", geom.c);
    let block_keys = derive_block_keys(&master_seed, geom.c);

    println!("[2/4] Initializing {} regular hints...", params.num_regular);
    println!("[3/4] Initializing {} backup hints...", params.num_backup);
    let (mut regular_hints, regular_hint_blocks, mut backup_hints, backup_hint_blocks) =
        init_hints(&master_seed, geom.c, &params);

    println!("[4/4] Streaming database ({} entries)...", geom.n_effective);
    if args.constant_time {
        println!("  [CT MODE] Using constant-time implementation for TEE");
    }

    let pb = ProgressBar::new(geom.n_effective as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template(
                "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})",
            )
            .unwrap()
            .progress_chars("#>-"),
    );
    let total_entries = geom.n_effective;
    let log_step = (total_entries / 100).max(1);
    let last_log = Cell::new(0usize);
    let progress = |i: usize| {
        pb.set_position(i as u64);
        if total_entries == 0 {
            return;
        }
        let processed = i.min(total_entries);
        let should_log = processed >= last_log.get().saturating_add(log_step)
            || processed == total_entries;
        if !should_log {
            return;
        }
        last_log.set(processed);
        let elapsed = start.elapsed().as_secs_f64();
        if elapsed <= 0.0 {
            return;
        }
        let pct = (processed as f64 / total_entries as f64) * 100.0;
        let entries_per_sec = processed as f64 / elapsed;
        let mib_per_sec =
            (processed as f64 * WORD_SIZE as f64) / (1024.0 * 1024.0) / elapsed;
        let remaining = total_entries.saturating_sub(processed);
        let eta = if entries_per_sec > 0.0 {
            remaining as f64 / entries_per_sec
        } else {
            0.0
        };
        println!(
            "  progress: {pct:.1}% ({processed}/{total_entries}) | {entries_per_sec:.0} entries/s | {mib_per_sec:.1} MiB/s | eta {}",
            format_eta(eta)
        );
    };

    if args.constant_time {
        let block_iprfs_ct: Vec<IprfTee> = block_keys
            .iter()
            .map(|key| IprfTee::new(*key, params.total_hints as u64, geom.w as u64))
            .collect();

        let regular_bitsets: Vec<BlockBitset> = regular_hint_blocks
            .iter()
            .map(|blocks| BlockBitset::from_sorted_blocks(blocks, geom.c))
            .collect();
        let backup_bitsets: Vec<BlockBitset> = backup_hint_blocks
            .iter()
            .map(|blocks| BlockBitset::from_sorted_blocks(blocks, geom.c))
            .collect();

        hint_gen::ct_path::process_entries_ct(
            db_bytes,
            geom.n_entries,
            geom.n_effective,
            geom.w,
            params.num_regular,
            params.num_backup,
            &block_iprfs_ct,
            &regular_bitsets,
            &backup_bitsets,
            &mut regular_hints,
            &mut backup_hints,
            progress,
        );
    } else {
        let block_iprfs: Vec<Iprf> = block_keys
            .iter()
            .map(|key| Iprf::new(*key, params.total_hints as u64, geom.w as u64))
            .collect();

        hint_gen::fast_path::process_entries_fast(
            db_bytes,
            geom.n_entries,
            geom.n_effective,
            geom.w,
            geom.c,
            params.num_regular,
            params.num_backup,
            &block_iprfs,
            &regular_hint_blocks,
            &backup_hint_blocks,
            &mut regular_hints,
            &mut backup_hints,
            progress,
        );
    }

    pb.set_position(total_entries as u64);
    pb.finish_with_message("Done");

    let duration = start.elapsed();
    print_results(
        duration,
        file_len,
        &regular_hints,
        &backup_hints,
        &params,
        &block_keys,
        geom.w,
        geom.c,
    );

    Ok(())
}
