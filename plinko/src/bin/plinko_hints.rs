//! Plinko PIR Hint Generator - see docs/hint_generation.md for details.

mod hint_gen;

use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use memmap2::MmapOptions;
use plinko::hints::{seed_check, HintGeometry, HintSet};
use plinko::iprf::{Iprf, IprfTee};
use plinko::subsets::SubsetPrf;
use std::fs::File;
use std::time::Instant;
use tracing::{debug, info};

use hint_gen::*;

fn main() -> eyre::Result<()> {
    tracing_subscriber::fmt::init();
    let args = Args::parse();
    validate_args(&args)?;

    info!(db_path = ?args.db_path, "Plinko PIR Hint Generator (paper-compliant)");

    let file = File::open(&args.db_path)?;
    let file_len = file.metadata()?.len() as usize;
    info!(
        size_gb = format_args!("{:.2}", file_len as f64 / 1024.0 / 1024.0 / 1024.0),
        "Database loaded"
    );

    let mmap = unsafe { MmapOptions::new().map(&file)? };
    let db_bytes: &[u8] = &mmap;

    let geom = compute_geometry(db_bytes.len(), &args)?;
    info!(
        n_entries = geom.n_entries,
        n_effective = geom.n_effective,
        pad_entries = geom.pad_entries,
        "Total entries"
    );

    debug!(
        w = geom.w,
        c = geom.c,
        lambda = args.lambda,
        "Plinko parameters"
    );

    let params = HintParams::from_args(&args, geom.w);
    if args.constant_time {
        validate_hint_params(&params, geom.w)?;
    }
    debug!(
        regular = params.num_regular,
        backup = params.num_backup,
        "Hint structure"
    );

    let master_seed = parse_or_generate_seed(&args)?;
    let start = Instant::now();

    info!(step = "1/4", count = geom.c, "Generating iPRF keys");
    let block_keys = derive_block_keys(&master_seed, geom.c);

    info!(
        step = "2/4",
        count = params.num_regular,
        "Initializing regular hints"
    );
    info!(
        step = "3/4",
        count = params.num_backup,
        "Initializing backup hints"
    );
    let subset_prf: SubsetPrf = SubsetPrf::new(&master_seed);
    let (mut regular_hints, mut backup_hints) = init_hints(&subset_prf, geom.c, &params);

    info!(
        step = "4/4",
        n_effective = geom.n_effective,
        constant_time = args.constant_time,
        "Streaming database"
    );

    let pb = ProgressBar::new(geom.n_effective as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template(
                "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})",
            )
            .unwrap()
            .progress_chars("#>-"),
    );

    if args.constant_time {
        let block_iprfs_ct: Vec<IprfTee> = block_keys
            .iter()
            .map(|key| IprfTee::new(*key, params.total_hints as u64, geom.w as u64))
            .collect();

        hint_gen::ct_path::process_entries_ct(
            hint_gen::ct_path::CtProcessInput {
                db_bytes,
                n_entries: geom.n_entries,
                n_effective: geom.n_effective,
                w: geom.w,
                num_regular: params.num_regular,
                num_backup: params.num_backup,
                block_iprfs_ct: &block_iprfs_ct,
                subset_prf: &subset_prf,
            },
            &mut regular_hints,
            &mut backup_hints,
            |i| pb.set_position(i as u64),
        );
    } else {
        let block_iprfs: Vec<Iprf> = block_keys
            .iter()
            .map(|key| Iprf::new(*key, params.total_hints as u64, geom.w as u64))
            .collect();

        hint_gen::fast_path::process_entries_fast(
            hint_gen::fast_path::FastProcessInput {
                db_bytes,
                n_entries: geom.n_entries,
                n_effective: geom.n_effective,
                w: geom.w,
                num_regular: params.num_regular,
                num_backup: params.num_backup,
                block_iprfs: &block_iprfs,
                subset_prf: &subset_prf,
            },
            &mut regular_hints,
            &mut backup_hints,
            |i| pb.set_position(i as u64),
        );
    }

    pb.finish_with_message("Done");

    let duration = start.elapsed();
    print_results(duration, file_len, &regular_hints, &backup_hints, &params);

    if let Some(output) = &args.output {
        let hints = HintSet {
            geometry: HintGeometry {
                num_entries: geom.n_entries as u64,
                block_size: geom.w as u64,
                num_blocks: geom.c as u64,
            },
            seed_check: seed_check(&master_seed),
            regular: regular_hints,
            backup: backup_hints,
        };
        write_hint_file(output, &hints)?;
        info!(path = ?output, bytes = hints.serialized_len(), "Hint file written");
    }

    Ok(())
}
