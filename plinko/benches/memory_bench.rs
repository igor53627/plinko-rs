//! Memory benchmark: Vec<Vec<usize>> vs Vec<BlockBitset> for block set storage.
//!
//! Measures allocation time and reports size comparison for typical c values.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::seq::index::sample;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use std::sync::Once;

/// Minimal BlockBitset replica (same layout as hint_gen::bitset::BlockBitset).
struct BlockBitset {
    bits: Vec<u64>,
}

impl BlockBitset {
    fn from_sorted_blocks(blocks: &[usize], num_blocks: usize) -> Self {
        let num_words = num_blocks.div_ceil(64);
        let mut bits = vec![0u64; num_words];
        for &block in blocks {
            if block < num_blocks {
                bits[block / 64] |= 1u64 << (block % 64);
            }
        }
        Self { bits }
    }

    fn heap_size(&self) -> usize {
        self.bits.len() * std::mem::size_of::<u64>()
    }
}

fn generate_subsets(c: usize, count: usize) -> Vec<Vec<usize>> {
    let mut rng = ChaCha20Rng::from_seed([42u8; 32]);
    let subset_size = c / 2 + 1;
    (0..count)
        .map(|_| {
            let mut blocks = sample(&mut rng, c, subset_size).into_vec();
            blocks.sort_unstable();
            blocks
        })
        .collect()
}

fn bench_vec_allocation(c: &mut Criterion) {
    let mut group = c.benchmark_group("block_set_allocation");

    for c_val in [100, 1000, 10000] {
        let count = 128; // typical lambda * w hints
        let subsets = generate_subsets(c_val, count);

        group.bench_with_input(
            BenchmarkId::new("Vec<Vec<usize>>", c_val),
            &c_val,
            |b, _| {
                b.iter(|| {
                    let vecs: Vec<Vec<usize>> = subsets.clone();
                    std::hint::black_box(vecs);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("Vec<BlockBitset>", c_val),
            &c_val,
            |b, _| {
                b.iter(|| {
                    let bitsets: Vec<BlockBitset> = subsets
                        .iter()
                        .map(|blocks| BlockBitset::from_sorted_blocks(blocks, c_val))
                        .collect();
                    std::hint::black_box(bitsets);
                });
            },
        );
    }

    group.finish();
}

static PRINT_SIZES: Once = Once::new();

fn bench_size_comparison(c: &mut Criterion) {
    // Print size comparison exactly once, regardless of Criterion warmup cycles.
    PRINT_SIZES.call_once(|| {
        eprintln!("\n=== Block Set Memory Comparison ===");
        for c_val in [100, 1000, 10000] {
            let count = 128;
            let subsets = generate_subsets(c_val, count);

            let vec_heap: usize = subsets
                .iter()
                .map(|v| v.len() * std::mem::size_of::<usize>())
                .sum();
            let vec_overhead = count * std::mem::size_of::<Vec<usize>>();

            let bitsets: Vec<BlockBitset> = subsets
                .iter()
                .map(|blocks| BlockBitset::from_sorted_blocks(blocks, c_val))
                .collect();
            let bitset_heap: usize = bitsets.iter().map(|bs| bs.heap_size()).sum();
            let bitset_overhead = count * std::mem::size_of::<Vec<u64>>();

            let vec_total = vec_heap + vec_overhead;
            let bitset_total = bitset_heap + bitset_overhead;
            let ratio = vec_total as f64 / bitset_total as f64;

            eprintln!(
                "c={c_val:>5}, hints={count}: Vec={vec_total:>10} B, Bitset={bitset_total:>10} B, ratio={ratio:.1}x"
            );
        }
    });

    // Benchmark the conversion from pre-generated subsets to bitsets (setup excluded).
    let all_subsets: Vec<(usize, Vec<Vec<usize>>)> = [100, 1000, 10000]
        .iter()
        .map(|&c_val| (c_val, generate_subsets(c_val, 128)))
        .collect();

    c.bench_function("size_report", |b| {
        b.iter(|| {
            for (c_val, subsets) in &all_subsets {
                let vec_heap: usize = subsets
                    .iter()
                    .map(|v| v.len() * std::mem::size_of::<usize>())
                    .sum();
                let vec_overhead = 128 * std::mem::size_of::<Vec<usize>>();

                let bitsets: Vec<BlockBitset> = subsets
                    .iter()
                    .map(|blocks| BlockBitset::from_sorted_blocks(blocks, *c_val))
                    .collect();
                let bitset_heap: usize = bitsets.iter().map(|bs| bs.heap_size()).sum();
                let bitset_overhead = 128 * std::mem::size_of::<Vec<u64>>();

                let vec_total = vec_heap + vec_overhead;
                let bitset_total = bitset_heap + bitset_overhead;

                std::hint::black_box((vec_total, bitset_total));
            }
        });
    });
}

criterion_group!(benches, bench_vec_allocation, bench_size_comparison);
criterion_main!(benches);
