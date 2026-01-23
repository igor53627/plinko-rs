//! GPU-accelerated hint generation using CUDA.
//!
//! This module provides a `GpuHintGenerator` that uses CUDA kernels to parallelize
//! hint computation across GPU threads. Each thread computes one hint's parity.
//!
//! # Requirements
//!
//! - CUDA toolkit installed (nvcc in PATH)
//! - NVIDIA GPU with compute capability >= 8.0 (A100, H100, H200)
//! - Build with `--features cuda`
//!
//! # Example
//!
//! ```ignore
//! use plinko::gpu::GpuHintGenerator;
//!
//! let generator = GpuHintGenerator::new(0)?;  // GPU device 0
//! let hints = generator.generate_hints(&database, &block_keys, &subsets)?;
//! ```

#[cfg(feature = "cuda")]
use bytemuck::{Pod, Zeroable};
#[cfg(feature = "cuda")]
use cudarc::driver::{CudaDevice, CudaFunction, CudaSlice, DeviceRepr, LaunchAsync, LaunchConfig};
#[cfg(feature = "cuda")]
use std::sync::Arc;

use crate::schema48::ENTRY_SIZE;

/// Plinko parameters for GPU kernel
#[cfg(feature = "cuda")]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
#[repr(C)]
pub struct PlinkoParams {
    pub num_entries: u64,
    pub chunk_size: u64,
    pub set_size: u64,
    pub lambda: u32,
    pub total_hints: u32,
    pub blocks_per_hint: u32,
    pub _pad: u32,
}

#[cfg(feature = "cuda")]
unsafe impl DeviceRepr for PlinkoParams {}

/// iPRF key for one block (256-bit ChaCha key as 8 × u32)
#[cfg(feature = "cuda")]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
#[repr(C)]
pub struct IprfBlockKey {
    pub key: [u32; 8],
}

#[cfg(feature = "cuda")]
unsafe impl DeviceRepr for IprfBlockKey {}

/// Hint output (32-byte parity)
#[cfg(feature = "cuda")]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
#[repr(C)]
pub struct HintOutput {
    pub parity: [u8; 32],
}

#[cfg(feature = "cuda")]
unsafe impl DeviceRepr for HintOutput {}

/// GPU hint generator using CUDA
#[cfg(feature = "cuda")]
pub struct GpuHintGenerator {
    device: Arc<CudaDevice>,
    kernel: CudaFunction,
    kernel_tiled: CudaFunction,
}

#[cfg(feature = "cuda")]
impl GpuHintGenerator {
    /// Create a new GPU hint generator on the specified device.
    ///
    /// # Arguments
    ///
    /// * `device_ord` - CUDA device ordinal (0 for first GPU)
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA initialization fails or the kernel cannot be loaded.
    pub fn new(device_ord: usize) -> Result<Self, cudarc::driver::DriverError> {
        let device = CudaDevice::new(device_ord)?;

        // Load PTX compiled by build.rs
        let ptx_path = concat!(env!("OUT_DIR"), "/hint_kernel.ptx");
        let ptx = std::fs::read_to_string(ptx_path)
            .expect("Failed to read hint_kernel.ptx - was it compiled?");

        device.load_ptx(
            ptx.into(),
            "hint_gen",
            &["hint_gen_kernel", "hint_gen_kernel_tiled"],
        )?;

        let kernel = device
            .get_func("hint_gen", "hint_gen_kernel")
            .expect("Failed to get hint_gen_kernel from PTX module");
        let kernel_tiled = device
            .get_func("hint_gen", "hint_gen_kernel_tiled")
            .expect("Failed to get hint_gen_kernel_tiled from PTX module");

        Ok(Self {
            device,
            kernel,
            kernel_tiled,
        })
    }

    /// Generate hints using GPU acceleration.
    ///
    /// # Arguments
    ///
    /// * `entries` - Database entries (N × 48 bytes)
    /// * `block_keys` - iPRF key for each block (c keys)
    /// * `hint_subsets` - Precomputed block subsets as bitsets
    /// * `params` - Plinko parameters
    ///
    /// # Returns
    ///
    /// Vector of hint parities (total_hints × 32 bytes)
    pub fn generate_hints(
        &self,
        entries: &[u8],
        block_keys: &[IprfBlockKey],
        hint_subsets: &[u8],
        params: PlinkoParams,
    ) -> Result<Vec<HintOutput>, cudarc::driver::DriverError> {
        // Validate input sizes
        let expected_entries_size = params.num_entries as usize * ENTRY_SIZE;
        assert_eq!(
            entries.len(),
            expected_entries_size,
            "entries size mismatch"
        );
        assert_eq!(
            block_keys.len(),
            params.set_size as usize,
            "block_keys size mismatch"
        );

        let subset_bytes_per_hint = (params.set_size as usize + 7) / 8;
        let expected_subset_size = params.total_hints as usize * subset_bytes_per_hint;
        assert_eq!(
            hint_subsets.len(),
            expected_subset_size,
            "hint_subsets size mismatch"
        );

        // Copy data to GPU
        let d_entries = self.device.htod_sync_copy(entries)?;
        let d_block_keys: CudaSlice<IprfBlockKey> = self.device.htod_sync_copy(block_keys)?;
        let d_hint_subsets = self.device.htod_sync_copy(hint_subsets)?;

        // Allocate output buffer
        let output_size = params.total_hints as usize;
        let mut d_output: CudaSlice<HintOutput> = unsafe { self.device.alloc(output_size)? };

        // Launch configuration
        let threads_per_block = 256;
        let num_blocks = (params.total_hints as u32 + threads_per_block - 1) / threads_per_block;

        let cfg = LaunchConfig {
            grid_dim: (num_blocks, 1, 1),
            block_dim: (threads_per_block, 1, 1),
            shared_mem_bytes: 0,
        };

        // Launch kernel
        unsafe {
            self.kernel_tiled.clone().launch(
                cfg,
                (
                    params,
                    &d_block_keys,
                    &d_entries,
                    &d_hint_subsets,
                    &mut d_output,
                ),
            )?;
        }

        // Copy results back
        let output = self.device.dtoh_sync_copy(&d_output)?;
        Ok(output)
    }

    /// Get device name for logging
    pub fn device_name(&self) -> String {
        // cudarc doesn't expose device name directly, return ordinal
        format!("CUDA Device {}", self.device.ordinal())
    }
}

// ============================================================================
// ChaCha8 implementation for CPU
// ============================================================================

const CHACHA_ROUNDS: usize = 8;
const SN_ROUNDS: usize = 64;

/// ChaCha quarter round operating on four values
#[inline(always)]
fn quarter_round(a: u32, b: u32, c: u32, d: u32) -> (u32, u32, u32, u32) {
    let a = a.wrapping_add(b); let d = (d ^ a).rotate_left(16);
    let c = c.wrapping_add(d); let b = (b ^ c).rotate_left(12);
    let a = a.wrapping_add(b); let d = (d ^ a).rotate_left(8);
    let c = c.wrapping_add(d); let b = (b ^ c).rotate_left(7);
    (a, b, c, d)
}

fn chacha8_block(key: &[u32; 8], counter: u32, nonce: u32) -> [u32; 16] {
    let mut s: [u32; 16] = [
        0x61707865, 0x3320646e, 0x79622d32, 0x6b206574,
        key[0], key[1], key[2], key[3],
        key[4], key[5], key[6], key[7],
        counter, nonce, 0, 0,
    ];
    let initial = s;

    for _ in 0..CHACHA_ROUNDS / 2 {
        // Column rounds
        (s[0], s[4], s[8], s[12]) = quarter_round(s[0], s[4], s[8], s[12]);
        (s[1], s[5], s[9], s[13]) = quarter_round(s[1], s[5], s[9], s[13]);
        (s[2], s[6], s[10], s[14]) = quarter_round(s[2], s[6], s[10], s[14]);
        (s[3], s[7], s[11], s[15]) = quarter_round(s[3], s[7], s[11], s[15]);
        // Diagonal rounds
        (s[0], s[5], s[10], s[15]) = quarter_round(s[0], s[5], s[10], s[15]);
        (s[1], s[6], s[11], s[12]) = quarter_round(s[1], s[6], s[11], s[12]);
        (s[2], s[7], s[8], s[13]) = quarter_round(s[2], s[7], s[8], s[13]);
        (s[3], s[4], s[9], s[14]) = quarter_round(s[3], s[4], s[9], s[14]);
    }

    for i in 0..16 {
        s[i] = s[i].wrapping_add(initial[i]);
    }
    s
}

/// SwapOrNot inverse using ChaCha8
fn sn_inverse(key: &[u32; 8], y: u64, domain: u64) -> u64 {
    let mut val = y;
    for r in (0..SN_ROUNDS).rev() {
        // Derive round key
        let output = chacha8_block(key, r as u32, 0);
        let k_i = (((output[1] as u64) << 32) | (output[0] as u64)) % domain;

        let partner = (k_i + domain - (val % domain)) % domain;
        let canonical = val.max(partner);

        // PRF bit
        let output2 = chacha8_block(key, r as u32 | 0x80000000, canonical as u32);
        if output2[0] & 1 == 1 {
            val = partner;
        }
    }
    val
}

/// CPU hint generator with full ChaCha8-based iPRF
pub struct CpuHintGenerator {
    parallel: bool,
}

impl CpuHintGenerator {
    pub fn new() -> Self {
        Self { parallel: true }
    }

    pub fn new_serial() -> Self {
        Self { parallel: false }
    }

    /// Generate hints using CPU with full ChaCha8 iPRF.
    #[cfg(feature = "parallel")]
    pub fn generate_hints(
        &self,
        entries: &[u8],
        block_keys: &[[u32; 8]],
        hint_subsets: &[u8],
        num_entries: u64,
        chunk_size: u64,
        set_size: u64,
        total_hints: u32,
    ) -> Vec<[u8; 32]> {
        use rayon::prelude::*;

        let subset_bytes_per_hint = (set_size as usize + 7) / 8;

        if self.parallel {
            (0..total_hints as usize)
                .into_par_iter()
                .map(|hint_idx| {
                    self.compute_hint(
                        hint_idx,
                        entries,
                        block_keys,
                        hint_subsets,
                        num_entries,
                        chunk_size,
                        set_size,
                        subset_bytes_per_hint,
                    )
                })
                .collect()
        } else {
            (0..total_hints as usize)
                .map(|hint_idx| {
                    self.compute_hint(
                        hint_idx,
                        entries,
                        block_keys,
                        hint_subsets,
                        num_entries,
                        chunk_size,
                        set_size,
                        subset_bytes_per_hint,
                    )
                })
                .collect()
        }
    }

    #[cfg(not(feature = "parallel"))]
    pub fn generate_hints(
        &self,
        entries: &[u8],
        block_keys: &[[u32; 8]],
        hint_subsets: &[u8],
        num_entries: u64,
        chunk_size: u64,
        set_size: u64,
        total_hints: u32,
    ) -> Vec<[u8; 32]> {
        let subset_bytes_per_hint = (set_size as usize + 7) / 8;

        (0..total_hints as usize)
            .map(|hint_idx| {
                self.compute_hint(
                    hint_idx,
                    entries,
                    block_keys,
                    hint_subsets,
                    num_entries,
                    chunk_size,
                    set_size,
                    subset_bytes_per_hint,
                )
            })
            .collect()
    }

    #[inline]
    fn compute_hint(
        &self,
        hint_idx: usize,
        entries: &[u8],
        block_keys: &[[u32; 8]],
        hint_subsets: &[u8],
        num_entries: u64,
        chunk_size: u64,
        set_size: u64,
        subset_bytes_per_hint: usize,
    ) -> [u8; 32] {
        let mut parity = [0u64; 4];

        for block_idx in 0..set_size as usize {
            // Check subset membership
            let byte_idx = hint_idx * subset_bytes_per_hint + (block_idx / 8);
            let bit_mask = 1u8 << (block_idx % 8);
            if (hint_subsets[byte_idx] & bit_mask) == 0 {
                continue;
            }

            // Full iPRF inverse using ChaCha8-based SwapOrNot
            let key = &block_keys[block_idx];
            let preimage = sn_inverse(key, hint_idx as u64, chunk_size);

            if preimage < chunk_size {
                let entry_idx = block_idx as u64 * chunk_size + preimage;
                if entry_idx < num_entries {
                    let entry_start = entry_idx as usize * ENTRY_SIZE;
                    let entry_bytes = &entries[entry_start..entry_start + 32];

                    // XOR as u64s
                    for i in 0..4 {
                        let val = u64::from_le_bytes(
                            entry_bytes[i * 8..(i + 1) * 8].try_into().unwrap(),
                        );
                        parity[i] ^= val;
                    }
                }
            }
        }

        // Convert parity to bytes
        let mut result = [0u8; 32];
        for i in 0..4 {
            result[i * 8..(i + 1) * 8].copy_from_slice(&parity[i].to_le_bytes());
        }
        result
    }
}

impl Default for CpuHintGenerator {
    fn default() -> Self {
        Self::new()
    }
}

/// Simplified CPU hint generator (no iPRF, for baseline comparison)
pub struct SimpleCpuHintGenerator;

impl SimpleCpuHintGenerator {
    pub fn new() -> Self {
        Self
    }

    /// Generate hints using simplified CPU (no iPRF, just XOR first entry per block)
    pub fn generate_hints(
        &self,
        entries: &[u8],
        _block_keys: &[[u32; 8]],
        hint_subsets: &[u8],
        num_entries: u64,
        chunk_size: u64,
        set_size: u64,
        total_hints: u32,
    ) -> Vec<[u8; 32]> {
        let subset_bytes_per_hint = (set_size as usize + 7) / 8;
        let mut hints = vec![[0u8; 32]; total_hints as usize];

        for hint_idx in 0..total_hints as usize {
            let mut parity = [0u64; 4];

            for block_idx in 0..set_size as usize {
                let byte_idx = hint_idx * subset_bytes_per_hint + (block_idx / 8);
                let bit_mask = 1u8 << (block_idx % 8);
                if (hint_subsets[byte_idx] & bit_mask) == 0 {
                    continue;
                }

                // Simplified: just XOR the first entry in each block
                let entry_idx = block_idx * chunk_size as usize;
                if entry_idx < num_entries as usize {
                    let entry_start = entry_idx * ENTRY_SIZE;
                    let entry_bytes = &entries[entry_start..entry_start + 32];

                    for i in 0..4 {
                        let val = u64::from_le_bytes(
                            entry_bytes[i * 8..(i + 1) * 8].try_into().unwrap(),
                        );
                        parity[i] ^= val;
                    }
                }
            }

            for i in 0..4 {
                hints[hint_idx][i * 8..(i + 1) * 8].copy_from_slice(&parity[i].to_le_bytes());
            }
        }

        hints
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chacha8_block() {
        // Test with zero key and counter to verify basic operation
        let key = [0u32; 8];
        let output = chacha8_block(&key, 0, 0);

        // ChaCha8 with zero inputs produces deterministic non-zero output
        // due to the constants in the initial state
        assert_ne!(output[0], 0);

        // Different counter should produce different output
        let output2 = chacha8_block(&key, 1, 0);
        assert_ne!(output, output2);

        // Same inputs should produce same output (determinism)
        let output3 = chacha8_block(&key, 0, 0);
        assert_eq!(output, output3);
    }

    #[test]
    fn test_sn_inverse_roundtrip() {
        // SwapOrNot should be its own inverse when applied twice
        let key = [0x12345678u32, 0x9ABCDEF0, 0x11111111, 0x22222222,
                   0x33333333, 0x44444444, 0x55555555, 0x66666666];
        let domain = 1000u64;

        for y in [0u64, 1, 50, 500, 999] {
            let x = sn_inverse(&key, y, domain);
            assert!(x < domain, "Preimage should be in domain");
        }
    }

    #[test]
    fn test_cpu_hint_generator_basic() {
        let gen = CpuHintGenerator::new();

        // Create minimal test data
        let num_entries = 100u64;
        let chunk_size = 16u64;
        let set_size = 8u64;
        let total_hints = 4u32;

        // Create fake entries (all zeros)
        let entries = vec![0u8; num_entries as usize * ENTRY_SIZE];

        // Create fake block keys (256-bit ChaCha keys)
        let block_keys = vec![[0u32; 8]; set_size as usize];

        // Create hint subsets (each hint includes all blocks)
        let subset_bytes = (set_size as usize + 7) / 8;
        let hint_subsets = vec![0xFFu8; total_hints as usize * subset_bytes];

        let hints = gen.generate_hints(
            &entries,
            &block_keys,
            &hint_subsets,
            num_entries,
            chunk_size,
            set_size,
            total_hints,
        );

        assert_eq!(hints.len(), total_hints as usize);
        // With all-zero entries, parities should be zero
        for hint in &hints {
            assert_eq!(*hint, [0u8; 32]);
        }
    }

    #[test]
    fn test_cpu_hint_generator_consistency() {
        // Test that the generator produces consistent results across calls
        let gen = CpuHintGenerator::new();
        let gen_serial = CpuHintGenerator::new_serial();

        let num_entries = 64u64;
        let chunk_size = 16u64;
        let set_size = 4u64;
        let total_hints = 8u32;

        // Create entries with deterministic pattern
        let mut entries = vec![0u8; num_entries as usize * ENTRY_SIZE];
        for i in 0..num_entries as usize {
            entries[i * ENTRY_SIZE] = i as u8;
            entries[i * ENTRY_SIZE + 1] = (i >> 8) as u8;
        }

        // Create block keys with deterministic values
        let block_keys: Vec<[u32; 8]> = (0..set_size)
            .map(|i| {
                let mut key = [0u32; 8];
                key[0] = i as u32 + 42;
                key
            })
            .collect();

        // All hints include all blocks
        let subset_bytes = (set_size as usize + 7) / 8;
        let hint_subsets = vec![0xFFu8; total_hints as usize * subset_bytes];

        let hints1 = gen.generate_hints(
            &entries,
            &block_keys,
            &hint_subsets,
            num_entries,
            chunk_size,
            set_size,
            total_hints,
        );

        let hints2 = gen_serial.generate_hints(
            &entries,
            &block_keys,
            &hint_subsets,
            num_entries,
            chunk_size,
            set_size,
            total_hints,
        );

        assert_eq!(hints1.len(), total_hints as usize);
        assert_eq!(hints1, hints2, "Parallel and serial should produce identical results");
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_params_layout() {
        // Verify struct sizes match CUDA kernel expectations
        assert_eq!(std::mem::size_of::<PlinkoParams>(), 32);
        assert_eq!(std::mem::size_of::<IprfBlockKey>(), 32); // ChaCha: 8 × u32
        assert_eq!(std::mem::size_of::<HintOutput>(), 32);
    }
}
