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

/// iPRF key for one block (16 bytes AES key)
#[cfg(feature = "cuda")]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
#[repr(C)]
pub struct IprfBlockKey {
    pub key: [u8; 16],
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

/// CPU fallback hint generator (for testing without GPU)
pub struct CpuHintGenerator;

impl CpuHintGenerator {
    pub fn new() -> Self {
        Self
    }

    /// Generate hints using CPU (serial, for testing/validation).
    pub fn generate_hints(
        &self,
        entries: &[u8],
        _block_keys: &[[u8; 16]],
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
                // Check subset membership
                let byte_idx = hint_idx * subset_bytes_per_hint + (block_idx / 8);
                let bit_mask = 1u8 << (block_idx % 8);
                if (hint_subsets[byte_idx] & bit_mask) == 0 {
                    continue;
                }

                // Simplified: just XOR the first entry in each block for testing
                // Full implementation would use iPRF.inverse()
                let entry_idx = block_idx * chunk_size as usize;
                if entry_idx < num_entries as usize {
                    let entry_start = entry_idx * ENTRY_SIZE;
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

            // Convert parity to bytes
            for i in 0..4 {
                hints[hint_idx][i * 8..(i + 1) * 8].copy_from_slice(&parity[i].to_le_bytes());
            }
        }

        hints
    }
}

impl Default for CpuHintGenerator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

        // Create fake block keys
        let block_keys = vec![[0u8; 16]; set_size as usize];

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
    fn test_cpu_hint_generator_nonzero() {
        let gen = CpuHintGenerator::new();

        let num_entries = 4u64;
        let chunk_size = 2u64;
        let set_size = 2u64;
        let total_hints = 1u32;

        // Create entries with known values
        let mut entries = vec![0u8; num_entries as usize * ENTRY_SIZE];
        // Entry 0: first 8 bytes = 0x01
        entries[0] = 0x01;
        // Entry 2: first 8 bytes = 0x02
        entries[2 * ENTRY_SIZE] = 0x02;

        let block_keys = vec![[0u8; 16]; set_size as usize];

        // Hint 0 includes both blocks
        let hint_subsets = vec![0b11u8];

        let hints = gen.generate_hints(
            &entries,
            &block_keys,
            &hint_subsets,
            num_entries,
            chunk_size,
            set_size,
            total_hints,
        );

        // Parity should be 0x01 ^ 0x02 = 0x03
        assert_eq!(hints[0][0], 0x03);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_params_layout() {
        // Verify PlinkoParams is correctly sized for GPU
        assert_eq!(std::mem::size_of::<PlinkoParams>(), 32);
        assert_eq!(std::mem::size_of::<IprfBlockKey>(), 16);
        assert_eq!(std::mem::size_of::<HintOutput>(), 32);
    }
}
