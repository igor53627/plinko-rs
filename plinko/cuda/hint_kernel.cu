/**
 * GPU Hint Generation Kernel for Plinko PIR
 *
 * Uses ChaCha8 instead of AES for GPU-friendly PRF operations.
 * ChaCha uses only ARX (Add, Rotate, XOR) - no lookup tables!
 *
 * Parallelization strategy: One thread per hint
 * - Each thread computes one hint's parity
 * - For each hint, iterate over blocks in subset
 * - Use iPRF.inverse() to find contributing entries per block
 * - XOR entries into parity accumulator
 *
 * Entry format: 48 bytes (3 × 16B for GPU alignment)
 * - Account: Balance(16B) | Nonce(8B) | CodeID(4B) | TAG(8B) | Pad(12B)
 * - Storage: Value(32B) | TAG(8B) | Pad(8B)
 */

#include <cstdint>
#include <cuda_runtime.h>

// Constants
#define ENTRY_SIZE 48
#define PARITY_SIZE 32
#define MAX_PREIMAGES 512
#define CHACHA_ROUNDS 8  // ChaCha8 for speed (ChaCha20 for max security)

// Plinko parameters (set at runtime)
struct PlinkoParams {
    uint64_t num_entries;      // N
    uint64_t chunk_size;       // w (block size)
    uint64_t set_size;         // c (number of blocks)
    uint32_t lambda;           // security parameter
    uint32_t total_hints;      // 2λw
    uint32_t blocks_per_hint;  // c/2 or c/2+1
    uint32_t _pad;
};

// iPRF key for one block (256-bit key for ChaCha)
struct IprfBlockKey {
    uint32_t key[8];  // 256-bit key as 8 × u32
};

// Hint output structure
struct HintOutput {
    uint8_t parity[PARITY_SIZE];  // XOR parity
};

// ============================================================================
// ChaCha8 Implementation - Pure ARX, no memory lookups!
// ============================================================================

// ChaCha quarter round - the core operation
__device__ __forceinline__ void chacha_quarter_round(
    uint32_t& a, uint32_t& b, uint32_t& c, uint32_t& d
) {
    a += b; d ^= a; d = (d << 16) | (d >> 16);
    c += d; b ^= c; b = (b << 12) | (b >> 20);
    a += b; d ^= a; d = (d << 8) | (d >> 24);
    c += d; b ^= c; b = (b << 7) | (b >> 25);
}

// ChaCha8 block function
// Input: 256-bit key (8 words), 96-bit nonce (3 words), 32-bit counter
// Output: 512-bit keystream (16 words)
__device__ void chacha8_block(
    const uint32_t key[8],
    uint32_t counter,
    const uint32_t nonce[3],
    uint32_t output[16]
) {
    // Initialize state with constants, key, counter, nonce
    // "expand 32-byte k" in ASCII
    uint32_t state[16] = {
        0x61707865, 0x3320646e, 0x79622d32, 0x6b206574,  // Constants
        key[0], key[1], key[2], key[3],                   // Key part 1
        key[4], key[5], key[6], key[7],                   // Key part 2
        counter, nonce[0], nonce[1], nonce[2]             // Counter + Nonce
    };

    // Copy initial state for final addition
    uint32_t initial[16];
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        initial[i] = state[i];
    }

    // 8 rounds (4 double-rounds)
    #pragma unroll
    for (int i = 0; i < CHACHA_ROUNDS / 2; i++) {
        // Column rounds
        chacha_quarter_round(state[0], state[4], state[8],  state[12]);
        chacha_quarter_round(state[1], state[5], state[9],  state[13]);
        chacha_quarter_round(state[2], state[6], state[10], state[14]);
        chacha_quarter_round(state[3], state[7], state[11], state[15]);
        // Diagonal rounds
        chacha_quarter_round(state[0], state[5], state[10], state[15]);
        chacha_quarter_round(state[1], state[6], state[11], state[12]);
        chacha_quarter_round(state[2], state[7], state[8],  state[13]);
        chacha_quarter_round(state[3], state[4], state[9],  state[14]);
    }

    // Final addition
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        output[i] = state[i] + initial[i];
    }
}

// Generate PRF output using ChaCha8
// Returns 64 bits of PRF output for the given input
__device__ __forceinline__ uint64_t chacha8_prf(
    const uint32_t key[8],
    uint64_t input
) {
    uint32_t nonce[3] = {
        (uint32_t)(input & 0xFFFFFFFF),
        (uint32_t)(input >> 32),
        0
    };
    uint32_t output[16];
    chacha8_block(key, 0, nonce, output);
    return ((uint64_t)output[1] << 32) | output[0];
}

// ============================================================================
// SwapOrNot PRP using ChaCha8
// ============================================================================

// Number of SwapOrNot rounds - fewer needed with ChaCha8's better mixing
#define SN_ROUNDS 64  // Reduced from 198 with AES

// Derive round key K_i for SwapOrNot
__device__ __forceinline__ uint64_t sn_derive_round_key(
    const uint32_t key[8],
    uint32_t round,
    uint64_t domain
) {
    // Use round as part of the nonce
    uint32_t nonce[3] = { round, 0, 0 };
    uint32_t output[16];
    chacha8_block(key, 0, nonce, output);

    uint64_t k = ((uint64_t)output[1] << 32) | output[0];
    return k % domain;
}

// PRF bit for SwapOrNot decision
__device__ __forceinline__ bool sn_prf_bit(
    const uint32_t key[8],
    uint32_t round,
    uint64_t canonical
) {
    // Tag round with high bit to differentiate from key derivation
    uint32_t nonce[3] = {
        round | 0x80000000,
        (uint32_t)(canonical & 0xFFFFFFFF),
        (uint32_t)(canonical >> 32)
    };
    uint32_t output[16];
    chacha8_block(key, 0, nonce, output);
    return (output[0] & 1) == 1;
}

// One round of SwapOrNot (involutory - same operation for encrypt/decrypt)
__device__ __forceinline__ uint64_t sn_round(
    const uint32_t key[8],
    uint32_t round,
    uint64_t x,
    uint64_t domain
) {
    uint64_t k_i = sn_derive_round_key(key, round, domain);
    uint64_t partner = (k_i + domain - (x % domain)) % domain;
    uint64_t canonical = (x > partner) ? x : partner;

    if (sn_prf_bit(key, round, canonical)) {
        return partner;
    }
    return x;
}

// SwapOrNot inverse (involutory, so same as forward with reversed rounds)
__device__ uint64_t sn_inverse(
    const uint32_t key[8],
    uint64_t y,
    uint64_t domain
) {
    uint64_t val = y;
    for (int r = SN_ROUNDS - 1; r >= 0; r--) {
        val = sn_round(key, r, val, domain);
    }
    return val;
}

// ============================================================================
// iPRF inverse (simplified for GPU prototype)
// ============================================================================

// For pure SwapOrNot PRP, there's exactly one preimage
__device__ uint32_t iprf_inverse_simple(
    const uint32_t key[8],
    uint64_t y,
    uint64_t domain,
    uint64_t* preimages,
    uint32_t max_preimages
) {
    if (max_preimages == 0) return 0;
    preimages[0] = sn_inverse(key, y, domain);
    return 1;
}

// ============================================================================
// Hint Generation Kernel
// ============================================================================

/**
 * Kernel: Compute hint parities using ChaCha8-based iPRF
 *
 * Each thread computes the parity for one hint by:
 * 1. Determining which blocks are in this hint's subset
 * 2. For each block, finding entries via iPRF.inverse()
 * 3. XORing contributing entries into parity
 */
extern "C" __global__ void hint_gen_kernel(
    const PlinkoParams params,
    const IprfBlockKey* __restrict__ block_keys,
    const uint8_t* __restrict__ entries,
    const uint8_t* __restrict__ hint_subsets,  // Bitset: total_hints × ceil(set_size/8) bytes
    HintOutput* __restrict__ output
) {
    uint32_t hint_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (hint_idx >= params.total_hints) return;

    // Initialize parity to zero
    uint64_t parity[4] = {0, 0, 0, 0};  // 32 bytes as 4 × u64

    uint32_t subset_bytes = (params.set_size + 7) / 8;

    // Iterate over all blocks
    for (uint64_t block_idx = 0; block_idx < params.set_size; block_idx++) {
        // Check if block is in this hint's subset
        uint32_t byte_idx = hint_idx * subset_bytes + (block_idx / 8);
        uint8_t bit_mask = 1 << (block_idx % 8);

        if ((hint_subsets[byte_idx] & bit_mask) == 0) continue;

        // Get block's iPRF key
        const uint32_t* key = block_keys[block_idx].key;

        // Find preimages for hint_idx in this block
        uint64_t preimages[MAX_PREIMAGES];
        uint32_t num_preimages = iprf_inverse_simple(
            key, hint_idx, params.chunk_size, preimages, MAX_PREIMAGES
        );

        // XOR contributing entries
        for (uint32_t p = 0; p < num_preimages; p++) {
            uint64_t pos_in_block = preimages[p];
            if (pos_in_block >= params.chunk_size) continue;

            uint64_t entry_idx = block_idx * params.chunk_size + pos_in_block;
            if (entry_idx >= params.num_entries) continue;

            // Read entry and XOR into parity (first 32 bytes)
            const uint64_t* entry_ptr = (const uint64_t*)(entries + entry_idx * ENTRY_SIZE);
            parity[0] ^= entry_ptr[0];
            parity[1] ^= entry_ptr[1];
            parity[2] ^= entry_ptr[2];
            parity[3] ^= entry_ptr[3];
        }
    }

    // Write output
    uint64_t* out_ptr = (uint64_t*)output[hint_idx].parity;
    out_ptr[0] = parity[0];
    out_ptr[1] = parity[1];
    out_ptr[2] = parity[2];
    out_ptr[3] = parity[3];
}

/**
 * Optimized kernel with shared memory for block keys
 */
extern "C" __global__ void hint_gen_kernel_tiled(
    const PlinkoParams params,
    const IprfBlockKey* __restrict__ block_keys,
    const uint8_t* __restrict__ entries,
    const uint8_t* __restrict__ hint_subsets,
    HintOutput* __restrict__ output
) {
    // Shared memory for block keys (one tile at a time)
    __shared__ IprfBlockKey s_block_keys[32];

    uint32_t hint_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (hint_idx >= params.total_hints) return;

    uint64_t parity[4] = {0, 0, 0, 0};
    uint32_t subset_bytes = (params.set_size + 7) / 8;

    // Process blocks in tiles
    for (uint64_t tile_start = 0; tile_start < params.set_size; tile_start += 32) {
        // Load block keys into shared memory
        if (threadIdx.x < 32 && tile_start + threadIdx.x < params.set_size) {
            s_block_keys[threadIdx.x] = block_keys[tile_start + threadIdx.x];
        }
        __syncthreads();

        // Process blocks in this tile
        uint64_t tile_end = min((uint64_t)(tile_start + 32), params.set_size);
        for (uint64_t block_idx = tile_start; block_idx < tile_end; block_idx++) {
            // Check subset membership
            uint32_t byte_idx = hint_idx * subset_bytes + (block_idx / 8);
            uint8_t bit_mask = 1 << (block_idx % 8);
            if ((hint_subsets[byte_idx] & bit_mask) == 0) continue;

            const uint32_t* key = s_block_keys[block_idx - tile_start].key;

            uint64_t preimages[MAX_PREIMAGES];
            uint32_t num_preimages = iprf_inverse_simple(
                key, hint_idx, params.chunk_size, preimages, MAX_PREIMAGES
            );

            for (uint32_t p = 0; p < num_preimages; p++) {
                uint64_t pos_in_block = preimages[p];
                if (pos_in_block >= params.chunk_size) continue;

                uint64_t entry_idx = block_idx * params.chunk_size + pos_in_block;
                if (entry_idx >= params.num_entries) continue;

                const uint64_t* entry_ptr = (const uint64_t*)(entries + entry_idx * ENTRY_SIZE);
                parity[0] ^= entry_ptr[0];
                parity[1] ^= entry_ptr[1];
                parity[2] ^= entry_ptr[2];
                parity[3] ^= entry_ptr[3];
            }
        }
        __syncthreads();
    }

    uint64_t* out_ptr = (uint64_t*)output[hint_idx].parity;
    out_ptr[0] = parity[0];
    out_ptr[1] = parity[1];
    out_ptr[2] = parity[2];
    out_ptr[3] = parity[3];
}
