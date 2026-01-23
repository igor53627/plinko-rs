/**
 * GPU Hint Generation Kernel for Plinko PIR
 *
 * Optimizations:
 * 1. Batched ChaCha8 - one block produces data for multiple SwapOrNot rounds
 * 2. Warp-level parallelism - 32 threads share block key loads via shuffle
 *
 * Uses ChaCha8 (ARX only - no memory lookups)
 */

#include <cstdint>
#include <cuda_runtime.h>

// Constants
#define ENTRY_SIZE 48
#define PARITY_SIZE 32
#define WARP_SIZE 32
#define CHACHA_ROUNDS 8

// Batching: ChaCha8 produces 512 bits, we need ~65 bits per SN round (64-bit key + 1 bit)
// So one ChaCha8 block can cover ~7 rounds
#define SN_ROUNDS 64
#define ROUNDS_PER_CHACHA 7

// Plinko parameters
struct PlinkoParams {
    uint64_t num_entries;
    uint64_t chunk_size;
    uint64_t set_size;
    uint32_t lambda;
    uint32_t total_hints;
    uint32_t blocks_per_hint;
    uint32_t _pad;
};

// iPRF key for one block (256-bit ChaCha key)
struct IprfBlockKey {
    uint32_t key[8];
};

// Hint output
struct HintOutput {
    uint8_t parity[PARITY_SIZE];
};

// ============================================================================
// ChaCha8 Implementation
// ============================================================================

__device__ __forceinline__ void chacha_quarter_round(
    uint32_t& a, uint32_t& b, uint32_t& c, uint32_t& d
) {
    a += b; d ^= a; d = (d << 16) | (d >> 16);
    c += d; b ^= c; b = (b << 12) | (b >> 20);
    a += b; d ^= a; d = (d << 8) | (d >> 24);
    c += d; b ^= c; b = (b << 7) | (b >> 25);
}

// ChaCha8 block - outputs 16 u32s (512 bits)
__device__ void chacha8_block(
    const uint32_t key[8],
    uint32_t counter,
    uint32_t nonce0,
    uint32_t output[16]
) {
    uint32_t state[16] = {
        0x61707865, 0x3320646e, 0x79622d32, 0x6b206574,
        key[0], key[1], key[2], key[3],
        key[4], key[5], key[6], key[7],
        counter, nonce0, 0, 0
    };

    uint32_t initial[16];
    #pragma unroll
    for (int i = 0; i < 16; i++) initial[i] = state[i];

    #pragma unroll
    for (int i = 0; i < CHACHA_ROUNDS / 2; i++) {
        chacha_quarter_round(state[0], state[4], state[8],  state[12]);
        chacha_quarter_round(state[1], state[5], state[9],  state[13]);
        chacha_quarter_round(state[2], state[6], state[10], state[14]);
        chacha_quarter_round(state[3], state[7], state[11], state[15]);
        chacha_quarter_round(state[0], state[5], state[10], state[15]);
        chacha_quarter_round(state[1], state[6], state[11], state[12]);
        chacha_quarter_round(state[2], state[7], state[8],  state[13]);
        chacha_quarter_round(state[3], state[4], state[9],  state[14]);
    }

    #pragma unroll
    for (int i = 0; i < 16; i++) output[i] = state[i] + initial[i];
}

// ============================================================================
// Batched SwapOrNot - process multiple rounds per ChaCha8 call
// ============================================================================

// Pre-generate random data for multiple rounds
struct BatchedRandom {
    uint64_t round_keys[ROUNDS_PER_CHACHA];  // K_i values
    uint8_t  prf_bits[ROUNDS_PER_CHACHA];    // swap decision bits
};

__device__ void generate_batch(
    const uint32_t key[8],
    uint32_t batch_idx,
    uint64_t domain,
    BatchedRandom& batch
) {
    uint32_t output[16];
    chacha8_block(key, batch_idx, 0x42415443, output);  // "BATC" tag

    // Extract round keys (need 64 bits each, mod domain)
    #pragma unroll
    for (int i = 0; i < ROUNDS_PER_CHACHA && i < 7; i++) {
        uint64_t raw = ((uint64_t)output[i*2 + 1] << 32) | output[i*2];
        batch.round_keys[i] = raw % domain;
    }

    // Extract PRF bits from remaining output
    uint32_t bits = output[14];
    #pragma unroll
    for (int i = 0; i < ROUNDS_PER_CHACHA; i++) {
        batch.prf_bits[i] = (bits >> i) & 1;
    }
}

// Get PRF bit for specific (round, canonical) - for non-batched cases
__device__ __forceinline__ bool get_prf_bit_direct(
    const uint32_t key[8],
    uint32_t round,
    uint64_t canonical
) {
    uint32_t output[16];
    chacha8_block(key, round | 0x80000000, (uint32_t)canonical, output);
    return (output[0] & 1) == 1;
}

// SwapOrNot inverse with batching
__device__ uint64_t sn_inverse_batched(
    const uint32_t key[8],
    uint64_t y,
    uint64_t domain
) {
    uint64_t val = y;
    BatchedRandom batch;

    for (int r = SN_ROUNDS - 1; r >= 0; r--) {
        int batch_idx = r / ROUNDS_PER_CHACHA;
        int in_batch = r % ROUNDS_PER_CHACHA;

        // Generate batch if needed (at start of each batch)
        if (in_batch == ROUNDS_PER_CHACHA - 1 || r == SN_ROUNDS - 1) {
            generate_batch(key, batch_idx, domain, batch);
        }

        uint64_t k_i = batch.round_keys[in_batch];
        uint64_t partner = (k_i + domain - (val % domain)) % domain;
        uint64_t canonical = (val > partner) ? val : partner;

        // For canonical-dependent PRF bit, we need direct computation
        // (batched bits are independent of canonical)
        bool swap = get_prf_bit_direct(key, r, canonical);

        if (swap) {
            val = partner;
        }
    }
    return val;
}

// ============================================================================
// Warp-optimized hint generation
// ============================================================================

/**
 * Warp-level parallel hint generation
 * - All 32 threads in a warp process different hints
 * - Block keys are broadcast via warp shuffle
 * - Reduces global memory reads by 32x for keys
 */
extern "C" __global__ void hint_gen_kernel_warp(
    const PlinkoParams params,
    const IprfBlockKey* __restrict__ block_keys,
    const uint8_t* __restrict__ entries,
    const uint8_t* __restrict__ hint_subsets,
    HintOutput* __restrict__ output
) {
    uint32_t hint_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (hint_idx >= params.total_hints) return;

    uint32_t lane = threadIdx.x % WARP_SIZE;

    uint64_t parity[4] = {0, 0, 0, 0};
    uint32_t subset_bytes = (params.set_size + 7) / 8;

    // Process blocks - lane 0 loads key, broadcasts to warp
    for (uint64_t block_idx = 0; block_idx < params.set_size; block_idx++) {
        // Check subset membership (each thread checks its own hint)
        uint32_t byte_idx = hint_idx * subset_bytes + (block_idx / 8);
        uint8_t bit_mask = 1 << (block_idx % 8);
        bool in_subset = (hint_subsets[byte_idx] & bit_mask) != 0;

        // Ballot to see if any thread in warp needs this block
        uint32_t warp_mask = __ballot_sync(0xFFFFFFFF, in_subset);
        if (warp_mask == 0) continue;  // No thread needs this block

        // Lane 0 loads the block key
        uint32_t key[8];
        if (lane == 0) {
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                key[i] = block_keys[block_idx].key[i];
            }
        }

        // Broadcast key to all lanes via shuffle
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            key[i] = __shfl_sync(0xFFFFFFFF, key[i], 0);
        }

        // Each thread computes iPRF inverse for its hint (if in subset)
        if (in_subset) {
            uint64_t preimage = sn_inverse_batched(key, hint_idx, params.chunk_size);

            if (preimage < params.chunk_size) {
                uint64_t entry_idx = block_idx * params.chunk_size + preimage;
                if (entry_idx < params.num_entries) {
                    const uint64_t* entry_ptr = (const uint64_t*)(entries + entry_idx * ENTRY_SIZE);
                    parity[0] ^= entry_ptr[0];
                    parity[1] ^= entry_ptr[1];
                    parity[2] ^= entry_ptr[2];
                    parity[3] ^= entry_ptr[3];
                }
            }
        }
    }

    // Write output
    uint64_t* out_ptr = (uint64_t*)output[hint_idx].parity;
    out_ptr[0] = parity[0];
    out_ptr[1] = parity[1];
    out_ptr[2] = parity[2];
    out_ptr[3] = parity[3];
}

// ============================================================================
// Original kernels (kept for comparison)
// ============================================================================

// Simple SwapOrNot inverse (non-batched, for comparison)
__device__ uint64_t sn_inverse_simple(
    const uint32_t key[8],
    uint64_t y,
    uint64_t domain
) {
    uint64_t val = y;
    for (int r = SN_ROUNDS - 1; r >= 0; r--) {
        // Derive round key
        uint32_t output[16];
        chacha8_block(key, r, 0, output);
        uint64_t k_i = (((uint64_t)output[1] << 32) | output[0]) % domain;

        uint64_t partner = (k_i + domain - (val % domain)) % domain;
        uint64_t canonical = (val > partner) ? val : partner;

        // PRF bit
        chacha8_block(key, r | 0x80000000, (uint32_t)canonical, output);
        if (output[0] & 1) {
            val = partner;
        }
    }
    return val;
}

extern "C" __global__ void hint_gen_kernel(
    const PlinkoParams params,
    const IprfBlockKey* __restrict__ block_keys,
    const uint8_t* __restrict__ entries,
    const uint8_t* __restrict__ hint_subsets,
    HintOutput* __restrict__ output
) {
    uint32_t hint_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (hint_idx >= params.total_hints) return;

    uint64_t parity[4] = {0, 0, 0, 0};
    uint32_t subset_bytes = (params.set_size + 7) / 8;

    for (uint64_t block_idx = 0; block_idx < params.set_size; block_idx++) {
        uint32_t byte_idx = hint_idx * subset_bytes + (block_idx / 8);
        uint8_t bit_mask = 1 << (block_idx % 8);
        if ((hint_subsets[byte_idx] & bit_mask) == 0) continue;

        const uint32_t* key = block_keys[block_idx].key;
        uint64_t preimage = sn_inverse_simple(key, hint_idx, params.chunk_size);

        if (preimage < params.chunk_size) {
            uint64_t entry_idx = block_idx * params.chunk_size + preimage;
            if (entry_idx < params.num_entries) {
                const uint64_t* entry_ptr = (const uint64_t*)(entries + entry_idx * ENTRY_SIZE);
                parity[0] ^= entry_ptr[0];
                parity[1] ^= entry_ptr[1];
                parity[2] ^= entry_ptr[2];
                parity[3] ^= entry_ptr[3];
            }
        }
    }

    uint64_t* out_ptr = (uint64_t*)output[hint_idx].parity;
    out_ptr[0] = parity[0];
    out_ptr[1] = parity[1];
    out_ptr[2] = parity[2];
    out_ptr[3] = parity[3];
}

// Tiled version - same as warp-optimized
extern "C" __global__ void hint_gen_kernel_tiled(
    const PlinkoParams params,
    const IprfBlockKey* __restrict__ block_keys,
    const uint8_t* __restrict__ entries,
    const uint8_t* __restrict__ hint_subsets,
    HintOutput* __restrict__ output
) {
    uint32_t hint_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (hint_idx >= params.total_hints) return;

    uint32_t lane = threadIdx.x % WARP_SIZE;

    uint64_t parity[4] = {0, 0, 0, 0};
    uint32_t subset_bytes = (params.set_size + 7) / 8;

    for (uint64_t block_idx = 0; block_idx < params.set_size; block_idx++) {
        uint32_t byte_idx = hint_idx * subset_bytes + (block_idx / 8);
        uint8_t bit_mask = 1 << (block_idx % 8);
        bool in_subset = (hint_subsets[byte_idx] & bit_mask) != 0;

        uint32_t warp_mask = __ballot_sync(0xFFFFFFFF, in_subset);
        if (warp_mask == 0) continue;

        uint32_t key[8];
        if (lane == 0) {
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                key[i] = block_keys[block_idx].key[i];
            }
        }

        #pragma unroll
        for (int i = 0; i < 8; i++) {
            key[i] = __shfl_sync(0xFFFFFFFF, key[i], 0);
        }

        if (in_subset) {
            uint64_t preimage = sn_inverse_batched(key, hint_idx, params.chunk_size);

            if (preimage < params.chunk_size) {
                uint64_t entry_idx = block_idx * params.chunk_size + preimage;
                if (entry_idx < params.num_entries) {
                    const uint64_t* entry_ptr = (const uint64_t*)(entries + entry_idx * ENTRY_SIZE);
                    parity[0] ^= entry_ptr[0];
                    parity[1] ^= entry_ptr[1];
                    parity[2] ^= entry_ptr[2];
                    parity[3] ^= entry_ptr[3];
                }
            }
        }
    }

    uint64_t* out_ptr = (uint64_t*)output[hint_idx].parity;
    out_ptr[0] = parity[0];
    out_ptr[1] = parity[1];
    out_ptr[2] = parity[2];
    out_ptr[3] = parity[3];
}
