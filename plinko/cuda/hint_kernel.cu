/**
 * GPU Hint Generation Kernel for Plinko PIR
 *
 * Key derivation:
 * - Block keys are derived using SHA-256: prp_key = SHA256(block_key || "prp")
 * - This matches the production Rust implementation
 *
 * Optimizations:
 * 1. Batched ChaCha8 - one block produces data for multiple SwapOrNot rounds
 * 2. Warp-level parallelism - 32 threads share block key loads via shuffle
 * 3. SHA-256 key derivation done once per block per warp, then broadcast
 * 4. Fast PRF Bit - eliminates dead code in ChaCha8 for the swap decision bit
 * 5. Vectorized Loads - uses 128-bit loads (ulong2) for database entries
 *
 * Uses ChaCha8 for PRP (ARX only - no memory lookups)
 * Uses SHA-256 for key derivation (ARX + bitwise ops)
 */

#include <cstdint>
#include <cuda_runtime.h>

// Constants
#define ENTRY_SIZE 40  // v3 schema: 40-byte entries, use 64-bit loads (not 128-bit)
#define PARITY_SIZE 32
#define WARP_SIZE 32
#define CHACHA_ROUNDS 8

// Batching: ChaCha8 produces 512 bits, we need ~65 bits per SN round (64-bit key + 1 bit)
// So one ChaCha8 block can cover ~7 rounds
#define SN_ROUNDS 759
#define ROUNDS_PER_CHACHA 7

// Plinko parameters
struct PlinkoParams {
    uint64_t num_entries;
    uint64_t chunk_size;
    uint64_t set_size;
    uint32_t lambda;
    uint32_t total_hints;
    uint32_t blocks_per_hint;
    uint32_t hint_start_offset;
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

// Optimization: Only compute first word of output for the PRF bit
// Saves 15 additions and stores per round
__device__ __forceinline__ bool get_prf_bit_direct_fast(
    const uint32_t key[8],
    uint32_t round,
    uint64_t canonical
) {
    uint32_t state[16] = {
        0x61707865, 0x3320646e, 0x79622d32, 0x6b206574,
        key[0], key[1], key[2], key[3],
        key[4], key[5], key[6], key[7],
        round | 0x80000000, (uint32_t)canonical, 0, 0
    };

    // We can't optimize the rounds themselves (mixing is required)
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

    // Only add initial state for index 0 (constant 0x61707865)
    return ((state[0] + 0x61707865) & 1) == 1;
}

// ============================================================================
// SHA-256 Implementation (for key derivation)
// ============================================================================

// SHA-256 round constants
__device__ __constant__ uint32_t SHA256_K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
    0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

// SHA-256 initial hash values
__device__ __constant__ uint32_t SHA256_H0[8] = {
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
};

__device__ __forceinline__ uint32_t sha256_rotr(uint32_t x, int n) {
    return (x >> n) | (x << (32 - n));
}

__device__ __forceinline__ uint32_t sha256_ch(uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) ^ (~x & z);
}

__device__ __forceinline__ uint32_t sha256_maj(uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) ^ (x & z) ^ (y & z);
}

__device__ __forceinline__ uint32_t sha256_sigma0(uint32_t x) {
    return sha256_rotr(x, 2) ^ sha256_rotr(x, 13) ^ sha256_rotr(x, 22);
}

__device__ __forceinline__ uint32_t sha256_sigma1(uint32_t x) {
    return sha256_rotr(x, 6) ^ sha256_rotr(x, 11) ^ sha256_rotr(x, 25);
}

__device__ __forceinline__ uint32_t sha256_gamma0(uint32_t x) {
    return sha256_rotr(x, 7) ^ sha256_rotr(x, 18) ^ (x >> 3);
}

__device__ __forceinline__ uint32_t sha256_gamma1(uint32_t x) {
    return sha256_rotr(x, 17) ^ sha256_rotr(x, 19) ^ (x >> 10);
}

// SHA-256 hash of (key || suffix) -> 32 bytes output
// Used for key derivation: SHA256(block_key || "prp") -> prp_key
__device__ void sha256_key_derive(
    const uint32_t key[8],      // 256-bit input key
    const uint8_t* suffix,      // suffix bytes (e.g., "prp")
    uint32_t suffix_len,        // length of suffix
    uint32_t output[8]          // 256-bit output hash
) {
    // Initialize state
    uint32_t state[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) state[i] = SHA256_H0[i];

    // Build message block (512 bits = 64 bytes)
    // key (32 bytes) + suffix + padding
    uint32_t w[64];

    // Load key (32 bytes = 8 words) - big endian
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        uint32_t k = key[i];
        // Convert to big endian for SHA-256
        w[i] = ((k & 0xFF) << 24) | ((k & 0xFF00) << 8) |
               ((k >> 8) & 0xFF00) | ((k >> 24) & 0xFF);
    }

    // Load suffix into w[8] (up to 4 bytes)
    uint32_t suffix_word = 0;
    for (uint32_t i = 0; i < suffix_len && i < 4; i++) {
        suffix_word |= ((uint32_t)suffix[i]) << (24 - i * 8);
    }

    // Add padding bit after suffix
    uint32_t msg_len = 32 + suffix_len;  // total message length in bytes
    uint32_t pad_pos = suffix_len;       // position within word for 0x80

    if (pad_pos < 4) {
        suffix_word |= 0x80 << (24 - pad_pos * 8);
        w[8] = suffix_word;
    } else {
        w[8] = suffix_word;
        w[9] = 0x80000000;
    }

    // Zero padding
    // Fix: correctly start zeroing from 9 or 10 depending on padding position
    int zero_start = (pad_pos < 4) ? 9 : 10;
    #pragma unroll
    for (int i = zero_start; i < 15; i++) w[i] = 0;

    // Message length in bits (big endian, 64-bit)
    w[14] = 0;
    w[15] = msg_len * 8;

    // Message schedule expansion
    #pragma unroll
    for (int i = 16; i < 64; i++) {
        w[i] = sha256_gamma1(w[i-2]) + w[i-7] + sha256_gamma0(w[i-15]) + w[i-16];
    }

    // Compression
    uint32_t a = state[0], b = state[1], c = state[2], d = state[3];
    uint32_t e = state[4], f = state[5], g = state[6], h = state[7];

    #pragma unroll
    for (int i = 0; i < 64; i++) {
        uint32_t t1 = h + sha256_sigma1(e) + sha256_ch(e, f, g) + SHA256_K[i] + w[i];
        uint32_t t2 = sha256_sigma0(a) + sha256_maj(a, b, c);
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }

    // Output (convert back from big endian)
    uint32_t h0 = state[0] + a, h1 = state[1] + b, h2 = state[2] + c, h3 = state[3] + d;
    uint32_t h4 = state[4] + e, h5 = state[5] + f, h6 = state[6] + g, h7 = state[7] + h;

    output[0] = ((h0 & 0xFF) << 24) | ((h0 & 0xFF00) << 8) | ((h0 >> 8) & 0xFF00) | ((h0 >> 24) & 0xFF);
    output[1] = ((h1 & 0xFF) << 24) | ((h1 & 0xFF00) << 8) | ((h1 >> 8) & 0xFF00) | ((h1 >> 24) & 0xFF);
    output[2] = ((h2 & 0xFF) << 24) | ((h2 & 0xFF00) << 8) | ((h2 >> 8) & 0xFF00) | ((h2 >> 24) & 0xFF);
    output[3] = ((h3 & 0xFF) << 24) | ((h3 & 0xFF00) << 8) | ((h3 >> 8) & 0xFF00) | ((h3 >> 24) & 0xFF);
    output[4] = ((h4 & 0xFF) << 24) | ((h4 & 0xFF00) << 8) | ((h4 >> 8) & 0xFF00) | ((h4 >> 24) & 0xFF);
    output[5] = ((h5 & 0xFF) << 24) | ((h5 & 0xFF00) << 8) | ((h5 >> 8) & 0xFF00) | ((h5 >> 24) & 0xFF);
    output[6] = ((h6 & 0xFF) << 24) | ((h6 & 0xFF00) << 8) | ((h6 >> 8) & 0xFF00) | ((h6 >> 24) & 0xFF);
    output[7] = ((h7 & 0xFF) << 24) | ((h7 & 0xFF00) << 8) | ((h7 >> 8) & 0xFF00) | ((h7 >> 24) & 0xFF);
}

// Derive PRP key from block key: SHA256(block_key || "prp") (32-byte output)
__device__ void derive_prp_key(const uint32_t block_key[8], uint32_t prp_key[8]) {
    const uint8_t suffix[3] = {'p', 'r', 'p'};
    sha256_key_derive(block_key, suffix, 3, prp_key);
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

// Optimized SwapOrNot using fast PRF bit extraction
__device__ uint64_t sn_inverse_batched_fast(
    const uint32_t key[8],
    uint64_t y,
    uint64_t domain
) {
    uint64_t val = y;
    BatchedRandom batch;

    for (int r = SN_ROUNDS - 1; r >= 0; r--) {
        int batch_idx = r / ROUNDS_PER_CHACHA;
        int in_batch = r % ROUNDS_PER_CHACHA;

        if (in_batch == ROUNDS_PER_CHACHA - 1 || r == SN_ROUNDS - 1) {
            generate_batch(key, batch_idx, domain, batch);
        }

        uint64_t k_i = batch.round_keys[in_batch];
        uint64_t partner = (k_i + domain - (val % domain)) % domain;
        uint64_t canonical = (val > partner) ? val : partner;

        // Optimization: Use fast PRF bit extractor (skips unused output words)
        bool swap = get_prf_bit_direct_fast(key, r, canonical);

        if (swap) {
            val = partner;
        }
    }
    return val;
}

// ============================================================================
// Warp-optimized hint generation
// ============================================================================

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

        // Lane 0 loads the block key and derives PRP key
        uint32_t block_key[8];
        uint32_t prp_key[8];
        if (lane == 0) {
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                block_key[i] = block_keys[block_idx].key[i];
            }
            // Derive PRP key: SHA256(block_key || "prp")
            derive_prp_key(block_key, prp_key);
        }

        // Broadcast derived PRP key to all lanes via shuffle
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            prp_key[i] = __shfl_sync(0xFFFFFFFF, prp_key[i], 0);
        }

        // Each thread computes iPRF inverse for its hint (if in subset)
        if (in_subset) {
            // Apply hint start offset for the actual iPRF value
            uint64_t global_hint_idx = (uint64_t)hint_idx + params.hint_start_offset;
            uint64_t preimage = sn_inverse_batched(prp_key, global_hint_idx, params.chunk_size);

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

        // Load block key and derive PRP key
        uint32_t prp_key[8];
        derive_prp_key(block_keys[block_idx].key, prp_key);
        
        // Apply hint start offset for the actual iPRF value
        uint64_t global_hint_idx = (uint64_t)hint_idx + params.hint_start_offset;
        uint64_t preimage = sn_inverse_simple(prp_key, global_hint_idx, params.chunk_size);

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

        uint32_t block_key[8];
        uint32_t prp_key[8];
        if (lane == 0) {
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                block_key[i] = block_keys[block_idx].key[i];
            }
            // Derive PRP key: SHA256(block_key || "prp")
            derive_prp_key(block_key, prp_key);
        }

        #pragma unroll
        for (int i = 0; i < 8; i++) {
            prp_key[i] = __shfl_sync(0xFFFFFFFF, prp_key[i], 0);
        }

        if (in_subset) {
            // Apply hint start offset for the actual iPRF value
            uint64_t global_hint_idx = (uint64_t)hint_idx + params.hint_start_offset;
            uint64_t preimage = sn_inverse_batched(prp_key, global_hint_idx, params.chunk_size);

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

/**
 * Fully Optimized Kernel
 * - Uses sn_inverse_batched_fast (eliminates dead ChaCha code)
 * - Uses 128-bit vector loads (ulong2) for database reads
 * - ASSUMES pre-derived PRP keys (no SHA-256 in loop)
 */
extern "C" __global__ void hint_gen_kernel_opt(
    const PlinkoParams params,
    const IprfBlockKey* __restrict__ prp_keys,
    const uint8_t* __restrict__ entries,
    const uint8_t* __restrict__ hint_subsets,
    HintOutput* __restrict__ output
) {
    uint32_t hint_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (hint_idx >= params.total_hints) return;

    uint32_t lane = threadIdx.x % WARP_SIZE;

    // Use 128-bit vector accumulators
    ulong2 parity_vec[2];
    parity_vec[0] = make_ulong2(0, 0);
    parity_vec[1] = make_ulong2(0, 0);
    
    uint32_t subset_bytes = (params.set_size + 7) / 8;

    for (uint64_t block_idx = 0; block_idx < params.set_size; block_idx++) {
        uint32_t byte_idx = hint_idx * subset_bytes + (block_idx / 8);
        uint8_t bit_mask = 1 << (block_idx % 8);
        bool in_subset = (hint_subsets[byte_idx] & bit_mask) != 0;

        uint32_t warp_mask = __ballot_sync(0xFFFFFFFF, in_subset);
        if (warp_mask == 0) continue;

        uint32_t prp_key[8];
        // Optimization: Directly load pre-derived PRP key
        // Still use warp-broadcast to minimize memory pressure
        if (lane == 0) {
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                prp_key[i] = prp_keys[block_idx].key[i];
            }
        }

        #pragma unroll
        for (int i = 0; i < 8; i++) {
            prp_key[i] = __shfl_sync(0xFFFFFFFF, prp_key[i], 0);
        }

        if (in_subset) {
            // Apply hint start offset for the actual iPRF value
            // hint_idx is local (0..total_hints-1), global is shifted
            uint64_t global_hint_idx = (uint64_t)hint_idx + params.hint_start_offset;
            uint64_t preimage = sn_inverse_batched_fast(prp_key, global_hint_idx, params.chunk_size);

            if (preimage < params.chunk_size) {
                uint64_t entry_idx = block_idx * params.chunk_size + preimage;
                if (entry_idx < params.num_entries) {
                    // Use 128-bit loads (ulong2) even for 40-byte entries
                    // Hardware handles misalignment (split transactions) better than 4x scalar loads
                    const ulong2* vec_ptr = (const ulong2*)(entries + entry_idx * ENTRY_SIZE);
                    
                    ulong2 v0 = vec_ptr[0];
                    ulong2 v1 = vec_ptr[1];

                    parity_vec[0].x ^= v0.x;
                    parity_vec[0].y ^= v0.y;
                    parity_vec[1].x ^= v1.x;
                    parity_vec[1].y ^= v1.y;
                }
            }
        }
    }

    ulong2* out_ptr = (ulong2*)output[hint_idx].parity;
    out_ptr[0] = parity_vec[0];
    out_ptr[1] = parity_vec[1];
}