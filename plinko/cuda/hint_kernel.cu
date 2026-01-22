/**
 * GPU Hint Generation Kernel for Plinko PIR
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
#define ENTRY_U64_COUNT 6
#define PARITY_SIZE 32  // XOR parity is 32 bytes (first 32 bytes of 48-byte entry)
#define MAX_PREIMAGES 512
#define WARP_SIZE 32

// AES-128 S-box (for SwapOrNot round function)
__constant__ uint8_t AES_SBOX[256] = {
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
    0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
    0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
    0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
    0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
    0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
    0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
    0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
    0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16
};

// AES round constants
__constant__ uint8_t AES_RCON[11] = {
    0x00, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36
};

// Plinko parameters (set at runtime)
struct PlinkoParams {
    uint64_t num_entries;      // N
    uint64_t chunk_size;       // w (block size)
    uint64_t set_size;         // c (number of blocks)
    uint32_t lambda;           // security parameter
    uint32_t total_hints;      // 2λw
    uint32_t blocks_per_hint;  // c/2 or c/2+1
};

// iPRF key for one block
struct IprfBlockKey {
    uint8_t key[16];  // AES-128 key
};

// Hint output structure
struct HintOutput {
    uint8_t parity[PARITY_SIZE];  // XOR parity
};

// ============================================================================
// AES-128 Implementation (for SwapOrNot round function)
// ============================================================================

__device__ __forceinline__ void aes_sub_bytes(uint8_t state[16]) {
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        state[i] = AES_SBOX[state[i]];
    }
}

__device__ __forceinline__ void aes_shift_rows(uint8_t state[16]) {
    uint8_t tmp;
    // Row 1: shift left by 1
    tmp = state[1]; state[1] = state[5]; state[5] = state[9]; state[9] = state[13]; state[13] = tmp;
    // Row 2: shift left by 2
    tmp = state[2]; state[2] = state[10]; state[10] = tmp;
    tmp = state[6]; state[6] = state[14]; state[14] = tmp;
    // Row 3: shift left by 3
    tmp = state[15]; state[15] = state[11]; state[11] = state[7]; state[7] = state[3]; state[3] = tmp;
}

__device__ __forceinline__ uint8_t xtime(uint8_t x) {
    return ((x << 1) ^ (((x >> 7) & 1) * 0x1b));
}

__device__ __forceinline__ void aes_mix_columns(uint8_t state[16]) {
    for (int i = 0; i < 4; i++) {
        uint8_t* col = &state[i * 4];
        uint8_t a = col[0], b = col[1], c = col[2], d = col[3];
        uint8_t h = a ^ b ^ c ^ d;
        col[0] ^= h ^ xtime(a ^ b);
        col[1] ^= h ^ xtime(b ^ c);
        col[2] ^= h ^ xtime(c ^ d);
        col[3] ^= h ^ xtime(d ^ a);
    }
}

__device__ __forceinline__ void aes_add_round_key(uint8_t state[16], const uint8_t* round_key) {
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        state[i] ^= round_key[i];
    }
}

// AES-128 key expansion
__device__ void aes_key_expand(const uint8_t key[16], uint8_t expanded[176]) {
    // Copy original key
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        expanded[i] = key[i];
    }

    // Expand to 11 round keys
    for (int i = 1; i <= 10; i++) {
        uint8_t* prev = &expanded[(i - 1) * 16];
        uint8_t* curr = &expanded[i * 16];

        // RotWord + SubWord + Rcon
        uint8_t tmp[4];
        tmp[0] = AES_SBOX[prev[13]] ^ AES_RCON[i];
        tmp[1] = AES_SBOX[prev[14]];
        tmp[2] = AES_SBOX[prev[15]];
        tmp[3] = AES_SBOX[prev[12]];

        curr[0] = prev[0] ^ tmp[0];
        curr[1] = prev[1] ^ tmp[1];
        curr[2] = prev[2] ^ tmp[2];
        curr[3] = prev[3] ^ tmp[3];

        for (int j = 1; j < 4; j++) {
            curr[j * 4 + 0] = prev[j * 4 + 0] ^ curr[(j - 1) * 4 + 0];
            curr[j * 4 + 1] = prev[j * 4 + 1] ^ curr[(j - 1) * 4 + 1];
            curr[j * 4 + 2] = prev[j * 4 + 2] ^ curr[(j - 1) * 4 + 2];
            curr[j * 4 + 3] = prev[j * 4 + 3] ^ curr[(j - 1) * 4 + 3];
        }
    }
}

// AES-128 encrypt block
__device__ void aes128_encrypt(const uint8_t key[16], const uint8_t plaintext[16], uint8_t ciphertext[16]) {
    uint8_t expanded[176];
    aes_key_expand(key, expanded);

    uint8_t state[16];
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        state[i] = plaintext[i];
    }

    // Initial round
    aes_add_round_key(state, &expanded[0]);

    // Main rounds (1-9)
    for (int round = 1; round < 10; round++) {
        aes_sub_bytes(state);
        aes_shift_rows(state);
        aes_mix_columns(state);
        aes_add_round_key(state, &expanded[round * 16]);
    }

    // Final round (no MixColumns)
    aes_sub_bytes(state);
    aes_shift_rows(state);
    aes_add_round_key(state, &expanded[160]);

    #pragma unroll
    for (int i = 0; i < 16; i++) {
        ciphertext[i] = state[i];
    }
}

// ============================================================================
// SwapOrNot PRP (simplified for GPU)
// ============================================================================

// Derive round key K_i for SwapOrNot
__device__ uint64_t sn_derive_round_key(const uint8_t key[16], uint32_t round, uint64_t domain) {
    uint8_t input[16] = {0};
    // round as big-endian u64 in bytes 0..8
    for (int i = 0; i < 8; i++) {
        input[i] = (round >> ((7 - i) * 8)) & 0xFF;
    }
    // domain as big-endian u64 in bytes 8..16
    for (int i = 0; i < 8; i++) {
        input[8 + i] = (domain >> ((7 - i) * 8)) & 0xFF;
    }

    uint8_t output[16];
    aes128_encrypt(key, input, output);

    uint64_t result = 0;
    for (int i = 0; i < 8; i++) {
        result = (result << 8) | output[i];
    }
    return result % domain;
}

// PRF bit for SwapOrNot
__device__ bool sn_prf_bit(const uint8_t key[16], uint32_t round, uint64_t canonical) {
    uint8_t input[16] = {0};
    uint64_t tagged_round = round | 0x8000000000000000ULL;
    for (int i = 0; i < 8; i++) {
        input[i] = (tagged_round >> ((7 - i) * 8)) & 0xFF;
    }
    for (int i = 0; i < 8; i++) {
        input[8 + i] = (canonical >> ((7 - i) * 8)) & 0xFF;
    }

    uint8_t output[16];
    aes128_encrypt(key, input, output);
    return (output[0] & 1) == 1;
}

// One round of SwapOrNot (involutory)
__device__ uint64_t sn_round(const uint8_t key[16], uint32_t round, uint64_t x, uint64_t domain) {
    uint64_t k_i = sn_derive_round_key(key, round, domain);
    uint64_t partner = (k_i + domain - (x % domain)) % domain;
    uint64_t canonical = (x > partner) ? x : partner;

    if (sn_prf_bit(key, round, canonical)) {
        return partner;
    }
    return x;
}

// SwapOrNot forward (encrypt)
__device__ uint64_t sn_forward(const uint8_t key[16], uint64_t x, uint64_t domain, uint32_t num_rounds) {
    uint64_t val = x;
    for (uint32_t r = 0; r < num_rounds; r++) {
        val = sn_round(key, r, val, domain);
    }
    return val;
}

// SwapOrNot inverse (decrypt) - involutory, so same as forward with reversed rounds
__device__ uint64_t sn_inverse(const uint8_t key[16], uint64_t y, uint64_t domain, uint32_t num_rounds) {
    uint64_t val = y;
    for (int r = num_rounds - 1; r >= 0; r--) {
        val = sn_round(key, r, val, domain);
    }
    return val;
}

// ============================================================================
// Simplified iPRF inverse (for GPU prototype)
// ============================================================================

// Simplified iPRF.inverse: finds all x such that iPRF(x) = y
// For the prototype, we use a simplified model where iPRF is just SwapOrNot PRP
// (full implementation would include PMNS binomial layer)
__device__ uint32_t iprf_inverse_simple(
    const uint8_t key[16],
    uint64_t y,
    uint64_t domain,
    uint32_t num_rounds,
    uint64_t* preimages,
    uint32_t max_preimages
) {
    // For pure SwapOrNot PRP, there's exactly one preimage
    // Full iPRF with PMNS would have multiple preimages
    if (max_preimages == 0) return 0;

    preimages[0] = sn_inverse(key, y, domain, num_rounds);
    return 1;
}

// ============================================================================
// Hint Generation Kernel
// ============================================================================

/**
 * Kernel: Compute one hint parity
 *
 * Each thread computes the parity for one hint by:
 * 1. Determining which blocks are in this hint's subset
 * 2. For each block, finding entries via iPRF.inverse()
 * 3. XORing contributing entries into parity
 *
 * @param params       Plinko parameters
 * @param block_keys   iPRF key for each block (c keys total)
 * @param entries      Database entries (N × 48 bytes)
 * @param hint_subsets Precomputed block subsets for each hint
 * @param output       Output hint parities (total_hints × 32 bytes)
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

    // Calculate number of rounds for SwapOrNot
    uint32_t num_rounds = 6 * 32 + 6;  // Simplified: use fixed round count

    // Iterate over all blocks
    for (uint64_t block_idx = 0; block_idx < params.set_size; block_idx++) {
        // Check if block is in this hint's subset
        uint32_t byte_idx = hint_idx * ((params.set_size + 7) / 8) + (block_idx / 8);
        uint8_t bit_mask = 1 << (block_idx % 8);

        if ((hint_subsets[byte_idx] & bit_mask) == 0) continue;

        // Get block's iPRF key
        const uint8_t* key = block_keys[block_idx].key;

        // Find preimages for hint_idx in this block
        uint64_t preimages[MAX_PREIMAGES];
        uint32_t num_preimages = iprf_inverse_simple(
            key, hint_idx, params.chunk_size, num_rounds, preimages, MAX_PREIMAGES
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
 * Kernel: Parallel hint generation with shared memory optimization
 *
 * Uses tiles to amortize block subset lookups across warps.
 */
extern "C" __global__ void hint_gen_kernel_tiled(
    const PlinkoParams params,
    const IprfBlockKey* __restrict__ block_keys,
    const uint8_t* __restrict__ entries,
    const uint8_t* __restrict__ hint_subsets,
    HintOutput* __restrict__ output
) {
    // Shared memory for block keys (one tile at a time)
    __shared__ IprfBlockKey s_block_keys[32];  // Cache 32 block keys

    uint32_t hint_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (hint_idx >= params.total_hints) return;

    uint64_t parity[4] = {0, 0, 0, 0};
    uint32_t num_rounds = 6 * 32 + 6;

    // Process blocks in tiles
    for (uint64_t tile_start = 0; tile_start < params.set_size; tile_start += 32) {
        // Load block keys into shared memory
        if (threadIdx.x < 32 && tile_start + threadIdx.x < params.set_size) {
            s_block_keys[threadIdx.x] = block_keys[tile_start + threadIdx.x];
        }
        __syncthreads();

        // Process blocks in this tile
        uint32_t tile_end = min((uint64_t)(tile_start + 32), params.set_size);
        for (uint64_t block_idx = tile_start; block_idx < tile_end; block_idx++) {
            // Check subset membership
            uint32_t byte_idx = hint_idx * ((params.set_size + 7) / 8) + (block_idx / 8);
            uint8_t bit_mask = 1 << (block_idx % 8);
            if ((hint_subsets[byte_idx] & bit_mask) == 0) continue;

            const uint8_t* key = s_block_keys[block_idx - tile_start].key;

            uint64_t preimages[MAX_PREIMAGES];
            uint32_t num_preimages = iprf_inverse_simple(
                key, hint_idx, params.chunk_size, num_rounds, preimages, MAX_PREIMAGES
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
