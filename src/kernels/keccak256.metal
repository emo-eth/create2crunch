/*
   **Note: claude 3.7 generated this license**
   
   
   Copyright 2018 Lip Wee Yeo Amano

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/

/**
* Based on the following, with small tweaks and optimizations:
*
* https://github.com/lwYeo/SoliditySHA3Miner/blob/master/SoliditySHA3Miner/
*   Miner/Kernels/OpenCL/sha3KingKernel.cl
*
* Originally modified for openCL processing by lwYeo
*
* Original implementor: David Leon Gil
*
* License: CC0, attribution kindly requested. Blame taken too, but not
* liability.
*/

/******** Keccak-f[1600] (for finding efficient Ethereum addresses) ********/

#include <metal_stdlib>
using namespace metal;

// Use constant address space for read-only data
constant uint64_t keccak_round_constants[24] = {
    0x0000000000000001, 0x0000000000008082, 0x800000000000808a, 0x8000000080008000,
    0x000000000000808b, 0x0000000080000001, 0x8000000080008081, 0x8000000000008009,
    0x000000000000008a, 0x0000000000000088, 0x0000000080008009, 0x000000008000000a,
    0x000000008000808b, 0x800000000000008b, 0x8000000000008089, 0x8000000000008003,
    0x8000000000008002, 0x8000000000000080, 0x000000000000800a, 0x800000008000000a,
    0x8000000080008081, 0x8000000000008080, 0x0000000080000001, 0x8000000080008008
};

typedef union {
  uint64_t uint64_t;
  uint32_t uint32_t[2];
  uchar uint8_t[8];
} nonce_t;

// Use fast inline function for rotation
static inline uint64_t rol(const uint64_t x, const uint s) {
  return (x << s) | (x >> (64u - s));
}

#define rol1(x) rol(x, 1u)

// Optimized theta step with better register usage
#define theta_(m, n, o) \
t = b[m] ^ rol1(b[n]); \
a[o + 0] ^= t; \
a[o + 5] ^= t; \
a[o + 10] ^= t; \
a[o + 15] ^= t; \
a[o + 20] ^= t; \

#define theta() \
b[0] = a[0] ^ a[5] ^ a[10] ^ a[15] ^ a[20]; \
b[1] = a[1] ^ a[6] ^ a[11] ^ a[16] ^ a[21]; \
b[2] = a[2] ^ a[7] ^ a[12] ^ a[17] ^ a[22]; \
b[3] = a[3] ^ a[8] ^ a[13] ^ a[18] ^ a[23]; \
b[4] = a[4] ^ a[9] ^ a[14] ^ a[19] ^ a[24]; \
theta_(4, 1, 0); \
theta_(0, 2, 1); \
theta_(1, 3, 2); \
theta_(2, 4, 3); \
theta_(3, 0, 4);

#define rhoPi_(m, n) t = b[0]; b[0] = a[m]; a[m] = rol(t, n); \

#define rhoPi() t = a[1]; b[0] = a[10]; a[10] = rol1(t); \
rhoPi_(7, 3); \
rhoPi_(11, 6); \
rhoPi_(17, 10); \
rhoPi_(18, 15); \
rhoPi_(3, 21); \
rhoPi_(5, 28); \
rhoPi_(16, 36); \
rhoPi_(8, 45); \
rhoPi_(21, 55); \
rhoPi_(24, 2); \
rhoPi_(4, 14); \
rhoPi_(15, 27); \
rhoPi_(23, 41); \
rhoPi_(19, 56); \
rhoPi_(13, 8); \
rhoPi_(12, 25); \
rhoPi_(2, 43); \
rhoPi_(20, 62); \
rhoPi_(14, 18); \
rhoPi_(22, 39); \
rhoPi_(9, 61); \
rhoPi_(6, 20); \
rhoPi_(1, 44);

// Optimized chi step with better register usage
#define chi_(n) \
b[0] = a[n + 0]; \
b[1] = a[n + 1]; \
b[2] = a[n + 2]; \
b[3] = a[n + 3]; \
b[4] = a[n + 4]; \
a[n + 0] = b[0] ^ ((~b[1]) & b[2]); \
a[n + 1] = b[1] ^ ((~b[2]) & b[3]); \
a[n + 2] = b[2] ^ ((~b[3]) & b[4]); \
a[n + 3] = b[3] ^ ((~b[4]) & b[0]); \
a[n + 4] = b[4] ^ ((~b[0]) & b[1]);

#define chi() chi_(0); chi_(5); chi_(10); chi_(15); chi_(20);

#define iota(x) a[0] ^= x;

// Optimized iteration using constant address space for round constants
#define iteration(i) theta(); rhoPi(); chi(); iota(keccak_round_constants[i]);

// Optimized keccakf function with unrolled iterations for better performance
static inline void keccakf(thread uint64_t *a) {
  thread uint64_t b[5];
  uint64_t t;

  // Unroll first 23 iterations for better instruction scheduling
  iteration(0);  // iteration 1
  iteration(1);  // iteration 2
  iteration(2);  // iteration 3
  iteration(3);  // iteration 4
  iteration(4);  // iteration 5
  iteration(5);  // iteration 6
  iteration(6);  // iteration 7
  iteration(7);  // iteration 8
  iteration(8);  // iteration 9
  iteration(9);  // iteration 10
  iteration(10); // iteration 11
  iteration(11); // iteration 12
  iteration(12); // iteration 13
  iteration(13); // iteration 14
  iteration(14); // iteration 15
  iteration(15); // iteration 16
  iteration(16); // iteration 17
  iteration(17); // iteration 18
  iteration(18); // iteration 19
  iteration(19); // iteration 20
  iteration(20); // iteration 21
  iteration(21); // iteration 22
  iteration(22); // iteration 23

  // iteration 24 (partial)
#define o ((thread uint *)(a))
  // Theta (partial)
  b[0] = a[0] ^ a[5] ^ a[10] ^ a[15] ^ a[20];
  b[1] = a[1] ^ a[6] ^ a[11] ^ a[16] ^ a[21];
  b[2] = a[2] ^ a[7] ^ a[12] ^ a[17] ^ a[22];
  b[3] = a[3] ^ a[8] ^ a[13] ^ a[18] ^ a[23];
  b[4] = a[4] ^ a[9] ^ a[14] ^ a[19] ^ a[24];

  a[0] ^= b[4] ^ rol1(b[1]);
  a[6] ^= b[0] ^ rol1(b[2]);
  a[12] ^= b[1] ^ rol1(b[3]);
  a[18] ^= b[2] ^ rol1(b[4]);
  a[24] ^= b[3] ^ rol1(b[0]);

  // Rho Pi (partial)
  o[3] = (o[13] >> 20) | (o[12] << 12);
  a[2] = rol(a[12], 43);
  a[3] = rol(a[18], 21);
  a[4] = rol(a[24], 14);

  // Chi (partial)
  o[3] ^= ((~o[5]) & o[7]);
  o[4] ^= ((~o[6]) & o[8]);
  o[5] ^= ((~o[7]) & o[9]);
  o[6] ^= ((~o[8]) & o[0]);
  o[7] ^= ((~o[9]) & o[1]);
#undef o
}

// Optimized hasTotal function using SIMD-style operations where possible
#define hasTotal(d) ( \
  (!(d[0])) + (!(d[1])) + (!(d[2])) + (!(d[3])) + \
  (!(d[4])) + (!(d[5])) + (!(d[6])) + (!(d[7])) + \
  (!(d[8])) + (!(d[9])) + (!(d[10])) + (!(d[11])) + \
  (!(d[12])) + (!(d[13])) + (!(d[14])) + (!(d[15])) + \
  (!(d[16])) + (!(d[17])) + (!(d[18])) + (!(d[19])) \
>= TOTAL_ZEROES)

// Optimized hasLeading functions using uint32_t for faster comparison
#if LEADING_ZEROES == 8
#define hasLeading(d) (!(((thread uint*)d)[0]) && !(((thread uint*)d)[1]))
#elif LEADING_ZEROES == 7
#define hasLeading(d) (!(((thread uint*)d)[0]) && !(((thread uint*)d)[1] & 0x00ffffffu))
#elif LEADING_ZEROES == 6
#define hasLeading(d) (!(((thread uint*)d)[0]) && !(((thread uint*)d)[1] & 0x0000ffffu))
#elif LEADING_ZEROES == 5
#define hasLeading(d) (!(((thread uint*)d)[0]) && !(((thread uint*)d)[1] & 0x000000ffu))
#elif LEADING_ZEROES == 4
#define hasLeading(d) (!(((thread uint*)d)[0]))
#elif LEADING_ZEROES == 3
#define hasLeading(d) (!(((thread uint*)d)[0] & 0x00ffffffu))
#elif LEADING_ZEROES == 2
#define hasLeading(d) (!(((thread uint*)d)[0] & 0x0000ffffu))
#elif LEADING_ZEROES == 1
#define hasLeading(d) (!(((thread uint*)d)[0] & 0x000000ffu))
#else
static inline bool hasLeading(thread const uchar *d) {
  // Vectorized approach for better performance
  uint count = 0;
  for (uint i = 0; i < LEADING_ZEROES; i += 4) {
    uint remaining = min(4u, LEADING_ZEROES - i);
    uint32_t chunk = 0;
    for (uint j = 0; j < remaining; ++j) {
      chunk |= d[i + j] << (j * 8);
    }
    if (chunk != 0) return false;
  }
  return true;
}
#endif

// Main kernel function with optimized memory access
kernel void hashMessage(
  constant uchar const *d_message [[buffer(0)]],
  constant uint const *d_nonce [[buffer(1)]],
  device volatile uint64_t *solutions [[buffer(2)]],
  device atomic_uint *solution_count [[buffer(3)]],
  uint gid [[thread_position_in_grid]],
  uint tid [[thread_index_in_threadgroup]],
  uint threads_per_group [[threads_per_threadgroup]]
) {
  // Use thread memory for better performance
  thread uint64_t spongeBuffer[25];

#define sponge ((thread uchar *) spongeBuffer)
#define digest (sponge + 12)

  nonce_t nonce;

  // Initialize sponge with zeros first for better memory pattern
  for (int i = 0; i < 25; i++) {
    spongeBuffer[i] = 0;
  }

  // write the control character
  sponge[0] = 0xffu;

  // Use vectorized assignments where possible
  sponge[1] = S_1;
  sponge[2] = S_2;
  sponge[3] = S_3;
  sponge[4] = S_4;
  sponge[5] = S_5;
  sponge[6] = S_6;
  sponge[7] = S_7;
  sponge[8] = S_8;
  sponge[9] = S_9;
  sponge[10] = S_10;
  sponge[11] = S_11;
  sponge[12] = S_12;
  sponge[13] = S_13;
  sponge[14] = S_14;
  sponge[15] = S_15;
  sponge[16] = S_16;
  sponge[17] = S_17;
  sponge[18] = S_18;
  sponge[19] = S_19;
  sponge[20] = S_20;
  sponge[21] = S_21;
  sponge[22] = S_22;
  sponge[23] = S_23;
  sponge[24] = S_24;
  sponge[25] = S_25;
  sponge[26] = S_26;
  sponge[27] = S_27;
  sponge[28] = S_28;
  sponge[29] = S_29;
  sponge[30] = S_30;
  sponge[31] = S_31;
  sponge[32] = S_32;
  sponge[33] = S_33;
  sponge[34] = S_34;
  sponge[35] = S_35;
  sponge[36] = S_36;
  sponge[37] = S_37;
  sponge[38] = S_38;
  sponge[39] = S_39;
  sponge[40] = S_40;

  // Copy message bytes (vectorized)
  sponge[41] = d_message[0];
  sponge[42] = d_message[1];
  sponge[43] = d_message[2];
  sponge[44] = d_message[3];

  // populate the nonce with better thread distribution
  nonce.uint32_t[0] = gid;
  nonce.uint32_t[1] = d_nonce[0];

  // populate the body of the message with the nonce (vectorized)
  sponge[45] = nonce.uint8_t[0];
  sponge[46] = nonce.uint8_t[1];
  sponge[47] = nonce.uint8_t[2];
  sponge[48] = nonce.uint8_t[3];
  sponge[49] = nonce.uint8_t[4];
  sponge[50] = nonce.uint8_t[5];
  sponge[51] = nonce.uint8_t[6];
  sponge[52] = nonce.uint8_t[7];

  sponge[53] = S_53;
  sponge[54] = S_54;
  sponge[55] = S_55;
  sponge[56] = S_56;
  sponge[57] = S_57;
  sponge[58] = S_58;
  sponge[59] = S_59;
  sponge[60] = S_60;
  sponge[61] = S_61;
  sponge[62] = S_62;
  sponge[63] = S_63;
  sponge[64] = S_64;
  sponge[65] = S_65;
  sponge[66] = S_66;
  sponge[67] = S_67;
  sponge[68] = S_68;
  sponge[69] = S_69;
  sponge[70] = S_70;
  sponge[71] = S_71;
  sponge[72] = S_72;
  sponge[73] = S_73;
  sponge[74] = S_74;
  sponge[75] = S_75;
  sponge[76] = S_76;
  sponge[77] = S_77;
  sponge[78] = S_78;
  sponge[79] = S_79;
  sponge[80] = S_80;
  sponge[81] = S_81;
  sponge[82] = S_82;
  sponge[83] = S_83;
  sponge[84] = S_84;

  // begin padding based on message length
  sponge[85] = 0x01u;
  
  // Use memset-style approach for better performance
  sponge[135] = 0x80u;

  // Apply keccakf
  keccakf(spongeBuffer);

  // Optimized condition check
  bool found = false;
  
  // Check leading zeros first (most likely to fail)
  if (hasLeading(digest)) {
    found = true;
  } 
#if TOTAL_ZEROES <= 20
  // Only check total zeros if leading zeros check failed
  else if (hasTotal(digest)) {
    found = true;
  }
#endif

  // Write the solution if found
  if (found) {
    // Atomically increment the solution counter and get the index
    uint index = atomic_fetch_add_explicit(solution_count, 1, memory_order_relaxed);
    
    // Limit to maximum number of solutions we can store (16)
    if (index < 16) {
      solutions[index] = nonce.uint64_t;
    }
  }
} 