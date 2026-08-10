#pragma once

#include <array>
#include <cstdint>
#include <immintrin.h>
#include <ranges>


#include "bigint/bigint_base.h"


namespace bigint::ntt {

constexpr uint8_t                          NUM_PRIMES = 3;
constexpr std::array<uint32_t, NUM_PRIMES> P          = {469762049, 2013265921, 2281701377};
static_assert(DIGIT_MASK < std::ranges::min(P));
constexpr std::array<uint32_t, NUM_PRIMES> G = {3, 31, 3};

constexpr uint32_t MAX_TRANSFORM_LEN_EXP =
    std::ranges::min(P | std::ranges::views::transform([](uint32_t p) -> uint32_t {
        return std::countr_zero(p - 1);
    }));

constexpr uint32_t MAX_TRANSFORM_LEN = 1u << MAX_TRANSFORM_LEN_EXP;

constexpr auto inv_G = [] {
    std::array<uint32_t, NUM_PRIMES> res{};
    for (uint8_t I = 0; I < NUM_PRIMES; ++I) {
        res[I] = modinv(G[I], P[I]);
    }
    return res;
}();

// Montgomery 模乘
namespace mont {

constexpr uint32_t w = 32;
constexpr uint64_t R = 1ull << w;

// -P^(-1) mod R
constexpr auto neg_inv_P = [] {
    std::array<uint32_t, NUM_PRIMES> res{};
    for (uint8_t I = 0; I < NUM_PRIMES; ++I) {
        int64_t inv_P = modinv(P[I], R);
        res[I]        = R - inv_P;
    }
    return res;
}();

// R^2 mod P
constexpr auto R_sq = [] {
    std::array<uint32_t, NUM_PRIMES> res{};
    for (uint8_t I = 0; I < NUM_PRIMES; ++I) {
        uint64_t R_mod_P = R % P[I];
        res[I]           = R_mod_P * R_mod_P % P[I];
    }
    return res;
}();

constexpr std::array<uint32_t, 3> vec_one = {R % P[0], R % P[1], R % P[2]};

constexpr auto mul(uint8_t I, uint32_t a, uint32_t b) -> uint32_t {
    uint64_t t  = uint64_t(a) * b;
    uint32_t u  = static_cast<uint32_t>(t) * neg_inv_P[I];
    uint64_t t2 = t + static_cast<uint64_t>(u) * P[I];
    uint32_t r  = t2 >> w;
    return r < P[I] ? r : r - P[I];
}

constexpr auto vec_mul(const std::array<uint32_t, NUM_PRIMES>& a,
    const std::array<uint32_t, NUM_PRIMES>& b) -> std::array<uint32_t, NUM_PRIMES> {
    return {mul(0, a[0], b[0]), mul(1, a[1], b[1]), mul(2, a[2], b[2])};
}

constexpr void vec_to_mont(std::array<uint32_t, NUM_PRIMES>& v) {
    v = vec_mul(v, R_sq);
}

constexpr auto fast_pow(std::array<uint32_t, NUM_PRIMES> base, uint32_t exponent)
    -> std::array<uint32_t, NUM_PRIMES> {
    std::array<uint32_t, NUM_PRIMES> res = vec_one;
    while (exponent) {
        if (exponent & 1)
            res = vec_mul(res, base);
        base = vec_mul(base, base);
        exponent >>= 1;
    }
    return res;
}

inline auto vec_to_simd(const std::array<uint32_t, NUM_PRIMES>& v) -> __m128i {
    return _mm_set_epi32(0, static_cast<int>(v[2]), static_cast<int>(v[1]), static_cast<int>(v[0]));
}

inline void simd_store3(uint32_t* dst, const __m128i& src) {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    _mm_storel_epi64(reinterpret_cast<__m128i*>(dst), src);
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    dst[2] = static_cast<uint32_t>(_mm_extract_epi32(src, 2));
}

const __m128i simd_one = vec_to_simd(vec_one);

inline auto simd_mul(const __m128i& a, const __m128i& b) -> __m128i {
    // 原始模数
    const __m256i Po = _mm256_set_epi32(0, 0, 0, static_cast<int>(P[2]), 0, P[1], 0, P[0]);
    // 模数左移 32 位
    const __m256i Ps = _mm256_slli_epi64(Po, w);
    // 符号位，XOR 后用有符号 cmpgt 即等效无符号比较
    const __m256i sign = _mm256_set1_epi64x(1LL << 63);
    // neg_inv_P
    const __m256i Pi =
        _mm256_set_epi32(0, 0, 0, static_cast<int>(neg_inv_P[2]), 0, neg_inv_P[1], 0, neg_inv_P[0]);
    // 重排索引
    const __m256i idx = _mm256_set_epi32(7, 6, 5, 4, 7, 5, 3, 1);

    __m256i A = _mm256_cvtepu32_epi64(a), B = _mm256_cvtepu32_epi64(b);

    __m256i T = _mm256_mul_epu32(A, B);

    __m256i U = _mm256_mul_epu32(T, Pi);

    U            = _mm256_mul_epu32(U, Po);
    T            = _mm256_add_epi64(T, U);
    __m256i mask = _mm256_cmpgt_epi64(_mm256_xor_si256(Ps, sign), _mm256_xor_si256(T, sign));
    T            = _mm256_sub_epi64(T, _mm256_andnot_si256(mask, Ps));
    __m256i perm = _mm256_permutevar8x32_epi32(T, idx);
    return _mm256_castsi256_si128(perm);
}

inline auto simd_add(const __m128i& a, const __m128i& b) -> __m128i {
    const __m128i p    = _mm_set_epi32(0, static_cast<int>(P[2]), P[1], P[0]);
    const __m128i sign = _mm_set1_epi32(1 << 31);
    __m128i       d    = _mm_sub_epi32(p, b);
    __m128i       t    = _mm_add_epi32(a, b);
    __m128i       mask = _mm_cmplt_epi32(_mm_xor_si128(a, sign), _mm_xor_si128(d, sign));
    return _mm_sub_epi32(t, _mm_andnot_si128(mask, p));
}

inline auto simd_sub(const __m128i& a, const __m128i& b) -> __m128i {
    const __m128i p    = _mm_set_epi32(0, static_cast<int>(P[2]), P[1], P[0]);
    const __m128i sign = _mm_set1_epi32(1 << 31);
    __m128i       t    = _mm_sub_epi32(a, b);
    __m128i       mask = _mm_cmplt_epi32(_mm_xor_si128(a, sign), _mm_xor_si128(b, sign));
    return _mm_add_epi32(t, _mm_and_si128(mask, p));
}

inline auto simd_add_and_sub(const __m128i& a, const __m128i& b) -> __m256i {
    const __m256i mask2 = _mm256_set_epi32(0, 0, 0, 0, -1, -1, -1, -1);
    const __m256i sign  = _mm256_set1_epi32(1 << 31);
    // (p, -p)
    const __m256i Pv = _mm256_set_epi32(0,
        static_cast<int>(-P[2]),
        static_cast<int>(-P[1]),
        static_cast<int>(-P[0]),
        0,
        static_cast<int>(P[2]),
        static_cast<int>(P[1]),
        static_cast<int>(P[0]));
    // (a, a)
    __m256i A = _mm256_broadcastsi128_si256(a);
    // -b
    __m128i neg_b_128 = _mm_sub_epi32(_mm_setzero_si128(), b);
    // (-b, b)
    __m256i B = _mm256_set_m128i(b, neg_b_128);
    // (a+b, a-b)
    __m256i T = _mm256_sub_epi32(A, B);
    // (p-b, b)
    __m256i C = _mm256_add_epi32(_mm256_and_si256(Pv, mask2), B);
    // (a<p-b, a<b)
    __m256i mask = _mm256_cmpgt_epi32(_mm256_xor_si256(C, sign), _mm256_xor_si256(A, sign));
    // (a>=p-b, a<b)
    mask = _mm256_xor_si256(mask, mask2);
    // (a>=p-b?a+b-p:a+b, a<b?a-b+p:a-b)
    __m256i result256 = _mm256_sub_epi32(T, _mm256_and_si256(mask, Pv));
    return result256;
}

inline auto fast_pow(__m128i base, uint32_t exponent) -> __m128i {
    __m128i res = simd_one;
    while (exponent) {
        if (exponent & 1)
            res = simd_mul(res, base);
        base = simd_mul(base, base);
        exponent >>= 1;
    }
    return res;
}


}  // namespace mont

constexpr auto steps = [] {
    std::array<std::array<std::array<uint32_t, NUM_PRIMES>, MAX_TRANSFORM_LEN_EXP>, 2> arr{};
    for (uint8_t I = 0; I < NUM_PRIMES; ++I) {
        for (size_type level = 1; level <= MAX_TRANSFORM_LEN_EXP; ++level) {
            arr[0][level - 1][I] = fast_pow(G[I], (P[I] - 1) >> level, P[I]);
            arr[1][level - 1][I] = fast_pow(inv_G[I], (P[I] - 1) >> level, P[I]);
        }
    }
    for (size_type level = 1; level <= MAX_TRANSFORM_LEN_EXP; ++level) {
        mont::vec_to_mont(arr[0][level - 1]);
        mont::vec_to_mont(arr[1][level - 1]);
    }
    return arr;
}();

constexpr auto inv_pow_of_two = [] {
    std::array<std::array<uint32_t, NUM_PRIMES>, MAX_TRANSFORM_LEN_EXP> arr{};
    for (uint8_t I = 0; I < NUM_PRIMES; ++I) {
        arr[0][I] = modinv(2, P[I]);
        for (size_type i = 1; i < MAX_TRANSFORM_LEN_EXP; ++i) {
            arr[i][I] = static_cast<uint64_t>(arr[i - 1][I]) * arr[0][I] % P[I];
        }
    }
    for (auto& vec : arr) {
        mont::vec_to_mont(vec);
    }
    return arr;
}();

constexpr auto is_invalid_ntt_len(size_type n) -> bool {
    return n == 0 || (n & (n - 1)) != 0 || n > MAX_TRANSFORM_LEN;
}

constexpr auto garner_merge(uint32_t c0, uint32_t c1, uint32_t c2) -> uint128_t {
    constexpr uint32_t inv_P0_mod_P1 = modinv(P[0], P[1]);
    constexpr uint32_t inv_P0_mod_P2 = modinv(P[0], P[2]);
    constexpr uint32_t inv_P1_mod_P2 = modinv(P[1], P[2]);

    uint32_t v0 = c0;

    uint32_t v1 = static_cast<uint64_t>(sub_mod(c1, v0, P[1])) * inv_P0_mod_P1 % P[1];

    uint32_t v2 = static_cast<uint64_t>(sub_mod(c2, v0, P[2])) * inv_P0_mod_P2 % P[2];
    v2          = static_cast<uint64_t>(sub_mod(v2, v1, P[2])) * inv_P1_mod_P2 % P[2];

    uint128_t x = static_cast<uint64_t>(P[0]) * P[1];
    x *= v2;
    x += static_cast<uint64_t>(P[0]) * v1;
    x += v0;

    return x;
}


// 位逆序置换
inline void bit_swap(Digits& v) {
    const size_type n = v.size();
    if (is_invalid_ntt_len(n)) {
        unreachable();
    }
    for (size_type i = 1, j = n >> 1; i < n - 1; ++i) {
        if (i < j)
            std::swap(v[i], v[j]);
        size_type k = n >> 1;
        while (j >= k) {
            j -= k;
            k >>= 1;
        }
        j += k;
    }
}

// 多重位逆序置换，对每个下标模 NUM_PRIMES 同余类分别做位逆序置换
inline void multi_bit_swap(Digits& v) {
    size_type n = v.size();
    if (n % NUM_PRIMES != 0 || is_invalid_ntt_len(n / NUM_PRIMES)) {
        unreachable();
    }
    n /= NUM_PRIMES;
    for (size_type i = 1, j = n >> 1; i < n - 1; ++i) {
        if (i < j) {
            for (uint8_t u = 0; u < NUM_PRIMES; ++u) {
                std::swap(v[NUM_PRIMES * i + u], v[NUM_PRIMES * j + u]);
            }
        }
        size_type k = n >> 1;
        while (j >= k) {
            j -= k;
            k >>= 1;
        }
        j += k;
    }
}

// 把数组每个元素重复 k 次
inline void copy_repeat(const Digits& v_in, size_type offset, size_type k, Digits& v_out) {
    v_out.clear();
    size_type new_size = (v_in.size() - offset) * k;
    v_out.reserve(new_size);
    if (new_size == 0) {
        return;
    } else if (k == 1) {
        v_out.insert(v_out.begin(), v_in.begin() + static_cast<int64_t>(offset), v_in.end());
        return;
    }
    for (size_type i = offset; i < v_in.size(); ++i) {
        uint32_t value = v_in[i];
        for (size_type rep = 0; rep < k; ++rep) {
            v_out.push_back(value);
        }
    }
}

}  // namespace bigint::ntt
