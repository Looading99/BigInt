#pragma once

#include <array>
#include <bit>
#include <cstdint>
#include <span>
#include <vector>


#include "bigint/bigint_base.h"
#include "ntt_base.h"


namespace bigint::mul {


namespace brute {

constexpr std::size_t MAX_TOTAL_BITS = 2 * 128 * 64;

inline auto mul(std::span<const uint64_t> A, std::span<const uint64_t> B) -> std::vector<uint64_t> {
    std::vector<uint64_t> res;
    const std::size_t     n = A.size(), m = B.size();
    res.resize(n + m);
    for (std::size_t i = 0; i < n; ++i) {
        uint64_t    carry = 0;
        std::size_t j     = 0;
        for (; j < m; ++j) {
            auto t = static_cast<uint128_t>(A[i]) * B[j] + res[i + j] + carry;
            res[i + j] = t;
            carry      = t >> 64;
        }
        for (; carry; ++j) {
            auto t = static_cast<uint128_t>(res[i + j]) + carry;
            res[i + j] = t;
            carry      = t >> 64;
        }
    }
    detail::remove_leading_zero(res);
    return res;
}

}  // namespace brute


namespace fft {

// 动态 digit_bits：根据输入总比特数在 [MIN_DIGIT_BITS, MAX_DIGIT_BITS] 之间选择。
// digit_bits 越大，同样比特数的输入所需 FFT 点数越少、越快，但可保证精度的规模越小。
// 可用编译宏 BIGINT_FFT_MIN_DIGIT_BITS 覆盖调参。
#ifndef BIGINT_FFT_MIN_DIGIT_BITS
#    define BIGINT_FFT_MIN_DIGIT_BITS 5
#endif
constexpr int MIN_DIGIT_BITS = BIGINT_FFT_MIN_DIGIT_BITS;
constexpr int MAX_DIGIT_BITS = 20;

// 每个 digit_bits 能保证精度的最大输入总比特数（最坏情况输入实测，见
// tests/measure_digit_bits.cpp）。 下标 = digit_bits，未使用下标为 0。
// 2026-08-27：fft-rewrite 分支 dif_fft_mixed/dit_ifft_mixed + R2 双锚点后重测 B4-20
// （默认 MIN=5，FFT 覆盖 ~84M bits；B4 极限在变换 2^26、B5 在 2^24，低于默认 MIN 的
// 值仅供编译时降低 MIN 实验用）。
constexpr std::array<std::size_t, MAX_DIGIT_BITS + 1> MAX_TOTAL_BITS_FOR_DIGIT_BITS = {
    0,          // digit_bits=0 （未使用）
    0,          // digit_bits=1 （未使用）
    0,          // digit_bits=2 （未使用）
    0,          // digit_bits=3 （未使用）
    209669000,  // digit_bits=4
    83886080,   // digit_bits=5
    50331648,   // digit_bits=6
    58720256,   // digit_bits=7
    44911776,   // digit_bits=8
    9437184,    // digit_bits=9
    10485760,   // digit_bits=10
    2883584,    // digit_bits=11
    1572864,    // digit_bits=12
    851968,     // digit_bits=13
    838572,     // digit_bits=14
    491520,     // digit_bits=15
    262144,     // digit_bits=16
    139264,     // digit_bits=17
    36864,      // digit_bits=18
    19456,      // digit_bits=19
    20480,      // digit_bits=20
};
constexpr std::size_t MAX_TOTAL_BITS = MAX_TOTAL_BITS_FOR_DIGIT_BITS[MIN_DIGIT_BITS];

// 可选编译宏 BIGINT_FFT_MAX_FFT_LAYER：强制指定最大层数（基-2，0=由 MAX_TOTAL_BITS 推导）。
// 用于精度测量 B < MIN 的容量（B 越小容量越大、所需层数越深）或限制内存上限。
#ifndef BIGINT_FFT_MAX_FFT_LAYER
#    define BIGINT_FFT_MAX_FFT_LAYER 0
#endif

// 最大变换长度/层数由 MAX_TOTAL_BITS 推导（不写死，基-2）：
// 任意输入总比特数 T ≤ MAX_TOTAL_BITS 都会被路由到 FFT，自动选择的 digit_bits ≥ MIN_DIGIT_BITS，
// 数字个数 dA+dB-1 ≤ ceil(T/MIN_DIGIT_BITS) ≤ ceil(MAX_TOTAL_BITS/MIN_DIGIT_BITS)，
// 故所需变换长度不超过 bit_ceil(ceil(MAX_TOTAL_BITS/MIN_DIGIT_BITS))，据此反推层数。
// （新 FFT 为基-2 混合 radix：长度可为任意 2 的幂，含 2*4^k，故用 bit_ceil 而非 pow4ceil。）
constexpr std::size_t MAX_FFT_LEN   = BIGINT_FFT_MAX_FFT_LAYER > 0
                                          ? (std::size_t(1) << BIGINT_FFT_MAX_FFT_LAYER)
                                          : std::bit_ceil((MAX_TOTAL_BITS + MIN_DIGIT_BITS - 1)
                                                          / MIN_DIGIT_BITS);
constexpr std::size_t MAX_FFT_LAYER = std::countr_zero(MAX_FFT_LEN);

// 根据输入总比特数查表，返回能保证精度的最大 digit_bits
inline auto digit_bits_for_total_bits(std::size_t total_bits) -> int {
    for (int b = MAX_DIGIT_BITS; b >= MIN_DIGIT_BITS; --b) {
        if (total_bits <= MAX_TOTAL_BITS_FOR_DIGIT_BITS[b]) {
            return b;
        }
    }
    unreachable();
    return MIN_DIGIT_BITS;  // 不可达，仅为满足返回值路径检查
}

// digit_bits 传入 [MIN, MAX] 时强制使用该值（供精度测量测试用），0 = 自动选择
auto mul(std::span<const uint64_t> A, std::span<const uint64_t> B, int digit_bits = 0)
    -> std::vector<uint64_t>;

}  // namespace fft


namespace ntt {

constexpr std::size_t MAX_TOTAL_BITS = (MAX_TRANSFORM_LEN - 2) * NTT_DIGIT_BITS;

auto mul(std::span<const uint64_t> A, std::span<const uint64_t> B) -> std::vector<uint64_t>;

}  // namespace ntt


namespace ssa {

constexpr std::size_t MIN_TOTAL_BITS = ntt::MAX_TOTAL_BITS + 1;

auto mul(std::span<const uint64_t> A, std::span<const uint64_t> B) -> std::vector<uint64_t>;

}  // namespace ssa


inline auto _mul(std::span<const uint64_t> A, std::span<const uint64_t> B)
    -> std::vector<uint64_t> {
    std::size_t total_bits = (A.size() + B.size()) * 64;
    if (total_bits <= brute::MAX_TOTAL_BITS) {
        return brute::mul(A, B);
    } else if (total_bits <= fft::MAX_TOTAL_BITS) {
        return fft::mul(A, B);
    } else if (total_bits <= ntt::MAX_TOTAL_BITS) {
        return ntt::mul(A, B);
    } else {
        return ssa::mul(A, B);
    }
}

// 将两个数组（按照 output_precision ）相乘。必须保证输入数组非全0。
// 内部表示与乘法分发均为 64 位 limb，无任何打包/拆包转换。
inline auto mul_digits(const Digits& a, const Digits& b, std::size_t output_precision = 0,
    int64_t* p_result_point_pos = nullptr) -> Digits {
    bool a_is_b = &a == &b;

    std::size_t offset_a =
        output_precision == 0 || a.size() <= output_precision ? 0 : a.size() - output_precision;
    std::size_t offset_b =
        output_precision == 0 || b.size() <= output_precision ? 0 : b.size() - output_precision;
    std::size_t tail_zero_a = 0, tail_zero_b = 0;
    while (a[offset_a + tail_zero_a] == 0) {
        ++tail_zero_a;
    }
    if (a_is_b) {
        tail_zero_b = tail_zero_a;
    } else {
        while (b[offset_b + tail_zero_b] == 0) {
            ++tail_zero_b;
        }
    }
    offset_a += tail_zero_a;
    offset_b += tail_zero_b;

    auto A = std::span<const uint64_t>(a).subspan(offset_a);

    auto C = _mul(A, a_is_b ? A : std::span<const uint64_t>(b).subspan(offset_b));

    if (output_precision) {
        auto offset_res = C.size() <= output_precision ? 0 : C.size() - output_precision;
        while (C[offset_res] == 0) {
            ++offset_res;
        }
        if (offset_res) {
            C.erase(C.begin(), C.begin() + static_cast<int64_t>(offset_res));
        }
        if (p_result_point_pos) {
            *p_result_point_pos -= static_cast<int64_t>(offset_a + offset_b + offset_res);
        }
    } else {
        C.insert(C.begin(), tail_zero_a + tail_zero_b, 0);
    }

    return C;
}

}  // namespace bigint::mul
