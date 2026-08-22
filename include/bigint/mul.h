#pragma once

#include <array>
#include <bit>
#include <cstdint>
#include <span>
#include <vector>


#include "bigint/bigint_base.h"
#include "ntt_base.h"


namespace bigint::mul {


// 与乘法实现有关的辅助函数，暂存在此，将来可能迁移至别处。
namespace utils {

// 计算 C = A + B，A 和 B 长度小于 C 时自动补 0，返回最高位是否进位，C 可以是 A、B 之一
inline auto add(std::span<const uint64_t> A, std::span<const uint64_t> B, std::span<uint64_t> C)
    -> bool {
    bool carry = false;
    for (size_type i = 0; i < C.size(); ++i) {
        uint64_t a = i < A.size() ? A[i] : 0, b = i < B.size() ? B[i] : 0;
        if (a == uint64_t(-1) && carry) {
            C[i] = b;
        } else {
            carry = add_overflow(a + carry, b, C[i]);
        }
    }
    return carry;
}

// 计算 C = A - B，A 和 B 长度小于 C 时自动补 0，返回最高位是否借位，C 可以是 A、B 之一
inline auto sub(std::span<const uint64_t> A, std::span<const uint64_t> B, std::span<uint64_t> C)
    -> bool {
    bool borrow = false;
    for (size_type i = 0; i < C.size(); ++i) {
        uint64_t a = i < A.size() ? A[i] : 0, b = i < B.size() ? B[i] : 0;
        if (a == 0 && borrow) {
            C[i] = uint64_t(-1) - b;
        } else {
            borrow = sub_overflow(a - borrow, b, C[i]);
        }
    }
    return borrow;
}

// 计算 A = A + 1，返回是否进位
inline auto add1(std::span<uint64_t> A) -> bool {
    size_type i = 0;
    for (; i < A.size(); ++i) {
        if (A[i] != uint64_t(-1)) {
            break;
        }
    }
    if (i > 0) {
        std::fill(A.begin(), A.begin() + static_cast<int64_t>(i), 0);
    }
    if (i == A.size()) {
        return true;
    } else {
        ++A[i];
        return false;
    }
}

template<class T> inline void remove_leading_zero(T& v) {
    while (v.size() > 0 && v.back() == 0) {
        v.pop_back();
    }
}

}  // namespace utils


namespace brute {

constexpr size_type MAX_TOTAL_BITS = 2 * 128 * 64;

inline auto mul(std::span<const uint64_t> A, std::span<const uint64_t> B) -> std::vector<uint64_t> {
    std::vector<uint64_t> res;
    const size_type       n = A.size(), m = B.size();
    res.resize(n + m);
    for (size_type i = 0; i < n; ++i) {
        uint64_t  carry = 0;
        size_type j     = 0;
        for (; j < m; ++j) {
            uint128_t t = static_cast<uint128_t>(A[i]) * B[j] + res[i + j] + carry;
            res[i + j]  = t;
            carry       = t >> 64;
        }
        for (; carry; ++j) {
            uint128_t t = static_cast<uint128_t>(res[i + j]) + carry;
            res[i + j]  = t;
            carry       = t >> 64;
        }
    }
    utils::remove_leading_zero(res);
    return res;
}

}  // namespace brute


namespace fft {

// 动态 digit_bits：根据输入总比特数在 [MIN_DIGIT_BITS, MAX_DIGIT_BITS] 之间选择。
// digit_bits 越大，同样比特数的输入所需 FFT 点数越少、越快，但可保证精度的规模越小。
// MIN_DIGIT_BITS 默认 10（FFT/SSA 性能基准选定的最优值：容量 10.49M bits、层上限恰为 10，
// 规避 11 层慢 FFT），可用编译宏 BIGINT_FFT_MIN_DIGIT_BITS 覆盖调参。
#ifndef BIGINT_FFT_MIN_DIGIT_BITS
#    define BIGINT_FFT_MIN_DIGIT_BITS 10
#endif
constexpr int MIN_DIGIT_BITS = BIGINT_FFT_MIN_DIGIT_BITS;
constexpr int MAX_DIGIT_BITS = 20;

// 向上取整到 4 的幂（基-4 FFT 的变换长度），要求 x >= 1
constexpr auto pow4ceil(size_type x) -> size_type {
    size_type p = std::bit_ceil(x);      // 最小 2 的幂 >= x
    if (std::countr_zero(p) % 2 == 1) {  // 指数为奇数时再乘 2，得到最小 4 的幂
        p <<= 1;
    }
    return p;
}

// 每个 digit_bits 能保证精度的最大输入总比特数（最坏情况输入实测，见
// tests/measure_digit_bits.cpp）。 下标 = digit_bits，未使用下标为 0。
constexpr std::array<size_type, MAX_DIGIT_BITS + 1> MAX_TOTAL_BITS_FOR_DIGIT_BITS = {
    0,         // digit_bits=0 （未使用）
    0,         // digit_bits=1 （未使用）
    0,         // digit_bits=2 （未使用）
    0,         // digit_bits=3 （未使用）
    0,         // digit_bits=4 （未使用）
    0,         // digit_bits=5 （未使用）
    0,         // digit_bits=6 （未使用）
    0,         // digit_bits=7 （未使用）
    16531472,  // digit_bits=8
    11048310,  // digit_bits=9
    10485760,  // digit_bits=10
    2827308,   // digit_bits=11
    1433736,   // digit_bits=12
    851968,    // digit_bits=13
    799288,    // digit_bits=14
    334830,    // digit_bits=15
    262144,    // digit_bits=16
    128826,    // digit_bits=17
    18432,     // digit_bits=18
    19456,     // digit_bits=19
    18200,     // digit_bits=20
};
constexpr size_type MAX_TOTAL_BITS = MAX_TOTAL_BITS_FOR_DIGIT_BITS[MIN_DIGIT_BITS];

// 可选编译宏 BIGINT_FFT_MAX_FFT_LAYER：强制指定最大层数（0=由 MAX_TOTAL_BITS 推导）。
// 用于精度测量 B < MIN 的容量（B 越小容量越大、所需层数越深）或限制内存上限。
#ifndef BIGINT_FFT_MAX_FFT_LAYER
#    define BIGINT_FFT_MAX_FFT_LAYER 0
#endif

// 最大变换长度/层数由 MAX_TOTAL_BITS 推导（不写死）：
// 任意输入总比特数 T ≤ MAX_TOTAL_BITS 都会被路由到 FFT，自动选择的 digit_bits ≥ MIN_DIGIT_BITS，
// 数字个数 dA+dB-1 ≤ ceil(T/MIN_DIGIT_BITS) ≤ ceil(MAX_TOTAL_BITS/MIN_DIGIT_BITS)，
// 故所需变换长度不超过 pow4ceil(ceil(MAX_TOTAL_BITS/MIN_DIGIT_BITS))，据此反推层数。
constexpr size_type MAX_FFT_LEN = BIGINT_FFT_MAX_FFT_LAYER > 0
                                      ? (size_type(1) << (2 * BIGINT_FFT_MAX_FFT_LAYER))
                                      : pow4ceil(
                                            (MAX_TOTAL_BITS + MIN_DIGIT_BITS - 1) / MIN_DIGIT_BITS);
constexpr size_type MAX_FFT_LAYER = std::countr_zero(MAX_FFT_LEN) / 2;

// 根据输入总比特数查表，返回能保证精度的最大 digit_bits
inline auto digit_bits_for_total_bits(size_type total_bits) -> int {
    for (int b = MAX_DIGIT_BITS; b >= MIN_DIGIT_BITS; --b) {
        if (total_bits <= MAX_TOTAL_BITS_FOR_DIGIT_BITS[b]) {
            return b;
        }
    }
    unreachable();
    return MIN_DIGIT_BITS;  // 不可达，仅为满足返回值路径检查
}

// digit_bits 传入 [MIN, MAX] 时强制使用该值（供精度测量测试用），0 = 自动选择
// 注意：为便于精度测量，本函数暂不加最大输入规模限制（上限由 MAX_FFT_LAYER/fft<> 约束）
auto mul(std::span<const uint64_t> A, std::span<const uint64_t> B, int digit_bits = 0)
    -> std::vector<uint64_t>;

}  // namespace fft


namespace ntt {

constexpr size_type MAX_TOTAL_BITS = (MAX_TRANSFORM_LEN - 2) * NTT_DIGIT_BITS;

auto mul(std::span<const uint64_t> A, std::span<const uint64_t> B) -> std::vector<uint64_t>;

}  // namespace ntt


namespace ssa {

constexpr size_type MIN_TOTAL_BITS = ntt::MAX_TOTAL_BITS + 1;

auto mul(std::span<const uint64_t> A, std::span<const uint64_t> B) -> std::vector<uint64_t>;

}  // namespace ssa


inline auto _mul(std::span<const uint64_t> A, std::span<const uint64_t> B)
    -> std::vector<uint64_t> {
    size_type total_bits = (A.size() + B.size()) * 64;
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
inline auto mul_digits(const Digits& a, const Digits& b, size_type output_precision = 0,
    int64_t* p_result_point_pos = nullptr) -> Digits {
    bool a_is_b = &a == &b;

    size_type offset_a =
        output_precision == 0 || a.size() <= output_precision ? 0 : a.size() - output_precision;
    size_type offset_b =
        output_precision == 0 || b.size() <= output_precision ? 0 : b.size() - output_precision;
    size_type tail_zero_a = 0, tail_zero_b = 0;
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

    auto digits_to_vec64 = [](const Digits& x, size_type offset) -> std::vector<uint64_t> {
        const size_type       n = x.size();
        std::vector<uint64_t> res;
        res.reserve(ceil_div((n - offset) * DIGIT_BITS, 64ull));
        uint128_t tmp      = 0;
        size_type tmp_bits = 0;
        size_type i        = offset;
        while (i < n) {
            while (tmp_bits < 64 && i < n) {
                tmp |= static_cast<uint128_t>(x[i]) << tmp_bits;
                tmp_bits += DIGIT_BITS;
                ++i;
            }
            res.push_back(tmp);
            tmp >>= 64;
            tmp_bits -= 64;
        }
        if (tmp) {
            res.push_back(tmp);
        } else {
            utils::remove_leading_zero(res);
        }
        return res;
    };

    auto A = digits_to_vec64(a, offset_a);

    auto C = _mul(A, a_is_b ? A : digits_to_vec64(b, offset_b));

    Digits res;
    res.reserve(ceil_div<size_type>(C.size() * 64, DIGIT_BITS));
    {
        uint128_t tmp      = 0;
        size_type tmp_bits = 0;
        size_type i        = 0;
        while (i < C.size()) {
            if (tmp_bits < DIGIT_BITS) {
                tmp |= static_cast<uint128_t>(C[i]) << tmp_bits;
                tmp_bits += 64;
                ++i;
            }
            while (tmp_bits >= DIGIT_BITS) {
                res.push_back(tmp & DIGIT_MASK);
                tmp >>= DIGIT_BITS;
                tmp_bits -= DIGIT_BITS;
            }
        }
        if (tmp) {
            while (tmp) {
                res.push_back(tmp & DIGIT_MASK);
                tmp >>= DIGIT_BITS;
            }
        } else {
            utils::remove_leading_zero(res);
        }
    }

    if (output_precision) {
        auto offset_res = res.size() <= output_precision ? 0 : res.size() - output_precision;
        while (res[offset_res] == 0) {
            ++offset_res;
        }
        if (offset_res) {
            res.erase(res.begin(), res.begin() + static_cast<int64_t>(offset_res));
        }
        if (p_result_point_pos) {
            *p_result_point_pos -= static_cast<int64_t>(offset_a + offset_b + offset_res);
        }
    } else {
        res.insert(res.begin(), tail_zero_a + tail_zero_b, 0);
    }

    return res;
}

}  // namespace bigint::mul
