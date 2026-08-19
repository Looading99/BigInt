#pragma once

#include <cstdint>
#include <span>
#include <vector>


#include "bigint/bigint_base.h"


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

inline void remove_leading_zero(std::vector<uint64_t>& v) {
    size_type i = v.size();
    while (i > 0) {
        --i;
        if (v[i] != 0) {
            break;
        }
    }
    v.resize(i + 1);
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

static constexpr size_type MAX_FFT_LAYER = 7;
static constexpr size_type MAX_FFT_LEN   = size_type(1) << (2 * MAX_FFT_LAYER);

#ifndef MUL_FFT_DIGIT_BITS
#    define MUL_FFT_DIGIT_BITS 17  // NOLINT
#endif
constexpr int FFT_DIGIT_BITS = MUL_FFT_DIGIT_BITS;

constexpr size_type MAX_TOTAL_BITS = (MAX_FFT_LEN + 1) * FFT_DIGIT_BITS;

auto mul(std::span<const uint64_t> A, std::span<const uint64_t> B) -> std::vector<uint64_t>;

}  // namespace fft


namespace ssa {

constexpr size_type MIN_TOTAL_BITS = fft::MAX_TOTAL_BITS + 1;

auto mul(std::span<const uint64_t> A, std::span<const uint64_t> B) -> std::vector<uint64_t>;

}  // namespace ssa


inline auto _mul(std::span<const uint64_t> A, std::span<const uint64_t> B)
    -> std::vector<uint64_t> {
    size_type total_bits = (A.size() + B.size()) * 64;
    if (total_bits <= brute::MAX_TOTAL_BITS) {
        return brute::mul(A, B);
    } else if (total_bits <= fft::MAX_TOTAL_BITS) {
        return fft::mul(A, B);
    } else {
        return ssa::mul(A, B);
    }
}

// 将两个数组（按照 output_precision ）相乘。必须保证输入数组非全0。
inline auto mul_digits(const Digits& a, const Digits& b, size_type output_precision = 0,
    int64_t* p_result_point_pos = nullptr) -> Digits {

    size_type offset_a =
        output_precision == 0 || a.size() <= output_precision ? 0 : a.size() - output_precision;
    size_type offset_b =
        output_precision == 0 || b.size() <= output_precision ? 0 : b.size() - output_precision;
    size_type tail_zero_a = 0, tail_zero_b = 0;
    while (a[offset_a + tail_zero_a] == 0) {
        ++tail_zero_a;
    }
    while (b[offset_b + tail_zero_b] == 0) {
        ++tail_zero_b;
    }
    offset_a += tail_zero_a;
    offset_b += tail_zero_b;

    auto digits_to_uint64_vec = [](const Digits& x, size_type offset) -> std::vector<uint64_t> {
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

    auto A = digits_to_uint64_vec(a, offset_a);
    auto B = digits_to_uint64_vec(b, offset_b);

    auto C = _mul(A, B);

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
            i = res.size() - 1;
            while (res[i] == 0) {
                --i;
            }
            res.resize(i + 1);
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
