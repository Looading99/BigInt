// 公共测试工具（tests/ 各测试文件共享）：
//   1) 固定种子随机源（可复现）；
//   2) 基数 2^64 小端 limb 数组上的朴素参考实现（ref_add/ref_sub/ref_mul/...），
//      独立于库实现，供随机对拍；
//   3) BigFloat 运算结果的 limb 级期望（RefFloat 系列）。
// 头文件自由函数必须 inline（避免 ODR multiple definition）。
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <random>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "bigint/bigint.h"
#include "test_framework.h"

namespace bigint_test {

using bigint::BigFloat;
using bigint::BigInt;
using bigint::DIGIT_BITS;
using bigint::Digits;
using bigint::RoundMode;
using bigint::RoundRelativeTo;

// 与库内部表示一致（基数 2^64 的小端 limb 数组）
using LimbVec = Digits;

// ---- 随机源 ----
// 固定种子，可复现；生成器为函数内 static（cppcoreguidelines-avoid-non-const-global-variables）
inline auto rng() -> uint64_t {
    static std::mt19937_64 r(0x123456789abcdefull);
    return r();
}

// 去除前导（高位）0，保持最小表示
inline auto nz(LimbVec& v) -> void {
    while (v.size() > 1 && v.back() == 0)
        v.pop_back();
}

// 恰含 n 个 limb 的随机向量（保证最高 limb 非 0 → 长度精确）
inline auto rand_limbs_exact(std::size_t n) -> LimbVec {
    LimbVec v(n);
    for (auto& x : v) {
        switch (rng() % 4) {
        case 0: x = 0; break;
        case 1: x = UINT64_MAX; break;
        case 2: x = uint64_t(1) << (rng() % 64); break;
        default: x = rng(); break;
        }
    }
    while (v.size() > 1 && v.back() == 0)
        v.back() = rng();
    nz(v);
    return v;
}

// 随机长度（0..max_n）的随机向量
inline auto rand_limbs(std::size_t max_n) -> LimbVec {
    const std::size_t n = rng() % (max_n + 1);
    return rand_limbs_exact(n ? n : 1);
}

// ---- 参考实现：无前导 0 的小端 limb 向量 ----

inline auto ref_cmp(const LimbVec& a, const LimbVec& b) -> int {
    if (a.size() != b.size())
        return a.size() < b.size() ? -1 : 1;
    for (std::size_t i = a.size(); i-- > 0;)
        if (a[i] != b[i])
            return a[i] < b[i] ? -1 : 1;
    return 0;
}

inline auto ref_add(const LimbVec& a, const LimbVec& b) -> LimbVec {
    LimbVec     c(std::max<std::size_t>(a.size(), b.size()), 0);
    __uint128_t carry = 0;
    for (std::size_t i = 0; i < c.size(); ++i) {
        __uint128_t t = (i < a.size() ? (__uint128_t)a[i] : 0) + (i < b.size() ? b[i] : 0) + carry;
        c[i]          = static_cast<uint64_t>(t);
        carry         = t >> 64;
    }
    if (carry)
        c.push_back(1);
    nz(c);
    return c;
}

// 要求 a >= b
inline auto ref_sub(const LimbVec& a, const LimbVec& b) -> LimbVec {
    LimbVec c(a.size(), 0);
    bool    borrow = false;
    for (std::size_t i = 0; i < a.size(); ++i) {
        uint64_t ai = a[i], bi = i < b.size() ? b[i] : 0;
        uint64_t t  = ai - bi;
        bool     nb = ai < bi;
        t -= borrow;
        nb |= borrow && ai == bi;
        borrow = nb;
        c[i]   = t;
    }
    nz(c);
    return c;
}

inline auto ref_mul(const LimbVec& a, const LimbVec& b) -> LimbVec {
    if ((a.size() == 1 && a[0] == 0) || (b.size() == 1 && b[0] == 0))
        return {0};
    LimbVec c(a.size() + b.size(), 0);
    for (std::size_t i = 0; i < a.size(); ++i) {
        __uint128_t carry = 0;
        for (std::size_t j = 0; j < b.size(); ++j) {
            __uint128_t t = (__uint128_t)a[i] * b[j] + c[i + j] + carry;
            c[i + j]      = static_cast<uint64_t>(t);
            carry         = t >> 64;
        }
        std::size_t k = i + b.size();
        while (carry) {
            __uint128_t t = (__uint128_t)c[k] + carry;
            c[k]          = static_cast<uint64_t>(t);
            carry         = t >> 64;
            ++k;
        }
    }
    nz(c);
    return c;
}

inline auto ref_shl(const LimbVec& v, std::size_t bits) -> LimbVec {
    LimbVec r(bits / 64, 0);
    r.insert(r.end(), v.begin(), v.end());
    std::size_t rem = bits % 64;
    if (rem) {
        uint64_t carry = 0;
        for (auto& x : r) {
            uint64_t nx = (x << rem) | carry;
            carry       = x >> (64 - rem);
            x           = nx;
        }
        if (carry)
            r.push_back(carry);
    }
    nz(r);
    return r;
}

inline auto ref_shr(const LimbVec& v, std::size_t bits) -> LimbVec {
    const std::size_t limb_shift = bits / 64;
    if (limb_shift >= v.size())
        return {0};
    LimbVec         r(v.begin() + static_cast<int64_t>(limb_shift), v.end());
    const std::size_t rem = bits % 64;
    if (rem) {
        for (std::size_t i = 0; i < r.size(); ++i) {
            uint64_t lo = r[i] >> rem;
            uint64_t hi = i + 1 < r.size() ? r[i + 1] << (64 - rem) : 0;
            r[i]        = lo | hi;
        }
    }
    nz(r);
    return r;
}

// 无符号二进制逐位长除，返回 (商, 余)；要求 b 非零
inline auto ref_divmod(const LimbVec& a, const LimbVec& b) -> std::pair<LimbVec, LimbVec> {
    LimbVec         cur{0}, q{0};
    const std::size_t bits = a.size() * 64;
    for (std::size_t i = bits; i-- > 0;) {
        cur = ref_shl(cur, 1);
        if ((a[i / 64] >> (i % 64)) & 1)
            cur = ref_add(cur, {1});
        q = ref_shl(q, 1);
        if (ref_cmp(cur, b) >= 0) {
            cur = ref_sub(cur, b);
            q   = ref_add(q, {1});
        }
    }
    return {q, cur};
}

// 十进制字符串 → limbs 的独立参考（逐位乘 10 累加，只用 ref_*；忽略非数字字符）
inline auto ref_parse_dec(const std::string& s) -> LimbVec {
    LimbVec res{0};
    for (char ch : s) {
        if (ch < '0' || ch > '9')
            continue;
        res = ref_add(ref_mul(res, {10}), {static_cast<uint64_t>(ch - '0')});
    }
    return res;
}

// 十六进制字符串 → limbs 的独立参考（忽略非 hex 字符）
inline auto ref_parse_hex(const std::string& s) -> LimbVec {
    LimbVec res{0};
    for (char ch : s) {
        uint64_t d = 0;
        if (ch >= '0' && ch <= '9')
            d = static_cast<uint64_t>(ch - '0');
        else if (ch >= 'a' && ch <= 'f')
            d = static_cast<uint64_t>(ch - 'a' + 10);
        else if (ch >= 'A' && ch <= 'F')
            d = static_cast<uint64_t>(ch - 'A' + 10);
        else
            continue;
        res = ref_add(ref_shl(res, 4), {d});
    }
    return res;
}

inline auto to_limbs(const BigInt& x) -> LimbVec {
    return {x.get_data().begin(), x.get_data().end()};
}

inline auto print_limbs(const char* name, const LimbVec& v) -> void {
    std::cout << name << "=[";
    for (auto x : v)
        std::cout << x << ",";
    std::cout << "]";
}

// 便捷断言：BigInt 的绝对值 limb 与符号（符号 0 时绝对值必为 0）
inline auto check_matches(const BigInt& got, const LimbVec& exp_abs, int exp_sign, const char* msg)
    -> void {
    const int sign = ref_cmp(exp_abs, {0}) == 0 ? 0 : exp_sign;
    TEST_CHECK(to_limbs(got) == exp_abs && got.sign() == sign, msg);
}

// ---- BigFloat 运算结果的 limb 级期望（绝对值 limbs + point + 符号） ----

struct RefFloat {
    LimbVec limbs{0};
    int64_t point = 0;
    int     sign  = 0;
};

// fa ± fb 的精确参考（基于内部表示 get_data/get_point_pos/sign）
inline auto ref_float_add_sub(const BigFloat& a, const BigFloat& b, bool is_sub) -> RefFloat {
    if (a.is_zero()) {
        if (is_sub) {
            const auto& d = b.get_data();
            const auto  s = b.sign();
            return {.limbs = d, .point = b.get_point_pos(), .sign = s == 0 ? 0 : -s};
        }
        return {.limbs = b.get_data(), .point = b.get_point_pos(), .sign = b.sign()};
    }
    if (b.is_zero()) {
        return {.limbs = a.get_data(), .point = a.get_point_pos(), .sign = a.sign()};
    }
    const int64_t pa = a.get_point_pos(), pb = b.get_point_pos();
    const bool    na = a.sign() < 0, nb0 = b.sign() < 0;
    const bool    nb = is_sub ? !nb0 : nb0;
    int64_t       point = std::max(pa, pb);
    // 对齐：值 = sum(data[i] * 2^(64*(i - point)))，对齐到 point 需左移 (point - p) 个 limb
    const auto align = [point](const LimbVec& v, int64_t p) {
        LimbVec r(static_cast<std::size_t>(point - p), 0);
        r.insert(r.end(), v.begin(), v.end());
        return r;
    };
    LimbVec A = align(a.get_data(), pa);
    LimbVec B = align(b.get_data(), pb);
    // 长度统一（ref_add/ref_sub 自己处理，但 ref_cmp 需要同长）
    if (A.size() < B.size())
        A.resize(B.size(), 0);
    else if (B.size() < A.size())
        B.resize(A.size(), 0);
    if (na == nb) {
        LimbVec c = ref_add(A, B);
        // 修剪尾零（库 add_or_sub 结果会 remove_tail_zero）
        while (c.size() > 1 && c[0] == 0) {
            c.erase(c.begin());
            point -= 1;
        }
        return {.limbs = std::move(c), .point = point, .sign = na ? -1 : 1};
    }
    const int cmp = ref_cmp(A, B);
    if (cmp == 0)
        return {.limbs = {0}, .point = 0, .sign = 0};
    if (cmp > 0) {
        LimbVec c = ref_sub(A, B);
        while (c.size() > 1 && c[0] == 0) {
            c.erase(c.begin());
            point -= 1;
        }
        return {.limbs = std::move(c), .point = point, .sign = na ? -1 : 1};
    }
    {
        LimbVec c = ref_sub(B, A);
        while (c.size() > 1 && c[0] == 0) {
            c.erase(c.begin());
            point -= 1;
        }
        return {.limbs = std::move(c), .point = point, .sign = nb ? -1 : 1};
    }
}

// fa * fb（不截断）的精确参考
inline auto ref_float_mul(const BigFloat& a, const BigFloat& b) -> RefFloat {
    if (a.is_zero() || b.is_zero())
        return {.limbs = {0}, .point = 0, .sign = 0};
    LimbVec c     = ref_mul(a.get_data(), b.get_data());
    int64_t point = a.get_point_pos() + b.get_point_pos();
    // BigFloat::mul 构造后 remove_tail_zero：低位 0 修剪，point 相应减小
    while (c.size() > 1 && c[0] == 0) {
        c.erase(c.begin());
        --point;
    }
    return {.limbs = std::move(c), .point = point, .sign = a.sign() * b.sign()};
}

// 模拟 _round/round 对内部表示 (v, point, neg) 的效果（与库语义一致，作回归参考）
inline auto ref_float_round(Digits v, int64_t point, bool neg, RoundMode mode, int64_t precision,
    RoundRelativeTo relative) -> RefFloat {
    if (v.size() == 1 && v[0] == 0)  // is_zero → round 无操作
        return {.limbs = std::move(v), .point = point, .sign = 0};
    const auto n = static_cast<int64_t>(v.size());
    if (relative == RoundRelativeTo::Point)
        precision = n - point - precision;
    const int64_t round_idx = n - precision - 1;
    if (round_idx >= n)  // 全部舍掉 → 0
        return {.limbs = {0}, .point = 0, .sign = 0};
    if (round_idx < 0)  // 精度超过长度 → 无操作
        return {.limbs = std::move(v), .point = point, .sign = neg ? -1 : 1};
    // 丢弃 v[0..round_idx]，按模式判断是否进位
    bool dropped_nonzero = false;
    for (int64_t i = 0; i <= round_idx; ++i)
        dropped_nonzero = dropped_nonzero || v[static_cast<std::size_t>(i)] != 0;
    bool inc = false;
    switch (mode) {
    case RoundMode::Floor: inc = neg && dropped_nonzero; break;
    case RoundMode::Ceil: inc = !neg && dropped_nonzero; break;
    case RoundMode::RoundHalfUp:
        inc = (v[static_cast<std::size_t>(round_idx)] >> (DIGIT_BITS - 1)) & 1;
        break;
    default: break;  // Truncate / 非法值：不进位
    }
    v.erase(v.begin(), v.begin() + round_idx + 1);
    point -= round_idx + 1;
    if (v.empty() && !inc)
        return {.limbs = {0}, .point = 0, .sign = 0};
    if (inc) {  // 进位链（add1：空向量进位 1）
        bool carry = true;
        for (auto& x : v) {
            if (x != UINT64_MAX) {
                ++x;
                carry = false;
                break;
            }
            x = 0;
        }
        if (carry)
            v.push_back(1);
    }
    return {.limbs = std::move(v), .point = point, .sign = neg ? -1 : 1};
}

// BigInt(f, mode) 的期望（绝对值 limbs + 符号）
inline auto ref_float_to_bigint_abs(const BigFloat& f, RoundMode mode) -> std::pair<LimbVec, int> {
    const LimbVec& d     = f.get_data();
    const int64_t  point = f.get_point_pos();
    const bool     neg   = f.sign() < 0;
    if (f.is_zero())
        return {{0}, 0};
    if (point <= 0)  // 纯整数（含低位补 0）：左移 -point 个 limb
        return {ref_shl(d, static_cast<std::size_t>(-point) * 64), neg ? -1 : 1};
    if (point > static_cast<int64_t>(d.size()))  // |值| < 1 → 0
        return {{0}, 0};
    // 与 convert_from_BigFloat 相同：_round(mode, d, neg, point - 1)
    RefFloat r = ref_float_round(
        d, point, neg, mode, static_cast<int64_t>(d.size()) - point, RoundRelativeTo::Significant);
    return {std::move(r.limbs), r.sign};
}

// 便捷断言：BigFloat 的内部表示与期望一致
inline auto check_ref_float(const BigFloat& got, const RefFloat& exp, const char* msg) -> void {
    TEST_CHECK(got.get_data() == exp.limbs && got.get_point_pos() == exp.point
                   && got.sign() == exp.sign,
        msg);
}

}  // namespace bigint_test
