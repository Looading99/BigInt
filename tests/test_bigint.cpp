// BigInt/BigFloat 核心功能测试：随机对拍朴素参考实现 + 边界用例 + 已知值。
//
// 覆盖：加减乘除（含符号）、divmod 四种舍入、移位、位运算、字符串与 hex 往返、
//       BigFloat（double 往返、加减对拍、mul(precision) 截断、round 全舍进位、
//       reciprocal/inv）、进位/借位长链与 2 的幂边界、跨 FFT/NTT 分发的大数除法。
// 随机输入使用固定种子，可复现。
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <numbers>
#include <random>
#include <string>
#include <vector>

#include "bigint/bigint.h"
#include "test_framework.h"

using bigint::BigFloat;
using bigint::BigInt;
using bigint::Digits;
using bigint::RoundMode;
using bigint::RoundRelativeTo;

// 与库内部表示一致（基数 2^64 的小端 limb 数组）
using LimbVec = Digits;

// ---- 朴素参考实现（基数 2^64 小端，无前导 0） ----

static void nz(LimbVec& v) {
    while (v.size() > 1 && v.back() == 0) v.pop_back();
}
static auto ref_cmp(const LimbVec& a, const LimbVec& b) -> int {
    if (a.size() != b.size()) return a.size() < b.size() ? -1 : 1;
    for (std::size_t i = a.size(); i-- > 0;)
        if (a[i] != b[i]) return a[i] < b[i] ? -1 : 1;
    return 0;
}
static auto ref_add(const LimbVec& a, const LimbVec& b) -> LimbVec {
    LimbVec c(std::max(a.size(), b.size()), 0);
    __uint128_t carry = 0;
    for (std::size_t i = 0; i < c.size(); ++i) {
        __uint128_t t = (i < a.size() ? (__uint128_t)a[i] : 0) + (i < b.size() ? b[i] : 0) + carry;
        c[i]          = (uint64_t)t;
        carry         = t >> 64;
    }
    if (carry) c.push_back(1);
    nz(c);
    return c;
}
// 要求 a >= b
static auto ref_sub(const LimbVec& a, const LimbVec& b) -> LimbVec {
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
static auto ref_mul(const LimbVec& a, const LimbVec& b) -> LimbVec {
    if ((a.size() == 1 && a[0] == 0) || (b.size() == 1 && b[0] == 0)) return {0};
    LimbVec c(a.size() + b.size(), 0);
    for (std::size_t i = 0; i < a.size(); ++i) {
        __uint128_t carry = 0;
        for (std::size_t j = 0; j < b.size(); ++j) {
            __uint128_t t = (__uint128_t)a[i] * b[j] + c[i + j] + carry;
            c[i + j]      = (uint64_t)t;
            carry         = t >> 64;
        }
        std::size_t k = i + b.size();
        while (carry) {
            __uint128_t t = (__uint128_t)c[k] + carry;
            c[k]          = (uint64_t)t;
            carry         = t >> 64;
            ++k;
        }
    }
    nz(c);
    return c;
}

static auto ref_shl(const LimbVec& v, std::size_t bits) -> LimbVec {
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
        if (carry) r.push_back(carry);
    }
    nz(r);
    return r;
}

// ---- 测试工具 ----

// 固定种子，可复现；生成器为函数内 static 而非非 const 全局变量
// （cppcoreguidelines-avoid-non-const-global-variables），rng() 直接返回随机数
static auto rng() -> uint64_t {
    static std::mt19937_64 r(0x123456789abcdefull);
    return r();
}

static auto rand_limbs(std::size_t max_n) -> LimbVec {
    std::size_t n = rng() % (max_n + 1);
    LimbVec     v(n ? n : 1);
    for (auto& x : v) {
        switch (rng() % 4) {
        case 0: x = 0; break;
        case 1: x = UINT64_MAX; break;
        case 2: x = uint64_t(1) << (rng() % 64); break;
        default: x = rng(); break;
        }
    }
    if (v.size() > 1) v.back() = rng();  // 前导 0 概率极低，直接随机
    nz(v);
    return v;
}

// v 值 = Σ v[i] * 2^(64i)，通过公开 API 拼装
static auto to_bigint(const LimbVec& v, bool neg) -> BigInt {
    BigInt res(0);
    for (std::size_t i = v.size(); i-- > 0;) {
        res <<= 64;
        res |= BigInt(v[i]);
    }
    if (neg) res.flip_sign();
    return res;
}

static auto to_limbs(const BigInt& x) -> LimbVec {
    return {x.get_data().begin(), x.get_data().end()};
}

static void print_limbs(const char* name, const LimbVec& v) {
    std::cout << name << "=[";
    for (auto x : v) std::cout << x << ",";
    std::cout << "]";
}

// ---- 用例 ----

static void test_boundary_chains() {
    std::cout << "[boundary chains]\n";
    for (std::size_t k = 1; k <= 3; ++k) {
        const LimbVec one  = {1};
        LimbVec       big  = LimbVec(k, UINT64_MAX);  // 2^(64k) - 1
        LimbVec       pow2 = LimbVec(k + 1, 0);
        pow2[k] = 1;  // 2^(64k)
        TEST_CHECK(ref_cmp(ref_add(big, one), pow2) == 0, "carry chain");
        TEST_CHECK(ref_cmp(ref_sub(pow2, one), big) == 0, "borrow chain");
        TEST_CHECK(ref_cmp(ref_sub(pow2, pow2), LimbVec{0}) == 0, "equal magnitude");
        // BigInt 层面对拍
        BigInt b(big[0]);
        for (std::size_t i = 1; i < big.size(); ++i) {
            b <<= 64;
            b |= BigInt(big[i]);
        }
        BigInt b2(b);
        ++b2;  // 进位链
        TEST_CHECK(to_limbs(b2) == pow2, "BigInt ++ carry chain");
        --b2;  // 借位链
        TEST_CHECK(to_limbs(b2) == big, "BigInt -- borrow chain");
        TEST_CHECK((b2 - b2).is_zero(), "x - x == 0");
        TEST_CHECK((b2 - b).is_zero(), "x - x == 0 (same obj)");
        BigInt neg = BigInt(0) - b;
        TEST_CHECK(neg.sign() == -1 && to_limbs(neg) == big, "0 - x == -x");
        TEST_CHECK(to_limbs(BigInt(0) - neg) == big, "-(-x) == x");
        TEST_CHECK((b - b2).is_zero(), "x - x2 == 0");
        TEST_CHECK((b2 - BigInt(1)) + BigInt(1) == b2, "b2-1+1 == b2");
        TEST_CHECK((BigInt(1) - b2) == -(b - BigInt(1)), "1-b2 == -(b-1)");
    }
    // 符号翻转：a - b 跨零
    const LimbVec x = {5};
    const LimbVec y = {7};
    BigInt X = to_bigint(x, false), Y = to_bigint(y, false);
    TEST_CHECK((X - Y) == BigInt(-2), "5 - 7 == -2");
    TEST_CHECK((Y - X) == BigInt(2), "7 - 5 == 2");
    TEST_CHECK((-X - Y) == BigInt(-12), "-5 - 7 == -12");
    TEST_CHECK((-X + Y) == BigInt(2), "-5 + 7 == 2");
    TEST_CHECK((X + (-Y)) == BigInt(-2), "5 + (-7) == -2");
    // 64 位整数混合
    TEST_CHECK((X - uint64_t(9)) == BigInt(-4), "5 - 9 == -4");
    TEST_CHECK((uint64_t(9) - X) == BigInt(4), "9 - 5 == 4");
    TEST_CHECK((-X + uint64_t(3)) == BigInt(-2), "-5 + 3 == -2");
    TEST_CHECK((X + uint64_t(3)) == BigInt(8), "5 + 3 == 8");
    TEST_CHECK((X - uint64_t(5)).is_zero(), "5 - 5 == 0");
    TEST_CHECK((X - uint64_t(6)) == BigInt(-1), "5 - 6 == -1");
    BigInt big2 = to_bigint(LimbVec{UINT64_MAX, UINT64_MAX}, false);  // 2^128 - 1
    TEST_CHECK((big2 + uint64_t(1)) == to_bigint(LimbVec{0, 0, 1}, false), "2^128-1 + 1");
    TEST_CHECK((big2 - uint64_t(UINT64_MAX)) == to_bigint(LimbVec{0, UINT64_MAX}, false),
        "2^128-1 - (2^64-1)");
    TEST_CHECK((big2 * uint64_t(3)) == to_bigint(LimbVec{UINT64_MAX - 2, UINT64_MAX, 2}, false),
        "(2^128-1)*3");
}

static void test_random_arithmetic() {
    std::cout << "[random arithmetic]\n";
    for (std::size_t iter = 0; iter < 3000; ++iter) {
        const LimbVec a = rand_limbs(3), b = rand_limbs(3);
        const bool    na = rng() & 1, nb = rng() & 1;
        const BigInt  A = to_bigint(a, na), B = to_bigint(b, nb);
        const int     sa = na ? -1 : 1, sb = nb ? -1 : 1;

        // 期望：A+B、A-B（绝对值 + 符号）
        // A+B：na==nb → abs 和、符号 na；na!=nb → abs 差、符号 = |a|>=|b| ? na : nb
        // A-B = A + (-B)：na!=nb → abs 和、符号 na；na==nb → abs 差、符号 = |a|>=|b| ? na : !na
        LimbVec   ab, a_b;
        int       s_ab = 0, s_a_b = 0;
        const bool ge = ref_cmp(a, b) >= 0;
        if (na == nb) {
            ab   = ref_add(a, b);
            s_ab = na ? -1 : 1;
            if (ge) {
                a_b   = ref_sub(a, b);
                s_a_b = na ? -1 : 1;
            } else {
                a_b   = ref_sub(b, a);
                s_a_b = na ? 1 : -1;
            }
        } else {
            if (ge) {
                ab   = ref_sub(a, b);
                s_ab = na ? -1 : 1;
            } else {
                ab   = ref_sub(b, a);
                s_ab = nb ? -1 : 1;
            }
            a_b   = ref_add(a, b);
            s_a_b = na ? -1 : 1;
        }
        BigInt sum = A + B, diff = A - B;
        TEST_CHECK(ref_cmp(to_limbs(sum), ab) == 0
                       && sum.sign() == (ref_cmp(ab, {0}) == 0 ? 0 : s_ab),
            "A+B random");
        TEST_CHECK(ref_cmp(to_limbs(diff), a_b) == 0
                       && diff.sign() == (ref_cmp(a_b, {0}) == 0 ? 0 : s_a_b),
            "A-B random");

        // 乘法（小规模走 brute）
        LimbVec exp_mul = ref_mul(a, b);
        int     s_mul   = (sa != sb) ? -1 : 1;
        TEST_CHECK((A * B).sign() == (ref_cmp(exp_mul, {0}) == 0 ? 0 : s_mul)
                       && ref_cmp(to_limbs(A * B), exp_mul) == 0,
            "A*B random");

        // 64 位整数混合
        const uint64_t c = rng();
        LimbVec        cc = {c};
        if (c != 0) {
            // A+c：A 正 → |a|+c；A 负 → |a|>=c ? -( |a|-c ) : +( c-|a| )
            LimbVec e_sum;
            int     s_sum = 0;
            if (!na) {
                e_sum = ref_add(a, cc);
                s_sum = 1;
            } else if (ref_cmp(a, cc) >= 0) {
                e_sum = ref_sub(a, cc);
                s_sum = -1;
            } else {
                e_sum = ref_sub(cc, a);
                s_sum = 1;
            }
            BigInt sumc = A + c;
            TEST_CHECK(ref_cmp(to_limbs(sumc), e_sum) == 0
                           && sumc.sign() == (ref_cmp(e_sum, {0}) == 0 ? 0 : s_sum),
                "A+c random");
            // A-c：A 负 → -( |a|+c )；A 正 → |a|>=c ? |a|-c : -( c-|a| )
            LimbVec e_sub;
            int     s_sub = 0;
            if (na) {
                e_sub = ref_add(a, cc);
                s_sub = -1;
            } else if (ref_cmp(a, cc) >= 0) {
                e_sub = ref_sub(a, cc);
                s_sub = 1;
            } else {
                e_sub = ref_sub(cc, a);
                s_sub = -1;
            }
            BigInt subc = A - c;
            TEST_CHECK(ref_cmp(to_limbs(subc), e_sub) == 0
                           && subc.sign() == (ref_cmp(e_sub, {0}) == 0 ? 0 : s_sub),
                "A-c random");
            // 恒等式
            TEST_CHECK(to_limbs((A - c) + c) == a, "A-c+c == A");
            TEST_CHECK(to_limbs((A + c) - c) == a, "A+c-c == A");
            TEST_CHECK(ref_cmp(to_limbs(A * c), ref_mul(a, cc)) == 0, "A*c random");
        }

        // 移位
        const std::size_t sh = rng() % 200;
        LimbVec           e_shl = a;
        for (std::size_t t = 0; t < sh; ++t) e_shl = ref_add(e_shl, e_shl);
        TEST_CHECK(ref_cmp(to_limbs(A << sh), e_shl) == 0, "A<<sh random");
        LimbVec e_shr = a;
        for (std::size_t t = 0; t < sh; ++t) {
            LimbVec r2;
            for (std::size_t i = 0; i < e_shr.size(); ++i) {
                uint64_t ai = e_shr[i];
                uint64_t hi = i + 1 < e_shr.size() ? e_shr[i + 1] : 0;
                r2.push_back((ai >> 1) | (hi << 63));
            }
            nz(r2);
            e_shr = r2;
        }
        TEST_CHECK(ref_cmp(to_limbs(A >> sh), e_shr) == 0, "A>>sh random");

        // 位运算（忽略符号）
        LimbVec e_and, e_or, e_xor;
        for (std::size_t i = 0; i < std::max(a.size(), b.size()); ++i) {
            uint64_t x = i < a.size() ? a[i] : 0, y = i < b.size() ? b[i] : 0;
            e_and.push_back(x & y);
            e_or.push_back(x | y);
            e_xor.push_back(x ^ y);
        }
        nz(e_and);
        nz(e_or);
        nz(e_xor);
        TEST_CHECK(ref_cmp(to_limbs(A & B), e_and) == 0, "A&B random");
        TEST_CHECK(ref_cmp(to_limbs(A | B), e_or) == 0, "A|B random");
        TEST_CHECK(ref_cmp(to_limbs(A ^ B), e_xor) == 0, "A^B random");
    }
}

static void test_divmod() {
    std::cout << "[divmod]\n";
    for (std::size_t iter = 0; iter < 500; ++iter) {
        const LimbVec a = rand_limbs(3), b = rand_limbs(2);
        if (ref_cmp(b, {0}) == 0) continue;
        const bool   na = rng() & 1, nb = rng() & 1;
        const BigInt A = to_bigint(a, na), B = to_bigint(b, nb);
        // 参考长除（二进制逐位，简单可靠）
        LimbVec R = a, Q;
        int     sQ = na ^ nb ? -1 : 1;
        {
            LimbVec   cur = {0};
            LimbVec   q   = {0};
            const std::size_t bits = a.size() * 64;
            for (std::size_t i = bits; i-- > 0;) {
                cur = ref_add(cur, cur);
                uint64_t bit = (a[i / 64] >> (i % 64)) & 1;
                if (bit) cur = ref_add(cur, {1});
                q = ref_add(q, q);
                if (ref_cmp(cur, b) >= 0) {
                    cur = ref_sub(cur, b);
                    q   = ref_add(q, {1});
                }
            }
            Q = q;
            R = cur;
        }
        auto [Qb, Rb] = A.divmod(B, RoundMode::Truncate);
        TEST_CHECK(ref_cmp(to_limbs(Qb), Q) == 0 && (Qb.sign() == (ref_cmp(Q, {0}) == 0 ? 0 : sQ)),
            "divmod Q random");
        TEST_CHECK(to_limbs(Rb) == R, "divmod R random");
        TEST_CHECK(to_limbs(Qb * B + Rb) == a, "Q*B+R == A");
    }
}

static void test_string_roundtrip() {
    std::cout << "[string roundtrip]\n";
    for (std::size_t iter = 0; iter < 200; ++iter) {
        const LimbVec a  = rand_limbs(3);
        const bool    na = rng() & 1;
        const BigInt  A  = to_bigint(a, na);
        TEST_CHECK(BigInt(A.to_string()) == A, "dec roundtrip");
        TEST_CHECK(BigInt(A.to_string(true), true) == A, "hex roundtrip");
    }
}

static void test_known_values() {
    std::cout << "[known values]\n";
    BigInt p256(1);
    p256 <<= 256;
    TEST_CHECK(p256.to_string()
                   == "115792089237316195423570985008687907853269984665640564039457584007913129639936",
        "2^256");
    TEST_CHECK((p256 - 1).to_string()
                   == "115792089237316195423570985008687907853269984665640564039457584007913129639935",
        "2^256-1");
    BigInt fact(1);
    for (int i = 2; i <= 30; ++i) fact *= BigInt(i);
    TEST_CHECK(fact.to_string() == "265252859812191058636308480000000", "30!");
    BigInt p10(1);
    p10 *= BigInt(10000000000000000000ull);
    p10 *= BigInt(10000000000000000000ull);
    TEST_CHECK(p10.to_string() == "100000000000000000000000000000000000000", "10^40");
    // divmod 四模式
    BigInt q1(17), q2(5);
    auto [Q1, R1] = q1.divmod(q2, RoundMode::Truncate);
    TEST_CHECK(Q1 == BigInt(3) && R1 == BigInt(2), "17 divmod 5 trunc");
    auto [Q2, R2] = q1.divmod(q2, RoundMode::Floor);
    TEST_CHECK(Q2 == BigInt(3) && R2 == BigInt(2), "17 divmod 5 floor");
    auto [Q3, R3] = BigInt(-17).divmod(q2, RoundMode::Floor);
    TEST_CHECK(Q3 == BigInt(-4) && R3 == BigInt(3), "-17 divmod 5 floor");
    auto [Q4, R4] = BigInt(-17).divmod(q2, RoundMode::Ceil);
    TEST_CHECK(Q4 == BigInt(-3) && R4 == BigInt(-2), "-17 divmod 5 ceil");
    auto [Q5, R5] = q1.divmod(q2, RoundMode::RoundHalfUp);
    TEST_CHECK(Q5 == BigInt(3) && R5 == BigInt(2), "17 divmod 5 halfup");
    auto [Q6, R6] = BigInt(18).divmod(BigInt(5), RoundMode::RoundHalfUp);
    TEST_CHECK(Q6 == BigInt(4) && R6 == BigInt(-2), "18 divmod 5 halfup");
}

static void test_bigfloat() {
    std::cout << "[bigfloat]\n";
    // double 往返
    const std::array vals{0.0, 1.0, -1.0, 0.5, 2.0, std::numbers::pi, 1e-300, 1e300,
        5e-324, 2.2250738585072014e-308, 1.7976931348623157e308, 0.1, -0.1, 123456.789};
    for (double d : vals) {
        BigFloat f(d);
        TEST_CHECK(std::bit_cast<uint64_t>(f.to_double()) == std::bit_cast<uint64_t>(d),
            "double roundtrip");
    }
    for (int i = 0; i < 200; ++i) {
        uint64_t bits = rng();
        auto     d    = std::bit_cast<double>(bits);
        if (std::isnan(d) || std::isinf(d)) continue;  // BigFloat 构造对 inf/nan 抛异常
        BigFloat f(d);
        TEST_CHECK(std::bit_cast<uint64_t>(f.to_double()) == bits, "double roundtrip random");
    }
    // BigFloat 加减与 BigInt 对拍（点位置 0）
    for (int i = 0; i < 200; ++i) {
        const LimbVec a  = rand_limbs(2), b = rand_limbs(2);
        const bool    na = rng() & 1, nb = rng() & 1;
        BigFloat fa(to_bigint(a, na)), fb(to_bigint(b, nb));
        BigFloat fs = fa + fb;
        BigInt   exp = to_bigint(a, na) + to_bigint(b, nb);
        if (!(BigInt(fs) == exp)) {
            ++bigint_test::fails();
            ++bigint_test::cases();
            std::cout << "FAIL: BigFloat add vs BigInt: ";
            print_limbs("a", a);
            std::cout << " na=" << na << " ";
            print_limbs("b", b);
            std::cout << " nb=" << nb << " fs=" << BigInt(fs).to_string()
                      << " exp=" << exp.to_string() << "\n";
            continue;
        }
        ++bigint_test::cases();
        BigFloat fd   = fa - fb;
        BigInt   exp2 = to_bigint(a, na) - to_bigint(b, nb);
        if (!(BigInt(fd) == exp2)) {
            ++bigint_test::fails();
            std::cout << "FAIL: BigFloat sub vs BigInt: ";
            print_limbs("a", a);
            std::cout << " na=" << na << " ";
            print_limbs("b", b);
            std::cout << " nb=" << nb << " fd=" << BigInt(fd).to_string()
                      << " exp=" << exp2.to_string() << "\n";
        }
    }
    // mul(precision) 截断语义：输入截到顶部 precision 个 limb（跳过窗口内低端 0）→
    // 乘积 → 结果截顶；point 补偿 = 总截断量（精确模拟 mul_digits 逻辑）
    for (int i = 0; i < 200; ++i) {
        const LimbVec a = rand_limbs(2), b = rand_limbs(2);
        BigFloat      fa(to_bigint(a, false)), fb(to_bigint(b, false));
        const std::size_t precision = 1 + rng() % 4;
        BigFloat fm = BigFloat::mul(fa, fb, precision);
        LimbVec  ta(a), tb(b);
        std::size_t off_a = ta.size() > precision ? ta.size() - precision : 0;
        std::size_t off_b = tb.size() > precision ? tb.size() - precision : 0;
        while (off_a < ta.size() && ta[off_a] == 0) ++off_a;
        while (off_b < tb.size() && tb[off_b] == 0) ++off_b;
        ta.erase(ta.begin(), ta.begin() + static_cast<int64_t>(off_a));
        tb.erase(tb.begin(), tb.begin() + static_cast<int64_t>(off_b));
        LimbVec c{0};
        if (!ta.empty() && !tb.empty()) {
            c = ref_mul(ta, tb);
            std::size_t off_r = c.size() > precision ? c.size() - precision : 0;
            while (off_r < c.size() && c[off_r] == 0) ++off_r;
            c.erase(c.begin(), c.begin() + static_cast<int64_t>(off_r));
            if (c.empty()) c.push_back(0);
            c = ref_shl(c, 64 * (off_a + off_b + off_r));
        }
        TEST_CHECK(to_limbs(BigInt(fm)) == c, "BigFloat mul precision");
    }
    // reciprocal：x * reciprocal(x) ≈ 1（带舍入）；inv 别名等价
    for (int i = 0; i < 50; ++i) {
        const LimbVec a = rand_limbs(2);
        if (ref_cmp(a, {0}) == 0) continue;
        BigFloat        f(to_bigint(a, false));
        const std::size_t prec = 2 + rng() % 6;
        BigFloat        inv  = f.reciprocal(prec);
        BigFloat        prod = BigFloat::mul(f, inv, prec + 4);
        BigFloat        err  = prod - BigFloat(1);
        // 误差应远小于 1 个 limb 的相对量级（乘积精度 prec+4 个 limb）
        TEST_CHECK(BigInt(err).len() <= 1, "reciprocal x*1/x ~= 1");
        // inv 别名与 reciprocal 等价（同路径计算，差必为 0）
        TEST_CHECK((f.inv(prec) - inv).is_zero(), "inv alias == reciprocal");
    }
    // round 各模式（14314.25，Point=0 舍到整数）
    {
        BigFloat f(BigInt("114514"), 0);
        f <<= -3;  // 114514/8 = 14314.25
        BigFloat g(f);
        g.round(RoundMode::RoundHalfUp, 0, RoundRelativeTo::Point);
        TEST_CHECK(BigInt(g) == BigInt(14314), "round halfup 14314.25 -> 14314");
        BigFloat g2(f);
        g2.round(RoundMode::Ceil, 0, RoundRelativeTo::Point);
        TEST_CHECK(BigInt(g2) == BigInt(14315), "round ceil 14314.25 -> 14315");
        BigFloat g3(f);
        g3.round(RoundMode::Floor, 0, RoundRelativeTo::Point);
        TEST_CHECK(BigInt(g3) == BigInt(14314), "round floor 14314.25 -> 14314");
        BigFloat g4(f);
        g4.round(RoundMode::Truncate, 0, RoundRelativeTo::Point);
        TEST_CHECK(BigInt(g4) == BigInt(14314), "round trunc 14314.25 -> 14314");
        BigFloat g5(f);
        g5.round(RoundMode::RoundHalfUp, 1, RoundRelativeTo::Significant);
        TEST_CHECK(BigInt(g5) == BigInt(14314), "round significant 1 -> 14314");
    }
    // 全部舍掉（保留 0 个 limb）且需要进位的路径：ceil/floor/halfup 0.5 类
    {
        const auto rnd = [](double d, RoundMode mode) {
            BigFloat x(d);
            x.round(mode, 0, RoundRelativeTo::Significant);
            return BigInt(x);
        };
        TEST_CHECK(rnd(0.5, RoundMode::Ceil) == BigInt(1), "ceil 0.5 -> 1");
        TEST_CHECK(rnd(0.5, RoundMode::Floor) == BigInt(0), "floor 0.5 -> 0");
        TEST_CHECK(rnd(0.5, RoundMode::RoundHalfUp) == BigInt(1), "halfup 0.5 -> 1");
        TEST_CHECK(rnd(0.5, RoundMode::Truncate) == BigInt(0), "trunc 0.5 -> 0");
        TEST_CHECK(rnd(-0.5, RoundMode::Ceil) == BigInt(0), "ceil -0.5 -> 0");
        TEST_CHECK(rnd(-0.5, RoundMode::Floor) == BigInt(-1), "floor -0.5 -> -1");
        TEST_CHECK(rnd(-0.5, RoundMode::RoundHalfUp) == BigInt(-1), "halfup -0.5 -> -1");
        TEST_CHECK(rnd(0.25, RoundMode::RoundHalfUp) == BigInt(0), "halfup 0.25 -> 0");
        TEST_CHECK(rnd(-0.25, RoundMode::Ceil) == BigInt(0), "ceil -0.25 -> 0");
        TEST_CHECK(rnd(-0.25, RoundMode::Floor) == BigInt(-1), "floor -0.25 -> -1");
    }
}

static void test_big_numbers() {
    std::cout << "[big numbers]\n";
    // 跨 FFT/NTT 分发的大数除法与 get_pow_of_ten
    BigInt a("123456789012345678901234567890123456789012345678901234567890");
    BigInt b("98765432109876543210987654321");
    auto [Q, R] = a.divmod(b, RoundMode::Truncate);
    TEST_CHECK(Q * b + R == a, "big divmod identity");
    BigInt p = BigInt::get_pow_of_ten(100);
    TEST_CHECK(p.to_string().size() == 101 && p.to_string()[0] == '1', "10^100");
    BigInt a2("9999999999999999999999999999999999999999999999999999999999999999");
    BigInt b2("999999999999999999999999999999");
    auto [Q2, R2] = a2.divmod(b2, RoundMode::Truncate);
    TEST_CHECK(Q2 * b2 + R2 == a2, "big divmod identity 2");
}

auto main() -> int {
    test_boundary_chains();
    test_random_arithmetic();
    test_divmod();
    test_string_roundtrip();
    test_known_values();
    test_bigfloat();
    test_big_numbers();
    return bigint_test::summary("test_bigint");
}
