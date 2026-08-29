// BigInt 算术运算测试：加减乘（含符号组合）、复合赋值、自增自减（前后缀）、
// 与 64 位整数混合运算、进位/借位长链、一元运算符。
// 与独立参考实现（ref_add/ref_sub/ref_mul）随机对拍。
#include <cstdint>
#include <limits>
#include <string>

#include "bigint/bigint.h"
#include "test_common.h"

namespace {

using bigint::BigInt;
using bigint_test::LimbVec;
using bigint_test::check_matches;
using bigint_test::rand_limbs;
using bigint_test::ref_add;
using bigint_test::ref_cmp;
using bigint_test::ref_mul;
using bigint_test::ref_sub;
using bigint_test::rng;
using bigint_test::to_limbs;

// ---- 进位/借位长链与 2 的幂边界 ----

static void test_boundary_chains() {
    std::cout << "[boundary chains]\n";
    for (std::size_t k = 1; k <= 3; ++k) {
        const auto one  = LimbVec{1};
        auto       big  = LimbVec(k, UINT64_MAX);  // 2^(64k) - 1
        auto       pow2 = LimbVec(k + 1, 0);
        pow2[k]         = 1;  // 2^(64k)
        TEST_CHECK(ref_cmp(ref_add(big, one), pow2) == 0, "carry chain");
        TEST_CHECK(ref_cmp(ref_sub(pow2, one), big) == 0, "borrow chain");
        TEST_CHECK(ref_cmp(ref_sub(pow2, pow2), LimbVec{0}) == 0, "equal magnitude");
        // BigInt 层面对拍
        BigInt b(big);
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
    const auto x = LimbVec{5};
    const auto y = LimbVec{7};
    const BigInt X = BigInt(x);
    const BigInt Y = BigInt(y);
    TEST_CHECK((X - Y) == BigInt(-2), "5 - 7 == -2");
    TEST_CHECK((Y - X) == BigInt(2), "7 - 5 == 2");
    TEST_CHECK((-X - Y) == BigInt(-12), "-5 - 7 == -12");
    TEST_CHECK((-X + Y) == BigInt(2), "-5 + 7 == 2");
    TEST_CHECK((X + (-Y)) == BigInt(-2), "5 + (-7) == -2");
}

// ---- 随机加减乘对拍（绝对值 + 符号）----

static void test_random_arithmetic() {
    std::cout << "[random arithmetic]\n";
    for (std::size_t iter = 0; iter < 3000; ++iter) {
        const auto  a  = rand_limbs(3), b = rand_limbs(3);
        const bool  na = rng() & 1, nb = rng() & 1;
        const BigInt A(a, na), B(b, nb);
        const int   sa = na ? -1 : 1, sb = nb ? -1 : 1;

        // 期望：A+B、A-B（绝对值 + 符号）
        LimbVec ab, a_b;
        int     s_ab = 0, s_a_b = 0;
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
        TEST_CHECK(to_limbs(A + B) == ab && (A + B).sign() == (ref_cmp(ab, {0}) == 0 ? 0 : s_ab),
            "A+B random");
        TEST_CHECK(
            to_limbs(A - B) == a_b && (A - B).sign() == (ref_cmp(a_b, {0}) == 0 ? 0 : s_a_b),
            "A-B random");

        // 乘法（小规模走 brute）
        const auto exp_mul = ref_mul(a, b);
        const int  s_mul   = (sa != sb) ? -1 : 1;
        TEST_CHECK((A * B).sign() == (ref_cmp(exp_mul, {0}) == 0 ? 0 : s_mul)
                       && to_limbs(A * B) == exp_mul,
            "A*B random");

        // 64 位整数混合
        const uint64_t c  = rng();
        const auto     cc = LimbVec{c};
        if (c != 0) {
            // A+c：A 正 → |a|+c；A 负 → |a|>=c ? -(|a|-c) : +(c-|a|)
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
            TEST_CHECK(to_limbs(A + c) == e_sum
                           && (A + c).sign() == (ref_cmp(e_sum, {0}) == 0 ? 0 : s_sum),
                "A+c random");
            // A-c：A 负 → -(|a|+c)；A 正 → |a|>=c ? |a|-c : -(c-|a|)
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
            TEST_CHECK(to_limbs(A - c) == e_sub
                           && (A - c).sign() == (ref_cmp(e_sub, {0}) == 0 ? 0 : s_sub),
                "A-c random");
            // 恒等式
            TEST_CHECK(to_limbs((A - c) + c) == a, "A-c+c == A");
            TEST_CHECK(to_limbs((A + c) - c) == a, "A+c-c == A");
            TEST_CHECK(to_limbs(A * c) == ref_mul(a, cc), "A*c random");
        }

        // 复合赋值与运算符等价
        BigInt x = A;
        x += B;
        TEST_CHECK(x == A + B, "+= equals +");
        BigInt y = A;
        y -= B;
        TEST_CHECK(y == A - B, "-= equals -");
        BigInt z = A;
        z *= B;
        TEST_CHECK(z == A * B, "*= equals *");
        // 链式复合赋值
        BigInt chain = A;
        (chain += B) += A;
        TEST_CHECK(chain == A + B + A, "chained +=");
    }
}

// ---- 自增自减 ----

static void test_inc_dec() {
    std::cout << "[inc/dec]\n";
    // 前缀返回值是引用；后缀返回旧值
    BigInt x(5);
    TEST_CHECK(&(++x) == &x && x == BigInt(6), "prefix ++ returns ref");
    TEST_CHECK(&(--x) == &x && x == BigInt(5), "prefix -- returns ref");
    BigInt old = x++;
    TEST_CHECK(old == BigInt(5) && x == BigInt(6), "postfix ++ returns old");
    old = x--;
    TEST_CHECK(old == BigInt(6) && x == BigInt(5), "postfix -- returns old");
    // 跨零与符号翻转
    BigInt n(-1);
    ++n;
    TEST_CHECK(n == BigInt(0) && n.sign() == 0, "-1 ++ -> 0");
    ++n;
    TEST_CHECK(n == BigInt(1), "0 ++ -> 1");
    --n;
    --n;
    TEST_CHECK(n == BigInt(-1), "1 -- -- -> -1");
    // 大边界随机对拍：++ 正数绝对值 +1；负数绝对值 -1（-1++ == 0，0++ == 1）
    for (int i = 0; i < 300; ++i) {
        const auto  a  = rand_limbs(3);
        const bool  na = rng() & 1;
        BigInt      v(a, na);
        LimbVec     exp;
        int         s_exp = 0;
        if (!na) {
            exp   = ref_add(a, {1});
            s_exp = 1;
        } else if (ref_cmp(a, {0}) == 0) {
            exp   = {1};
            s_exp = 1;  // 0++ == 1
        } else if (ref_cmp(a, {1}) == 0) {
            exp   = {0};
            s_exp = 0;  // -1++ == 0
        } else {
            exp   = ref_sub(a, {1});
            s_exp = -1;  // 负数绝对值减 1
        }
        ++v;
        check_matches(v, exp, s_exp, "++ random");
        // -- 应用于 ++ 后的值：负数绝对值 +1；0 → -1；1 → 0；正数 >1 减 1
        LimbVec exp2;
        int     s_exp2 = 0;
        if (s_exp < 0) {
            exp2   = ref_add(exp, {1});
            s_exp2 = -1;
        } else if (ref_cmp(exp, {0}) == 0) {
            exp2   = {1};
            s_exp2 = -1;  // 0-- == -1
        } else if (ref_cmp(exp, {1}) == 0) {
            exp2   = {0};
            s_exp2 = 0;  // 1-- == 0
        } else {
            exp2   = ref_sub(exp, {1});
            s_exp2 = 1;
        }
        --v;
        check_matches(v, exp2, s_exp2, "-- random");
    }
}

// ---- 64 位边界值混合 ----

static void test_mixed_limits() {
    std::cout << "[mixed limits]\n";
    constexpr int64_t I64_MIN = std::numeric_limits<int64_t>::min();
    constexpr int64_t I64_MAX = std::numeric_limits<int64_t>::max();
    // 符号混合
    TEST_CHECK(BigInt(5) + I64_MIN == BigInt(5) - BigInt("9223372036854775808"),
        "5 + INT64_MIN");
    TEST_CHECK(BigInt(5) * I64_MIN == -BigInt("46116860184273879040"), "5 * INT64_MIN");
    TEST_CHECK(BigInt(I64_MIN) / -1 == BigInt("9223372036854775808"), "INT64_MIN / -1");
    TEST_CHECK(BigInt(I64_MIN) / 2 == -BigInt("4611686018427387904"), "INT64_MIN / 2");
    TEST_CHECK(BigInt(I64_MAX) + 1 == BigInt("9223372036854775808"), "INT64_MAX + 1");
    TEST_CHECK(BigInt(UINT64_MAX) + 1 == (BigInt(1) << 64), "UINT64_MAX + 1");
    TEST_CHECK(BigInt(1) - UINT64_MAX == BigInt(1) - BigInt(UINT64_MAX), "1 - UINT64_MAX");
    TEST_CHECK(BigInt(-3) * I64_MIN == BigInt("27670116110564327424"), "-3 * INT64_MIN");
    TEST_CHECK(BigInt(I64_MIN) - 1 == -BigInt("9223372036854775809"), "INT64_MIN - 1");
    // 随机有符号混合对拍
    for (int i = 0; i < 300; ++i) {
        const auto     c  = static_cast<int64_t>(rng());
        const auto     a  = rand_limbs(2);
        const bool     na = rng() & 1;
        const BigInt   A(a, na);
        const auto     cc = LimbVec{static_cast<uint64_t>(c < 0 ? -c : c)};
        const bool     nc = c < 0;
        // A + c（符号逻辑同 A+B 参考）
        int     s_sum = 0;
        LimbVec e_sum;
        if (na == nc) {
            e_sum = ref_add(a, cc);
            s_sum = na ? -1 : 1;
        } else if (ref_cmp(a, cc) >= 0) {
            e_sum = ref_sub(a, cc);
            s_sum = na ? -1 : 1;
        } else {
            e_sum = ref_sub(cc, a);
            s_sum = nc ? -1 : 1;
        }
        TEST_CHECK(to_limbs(A + c) == e_sum
                       && (A + c).sign() == (ref_cmp(e_sum, {0}) == 0 ? 0 : s_sum),
            "A + int64 random");
        // A * c
        TEST_CHECK(to_limbs(A * c) == ref_mul(a, cc), "A * int64 random");
        // 自增自减与复合
        BigInt v = A;
        v += c;
        TEST_CHECK(v == A + c, "+= int64 random");
    }
}

// ---- 一元运算符与已知值 ----

static void test_unary_and_known() {
    std::cout << "[unary and known]\n";
    // 一元 + / -
    BigInt p(7), z(0);
    TEST_CHECK(+p == p, "unary +");
    TEST_CHECK(-p == BigInt(-7), "unary -");
    TEST_CHECK(-z == z && (-z).sign() == 0, "-0 == 0");
    // 分配律/结合律抽查（小规模）
    for (int i = 0; i < 100; ++i) {
        const BigInt A(rand_limbs(2), rng() & 1), B(rand_limbs(2), rng() & 1),
            C(rand_limbs(2), rng() & 1);
        TEST_CHECK((A + B) * C == A * C + B * C, "distributive");
        TEST_CHECK((A + B) + C == A + (B + C), "associative +");
        TEST_CHECK((A * B) * C == A * (B * C), "associative *");
        TEST_CHECK(-(A + B) == -A - B, "-(A+B) == -A-B");
    }
    // 已知值
    TEST_CHECK(BigInt(2) * BigInt(2) == BigInt(4), "2*2");
    TEST_CHECK((BigInt(1) << 100) * (BigInt(1) << 100) == (BigInt(1) << 200), "2^100 * 2^100");
}

}  // namespace

auto main() -> int {
    test_boundary_chains();
    test_random_arithmetic();
    test_inc_dec();
    test_mixed_limits();
    test_unary_and_known();
    return bigint_test::summary("test_arithmetic");
}
