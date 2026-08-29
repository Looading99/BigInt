// BigInt 比较运算测试：<=> 全序（随机对拍 + 传递性）、==、与字面量 0 的快速比较、
// compare_abs、符号/长度查询的一致性。
#include <array>
#include <compare>

#include "bigint/bigint.h"
#include "test_common.h"

namespace {

using bigint::BigInt;
using bigint_test::LimbVec;
using bigint_test::rand_limbs;
using bigint_test::ref_cmp;
using bigint_test::rng;

// 期望的序：-1/0/1（零的符号按裁剪后的实际语义：|x|=0 → 0）
static auto expected_cmp(const LimbVec& a, const LimbVec& b, bool na, bool nb) -> int {
    const int sa = ref_cmp(a, {0}) == 0 ? 0 : (na ? -1 : 1);
    const int sb = ref_cmp(b, {0}) == 0 ? 0 : (nb ? -1 : 1);
    if (sa != sb)
        return sa < sb ? -1 : 1;
    if (sa == 0)
        return 0;
    return sa * ref_cmp(a, b);
}

// ---- 随机对拍 <=> 与 == ----

static void test_spaceship_random() {
    std::cout << "[spaceship random]\n";
    for (int i = 0; i < 3000; ++i) {
        const auto   a = rand_limbs(3), b = rand_limbs(3);
        const bool   na = rng() & 1, nb = rng() & 1;
        const BigInt A(a, na), B(b, nb);
        const int    exp = expected_cmp(a, b, na, nb);
        TEST_CHECK((A <=> B) == (exp <=> 0), "A<=>B random");
        TEST_CHECK((B <=> A) == ((-exp) <=> 0), "B<=>A random");
        TEST_CHECK((A == B) == (exp == 0), "A==B random");
        TEST_CHECK((A != B) == (exp != 0), "A!=B random");
        // 与符号组合的一致性
        TEST_CHECK((A < B) == (exp < 0), "A<B random");
        TEST_CHECK((A > B) == (exp > 0), "A>B random");
        TEST_CHECK((A <= B) == (exp <= 0), "A<=B random");
        TEST_CHECK((A >= B) == (exp >= 0), "A>=B random");
    }
}

// ---- 与字面量 0 的快速比较 ----

static void test_literal_zero() {
    std::cout << "[literal zero]\n";
    const std::array vals{
        BigInt(0), BigInt(1), BigInt(-1), BigInt(123), BigInt(-456), BigInt(1) << 100};
    for (const auto& x : vals) {
        const int s = x.sign();
        TEST_CHECK((x == 0) == (s == 0), "x==0");
        TEST_CHECK((0 == x) == (s == 0), "0==x");
        TEST_CHECK((x != 0) == (s != 0), "x!=0");
        TEST_CHECK((x < 0) == (s < 0), "x<0");
        TEST_CHECK((x > 0) == (s > 0), "x>0");
        TEST_CHECK((x <= 0) == (s <= 0), "x<=0");
        TEST_CHECK((x >= 0) == (s >= 0), "x>=0");
        TEST_CHECK((0 < x) == (s > 0), "0<x");
        TEST_CHECK((0 > x) == (s < 0), "0>x");
    }
}

// ---- compare_abs ----

static void test_compare_abs() {
    std::cout << "[compare abs]\n";
    for (int i = 0; i < 1000; ++i) {
        const auto a = rand_limbs(3), b = rand_limbs(3);
        // 符号不影响绝对值比较
        const BigInt A(a, rng() & 1), B(b, rng() & 1);
        const int    exp = ref_cmp(a, b);
        TEST_CHECK(BigInt::compare_abs(A, B) == (exp <=> 0), "compare_abs random");
        TEST_CHECK(BigInt::compare_abs(B, A) == ((-exp) <=> 0), "compare_abs swapped");
    }
    // 自身与零
    TEST_CHECK(BigInt::compare_abs(BigInt(5), BigInt(-5)) == std::strong_ordering::equal,
        "compare_abs equal mag");
    TEST_CHECK(BigInt::compare_abs(BigInt(0), BigInt(1)) == std::strong_ordering::less,
        "compare_abs 0 < 1");
}

// ---- 全序传递性与反自反性 ----

static void test_order_axioms() {
    std::cout << "[order axioms]\n";
    // 反自反：A < A 恒假
    for (int i = 0; i < 200; ++i) {
        const BigInt A(rand_limbs(2), rng() & 1);
        TEST_CHECK(!(A < A) && A <= A && A == A && !(A != A), "reflexive");
    }
    // 传递性：随机三数
    for (int i = 0; i < 300; ++i) {
        const BigInt A(rand_limbs(2), rng() & 1), B(rand_limbs(2), rng() & 1),
            C(rand_limbs(2), rng() & 1);
        if (A <= B && B <= C)
            TEST_CHECK(A <= C, "transitive");
        if (A < B && B < C)
            TEST_CHECK(A < C, "transitive strict");
    }
    // 已知值排序集合
    const std::array sorted{BigInt(1) << 200,
        BigInt(100),
        BigInt(2),
        BigInt(1),
        BigInt(0),
        BigInt(-1),
        BigInt(-2),
        BigInt(-100),
        -(BigInt(1) << 200)};
    for (std::size_t i = 0; i < sorted.size(); ++i)
        for (std::size_t j = i + 1; j < sorted.size(); ++j)
            TEST_CHECK(sorted[i] > sorted[j], "sorted order");
}

// ---- 查询一致性 ----

static void test_queries() {
    std::cout << "[queries]\n";
    for (int i = 0; i < 300; ++i) {
        const auto   a  = rand_limbs(3);
        const bool   na = rng() & 1;
        const BigInt A(a, na);
        TEST_CHECK(A.len() == a.size(), "len == limbs size");
        TEST_CHECK(A.sign() == (ref_cmp(a, {0}) == 0 ? 0 : (na ? -1 : 1)), "sign");
        TEST_CHECK(A.is_zero() == (ref_cmp(a, {0}) == 0), "is_zero");
        TEST_CHECK(static_cast<bool>(A) == !A.is_zero(), "operator bool");
        TEST_CHECK(A.get_data().size() == A.len(), "get_data size");
    }
}

}  // namespace

auto main() -> int {
    test_spaceship_random();
    test_literal_zero();
    test_compare_abs();
    test_order_axioms();
    test_queries();
    return bigint_test::summary("test_compare");
}
