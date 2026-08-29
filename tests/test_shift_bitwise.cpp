// BigInt 移位与位运算测试：左/右移（含负偏移与超大偏移）、复合移位、
// 位与/或/异或（忽略符号）、bitwise_not（含指定长度）、与 2 的幂乘除等价性。
#include <cstdint>

#include "bigint/bigint.h"
#include "test_common.h"

namespace {

using bigint::BigInt;
using bigint_test::LimbVec;
using bigint_test::rand_limbs;
using bigint_test::ref_add;
using bigint_test::ref_cmp;
using bigint_test::ref_shl;
using bigint_test::ref_shr;
using bigint_test::rng;
using bigint_test::to_limbs;

// ---- 移位 ----

static void test_shifts() {
    std::cout << "[shifts]\n";
    for (int i = 0; i < 1000; ++i) {
        const auto    a  = rand_limbs(3);
        const bool    na = rng() & 1;
        const BigInt  A(a, na);
        const int64_t sh = static_cast<int64_t>(rng() % 400) - 200;  // [-200, 199]
        // 负偏移 = 反向移位
        if (sh >= 0) {
            TEST_CHECK(to_limbs(A << sh) == ref_shl(a, static_cast<std::size_t>(sh)),
                "A<<sh random");
            TEST_CHECK(to_limbs(A >> sh) == ref_shr(a, static_cast<std::size_t>(sh)),
                "A>>sh random");
        } else {
            const auto s = static_cast<std::size_t>(-sh);
            TEST_CHECK(to_limbs(A << sh) == ref_shr(a, s), "A<<(-sh) == A>>sh");
            TEST_CHECK(to_limbs(A >> sh) == ref_shl(a, s), "A>>(-sh) == A<<sh");
        }
        // 负数移位保留符号（忽略符号语义）
        if (na) {
            TEST_CHECK((A << sh) == -(BigInt(a) << sh), "neg shift keeps sign");
            TEST_CHECK((A >> sh) == -(BigInt(a) >> sh), "neg shift keeps sign (>>)");
        }
        // 与 2 的幂乘除等价：x << k == x * 2^k；x >> k == x / 2^k
        if (sh > 0) {
            const BigInt pow2 = BigInt(1) << sh;
            TEST_CHECK((A << sh) == A * pow2, "shift == mul by 2^k");
            TEST_CHECK((A >> sh) == A.divmod(pow2).first, "shift == div by 2^k");
        }
    }
    // 0 偏移
    const BigInt z(12345);
    TEST_CHECK((z << 0) == z && (z >> 0) == z, "shift by 0");
    // 超大偏移：右移清空，左移成 2^k
    TEST_CHECK((BigInt(12345) >> 10000).is_zero(), "big >> -> 0");
    TEST_CHECK(to_limbs(BigInt(1) << 10000) == ref_shl({1}, 10000), "big << limbs");
    // 复合移位
    BigInt c(1);
    c <<= 70;
    TEST_CHECK(c == (BigInt(1) << 70), "<<= compound");
    c >>= 70;
    TEST_CHECK(c == BigInt(1), ">>= compound");
    c <<= -3;  // 负偏移复合
    TEST_CHECK(c == BigInt(0), "<<= negative -> 0");
}

// ---- 位运算（忽略符号）----

static void test_bitwise() {
    std::cout << "[bitwise]\n";
    for (int i = 0; i < 1500; ++i) {
        const auto   a  = rand_limbs(3), b = rand_limbs(3);
        const bool   na = rng() & 1, nb = rng() & 1;
        const BigInt A(a, na), B(b, nb);
        LimbVec      e_and, e_or, e_xor;
        for (std::size_t j = 0; j < std::max<std::size_t>(a.size(), b.size()); ++j) {
            const uint64_t x = j < a.size() ? a[j] : 0, y = j < b.size() ? b[j] : 0;
            e_and.push_back(x & y);
            e_or.push_back(x | y);
            e_xor.push_back(x ^ y);
        }
        while (e_and.size() > 1 && e_and.back() == 0)
            e_and.pop_back();
        while (e_or.size() > 1 && e_or.back() == 0)
            e_or.pop_back();
        while (e_xor.size() > 1 && e_xor.back() == 0)
            e_xor.pop_back();
        TEST_CHECK(to_limbs(A & B) == e_and, "A&B random");
        TEST_CHECK(to_limbs(A | B) == e_or, "A|B random");
        TEST_CHECK(to_limbs(A ^ B) == e_xor, "A^B random");
        // 复合赋值等价
        BigInt x = A;
        x &= B;
        TEST_CHECK(x == (A & B), "&= equals &");
        BigInt y = A;
        y |= B;
        TEST_CHECK(y == (A | B), "|= equals |");
        BigInt z = A;
        z ^= B;
        TEST_CHECK(z == (A ^ B), "^= equals ^");
    }
    // 幂等/吸收律
    {
        const BigInt A(rand_limbs(3), rng() & 1);
        TEST_CHECK((A & A) == A, "A&A == A");
        TEST_CHECK((A | A) == A, "A|A == A");
        TEST_CHECK((A ^ A).is_zero(), "A^A == 0");
        const BigInt Z(0);
        TEST_CHECK((A & Z).is_zero() && (A | Z) == A && (A ^ Z) == A, "with zero");
        // 已知值
        TEST_CHECK((BigInt(0b1100) & BigInt(0b1010)) == BigInt(0b1000), "1100&1010");
        TEST_CHECK((BigInt(0b1100) | BigInt(0b1010)) == BigInt(0b1110), "1100|1010");
        TEST_CHECK((BigInt(0b1100) ^ BigInt(0b1010)) == BigInt(0b0110), "1100^1010");
    }
}

// bitwise_not 参考：对低 len 个 limb 取反（len=0 用自身长度），高位补 0 再取反
static auto ref_not(const LimbVec& v, std::size_t len) -> LimbVec {
    const std::size_t n   = len ? len : v.size();
    LimbVec           res(n);
    for (std::size_t i = 0; i < n; ++i)
        res[i] = ~(i < v.size() ? v[i] : 0);
    while (res.size() > 1 && res.back() == 0)
        res.pop_back();
    return res;
}

// ---- bitwise_not ----

static void test_bitwise_not() {
    std::cout << "[bitwise not]\n";
    for (int i = 0; i < 500; ++i) {
        const auto   a  = rand_limbs(3);
        const bool   na = rng() & 1;
        const BigInt A(a, na);
        const std::size_t len = rng() % 8;  // 0 = 自身长度
        const int  s0   = A.sign();          // 操作前实际符号（零的负号已被裁剪）
        BigInt     x    = A;
        x.bitwise_not(len);
        TEST_CHECK(to_limbs(x) == ref_not(a, len), "bitwise_not random");
        TEST_CHECK(x.sign() == 0 || x.sign() == (s0 != 0 ? s0 : 1), "not keeps sign");
    }
    // 双重取反复原（仅当 ~x 无前导零裁剪时成立：x 最高 limb 非全 1）
    for (int i = 0; i < 100; ++i) {
        const auto  a = rand_limbs(2);
        if (a.back() == UINT64_MAX)
            continue;
        BigInt      x(a, rng() & 1);
        const BigInt orig = x;
        x.bitwise_not();
        x.bitwise_not();
        TEST_CHECK(x == orig, "~~x == x");
    }
    // 已知值
    BigInt k(0);
    k.bitwise_not();
    TEST_CHECK(to_limbs(k) == LimbVec{UINT64_MAX}, "~0 == UINT64_MAX");
    BigInt k2(0);
    k2.bitwise_not(2);
    const LimbVec two_ones{UINT64_MAX, UINT64_MAX};
    TEST_CHECK(to_limbs(k2) == two_ones, "~0 len=2");
}

}  // namespace

auto main() -> int {
    test_shifts();
    test_bitwise();
    test_bitwise_not();
    return bigint_test::summary("test_shift_bitwise");
}
