// BigInt 除法家族测试：divmod 四种舍入模式（随机对拍 + 已知值）、除零异常、
// check 参数、非法模式值、operator/= 与 unsigned_inplace_divmod（单 limb 除法）。
//
// 随机对拍的参考策略：先对绝对值做二进制长除得到截断商/余，
// 再按舍入模式调整商，并用 ref_mul/ref_sub 独立验证恒等式 |a| = |Q|*|b| ± |R|。
#include <array>
#include <cstdint>
#include <stdexcept>
#include <utility>

#include "bigint/bigint.h"
#include "test_common.h"

namespace {

using bigint::BigInt;
using bigint::RoundMode;
using bigint_test::LimbVec;
using bigint_test::check_matches;
using bigint_test::rand_limbs;
using bigint_test::ref_add;
using bigint_test::ref_cmp;
using bigint_test::ref_divmod;
using bigint_test::ref_mul;
using bigint_test::ref_sub;
using bigint_test::rng;

// 计算某模式下的期望 (商绝对值, 商符号, 余绝对值, 余符号)
static auto expected_divmod(const LimbVec& a, const LimbVec& b, bool na, bool nb, RoundMode mode)
    -> std::tuple<LimbVec, int, LimbVec, int> {
    auto [q0, r0] = ref_divmod(a, b);  // 截断：|a| = q0*|b| + r0，0 <= r0 < |b|
    LimbVec q_abs = q0;
    const bool neg_q = na != nb;
    // 按模式调整商（向远离/靠近零方向取整）
    if (ref_cmp(r0, {0}) != 0) {
        if (mode == RoundMode::Floor && neg_q) {
            q_abs = ref_add(q_abs, {1});  // 商向 -∞
        } else if (mode == RoundMode::Ceil && !neg_q) {
            q_abs = ref_add(q_abs, {1});  // 商向 +∞
        } else if (mode == RoundMode::RoundHalfUp && ref_cmp(ref_mul(r0, {2}), b) >= 0) {
            q_abs = ref_add(q_abs, {1});  // 远离零
        }
    }
    // 余数：R = a - Q*b → |R| = ||a| - |Q|*|b||，符号与被减数一致
    const auto  qb   = ref_mul(q_abs, b);
    const int   cmp  = ref_cmp(a, qb);
    const bool  r_neg = (cmp < 0) != na;  // R 与 a 异号当且仅当 |a| < |Q|*|b|
    const auto  r_abs = cmp >= 0 ? ref_sub(a, qb) : ref_sub(qb, a);
    const int   s_q   = ref_cmp(q_abs, {0}) == 0 ? 0 : (neg_q ? -1 : 1);
    const int   s_r   = ref_cmp(r_abs, {0}) == 0 ? 0 : (r_neg ? -1 : 1);
    return {q_abs, s_q, r_abs, s_r};
}

// ---- 随机对拍：四种舍入模式 ----

static void test_random_divmod() {
    std::cout << "[random divmod]\n";
    const std::array modes{
        RoundMode::Truncate, RoundMode::Floor, RoundMode::Ceil, RoundMode::RoundHalfUp};
    for (std::size_t iter = 0; iter < 2000; ++iter) {
        const auto  a  = rand_limbs(3), b = rand_limbs(2);
        if (ref_cmp(b, {0}) == 0)
            continue;
        const bool    na = rng() & 1, nb = rng() & 1;
        const BigInt  A(a, na), B(b, nb);
        const RoundMode mode = modes[iter % 4];
        const auto [q_abs, s_q, r_abs, s_r] = expected_divmod(a, b, na, nb, mode);
        auto [Q, R] = A.divmod(B, mode);
        check_matches(Q, q_abs, s_q, "divmod Q random");
        check_matches(R, r_abs, s_r, "divmod R random");
        // 独立恒等式：|a| == |Q|*|b| + |R| 或 |a| == |Q|*|b| - |R|
        const auto qb = ref_mul(q_abs, b);
        const bool ok = (s_r == (na ? -1 : 1)) ? ref_cmp(ref_add(qb, r_abs), a) == 0
                                               : ref_cmp(ref_sub(qb, r_abs), a) == 0;
        TEST_CHECK(ok, "|a| == |Q|*|b| ± |R|");
        // 余数界：Truncate/Floor/Ceil |R| < |b|；RoundHalfUp |2R| <= |b|
        if (mode == RoundMode::RoundHalfUp)
            TEST_CHECK(ref_cmp(ref_mul(r_abs, {2}), b) <= 0, "halfup |2R| <= |b|");
        else
            TEST_CHECK(ref_cmp(r_abs, b) < 0, "|R| < |b|");
    }
}

// ---- 已知值（手工验算的舍入行为）----

static void test_known_values() {
    std::cout << "[known values]\n";
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
    // 负除数
    auto [Q7, R7] = BigInt(-17).divmod(BigInt(-5), RoundMode::Floor);
    TEST_CHECK(Q7 == BigInt(3) && R7 == BigInt(-2), "-17 divmod -5 floor");
    auto [Q8, R8] = BigInt(-17).divmod(BigInt(-5), RoundMode::Ceil);
    TEST_CHECK(Q8 == BigInt(4) && R8 == BigInt(3), "-17 divmod -5 ceil");
    auto [Q9, R9] = BigInt(17).divmod(BigInt(-5), RoundMode::Floor);
    TEST_CHECK(Q9 == BigInt(-4) && R9 == BigInt(-3), "17 divmod -5 floor");
    auto [Qa, Ra] = BigInt(17).divmod(BigInt(-5), RoundMode::Ceil);
    TEST_CHECK(Qa == BigInt(-3) && Ra == BigInt(2), "17 divmod -5 ceil");
    // 0.5 舍入边界（halfup 远离零）
    auto [Qb, Rb] = BigInt(15).divmod(BigInt(2), RoundMode::RoundHalfUp);
    TEST_CHECK(Qb == BigInt(8) && Rb == BigInt(-1), "15 divmod 2 halfup");
    auto [Qc, Rc] = BigInt(-15).divmod(BigInt(2), RoundMode::RoundHalfUp);
    TEST_CHECK(Qc == BigInt(-8) && Rc == BigInt(1), "-15 divmod 2 halfup");
    auto [Qd, Rd] = BigInt(14).divmod(BigInt(2), RoundMode::RoundHalfUp);
    TEST_CHECK(Qd == BigInt(7) && Rd == BigInt(0), "14 divmod 2 halfup");
    // 整除
    auto [Qe, Re] = BigInt(-20).divmod(BigInt(5), RoundMode::Floor);
    TEST_CHECK(Qe == BigInt(-4) && Re == BigInt(0), "-20 divmod 5 floor");
    // 被除数为 0
    auto [Qf, Rf] = BigInt(0).divmod(BigInt(7), RoundMode::Floor);
    TEST_CHECK(Qf == BigInt(0) && Rf == BigInt(0), "0 divmod 7");
}

// ---- 异常与参数边界 ----

static void test_errors_and_args() {
    std::cout << "[errors and args]\n";
    // 除零
    bool threw = false;
    try {
        static_cast<void>(BigInt(10).divmod(BigInt(0)));
    } catch (const std::domain_error&) {
        threw = true;
    }
    TEST_CHECK(threw, "divmod by zero throws");
    threw = false;
    try {
        BigInt x(10);
        x /= 0;
    } catch (const std::domain_error&) {
        threw = true;
    }
    TEST_CHECK(threw, "operator/= by zero throws");
    // 非法模式值视为 Truncate
    const auto mode = static_cast<RoundMode>(99);
    auto [Q, R]     = BigInt(-17).divmod(BigInt(5), mode);
    TEST_CHECK(Q == BigInt(-3) && R == BigInt(-2), "invalid mode == truncate");
    // check=false：不调整，但恒等式 Q*b + R == A 恒成立
    for (int i = 0; i < 200; ++i) {
        const auto    a = rand_limbs(3), b = rand_limbs(2);
        if (ref_cmp(b, {0}) == 0)
            continue;
        const BigInt A(a, rng() & 1), B(b, rng() & 1);
        auto [Q2, R2] = A.divmod(B, RoundMode::Floor, false);
        TEST_CHECK(Q2 * B + R2 == A, "check=false identity");
    }
}

// ---- operator/= 与 unsigned_inplace_divmod（单 limb 除法）----

static void test_single_limb_division() {
    std::cout << "[single limb division]\n";
    // unsigned_inplace_divmod 与通用 divmod 交叉验证
    for (int i = 0; i < 500; ++i) {
        const auto    a = rand_limbs(3);
        const uint64_t c = rng();
        if (c == 0)
            continue;
        BigInt q(a, rng() & 1);
        const bool  neg = q.sign() < 0;
        const uint64_t r = q.unsigned_inplace_divmod(c);
        // 期望：q == trunc(a/c)，r == a%c（绝对值语义）
        const auto [qq, rr] = ref_divmod(a, {c});
        check_matches(q, qq, neg ? -1 : 1, "unsigned_inplace_divmod Q");
        TEST_CHECK(r == rr[0], "unsigned_inplace_divmod R");
    }
    // 边界
    {
        BigInt q(123);
        TEST_CHECK(q.unsigned_inplace_divmod(1) == 0 && q == BigInt(123), "divmod by 1");
        BigInt q2(0);
        TEST_CHECK(q2.unsigned_inplace_divmod(7) == 0 && q2.is_zero(), "0 divmod by c");
        BigInt q3(UINT64_MAX);
        TEST_CHECK(q3.unsigned_inplace_divmod(UINT64_MAX) == 0 && q3 == BigInt(1),
            "UINT64_MAX divmod UINT64_MAX");
    }
    // operator/= 与 divmod 交叉验证
    for (int i = 0; i < 300; ++i) {
        const auto    a = rand_limbs(2);
        const auto    c = static_cast<int64_t>(rng());
        if (c == 0)
            continue;
        const BigInt A(a, rng() & 1);
        BigInt       q = A;
        q /= c;
        const auto [Q, R] = A.divmod(BigInt(c));
        TEST_CHECK(q == Q, "operator/= equals divmod");
        TEST_CHECK(q * BigInt(c) + R == A, "/= identity");
    }
}

}  // namespace

auto main() -> int {
    test_random_divmod();
    test_known_values();
    test_errors_and_args();
    test_single_limb_division();
    return bigint_test::summary("test_divmod");
}
