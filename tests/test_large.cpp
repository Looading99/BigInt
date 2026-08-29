// 大数测试：验证乘法分发（brute → FFT → NTT → SSA）在各级规模下的正确性。
//
// 策略：
//   1) 结构模式验证（数学上可手工推导的乘积 limb 布局，O(1) 期望构造，与库实现完全独立）：
//      (2^(64k)-1)^2、(1<<64k)^2、(2^(64k)-1)(2^(64k)+1)，取不同 k 覆盖各分发深度；
//   2) 中小规模（≤8192 limbs）与朴素 O(n²) 参考实现随机对拍；
//   3) 大数除法：结构模式（2^N-1 整除 2^M-1）、随机长除参考对拍、整除/商 1 场景；
//   4) get_pow_of_ten（FFT 倍增路径）已知值验证；
//   5) NTT 路径的回归（默认跳过，设置环境变量 BIGINT_TEST_NTT=1 启用）：
//      规模超出 FFT 上限（> 83886080 bits），单次运行耗时较长。
#include <cstdint>
#include <cstdlib>
#include <string>

// MSVC CRT 将 std::getenv 标记为 deprecated（建议 _dupenv_s）；
// 此处仅读取可选测试开关（BIGINT_TEST_NTT），使用标准函数即可。
#if defined(_MSC_VER)
#    pragma warning(disable : 4996)
#endif

#include "bigint/bigint.h"
#include "test_common.h"

namespace {

using bigint::BigInt;
using bigint::RoundMode;
using bigint_test::LimbVec;
using bigint_test::rand_limbs_exact;
using bigint_test::ref_add;
using bigint_test::ref_cmp;
using bigint_test::ref_divmod;
using bigint_test::ref_mul;
using bigint_test::ref_shr;
using bigint_test::ref_sub;
using bigint_test::rng;
using bigint_test::to_limbs;

// ---- 结构模式验证 ----

// (2^(64k) - 1)^2 = 2^(128k) - 2^(64k+1) + 1
//   limbs: [0]=1, [k]=0xFFFF...FE, [k+1..2k-1]=0xFF...F
static void test_square_all_ones(std::size_t k, const char* tag) {
    const BigInt a(LimbVec(k, UINT64_MAX));
    LimbVec      exp(2 * k, 0);
    exp[0]                = 1;
    exp[k]                = UINT64_MAX - 1;
    for (std::size_t i = k + 1; i < 2 * k; ++i)
        exp[i] = UINT64_MAX;
    TEST_CHECK(to_limbs(a * a) == exp, tag);
    // 差平方：(2^(64k)-1)(2^(64k)+1) = 2^(128k) - 1
    const BigInt b = a + BigInt(2);
    TEST_CHECK(to_limbs(a * b) == LimbVec(2 * k, UINT64_MAX), tag);
}

// (1 << 64k)^2 = 1 << 128k
static void test_square_pow2(std::size_t k, const char* tag) {
    const BigInt a = BigInt(1) << (64 * k);
    LimbVec      exp(2 * k + 1, 0);
    exp[2 * k] = 1;
    TEST_CHECK(to_limbs(a * a) == exp, tag);
}

static void test_mul_patterns() {
    std::cout << "[mul patterns]\n";
    // brute 边界附近（brute ≤ 16384 bits = 256 limbs）
    test_square_all_ones(2, "square all-ones k=2");
    test_square_all_ones(255, "square all-ones k=255 (brute)");
    test_square_all_ones(256, "square all-ones k=256 (brute boundary)");
    test_square_all_ones(257, "square all-ones k=257 (FFT entry)");
    test_square_all_ones(258, "square all-ones k=258 (FFT)");
    // FFT 深度递增
    test_square_all_ones(4096, "square all-ones k=4096");
    test_square_all_ones(65536, "square all-ones k=65536");
    test_square_all_ones(1310720, "square all-ones k=1310720 (FFT max)");
    // 单 bit 平方
    test_square_pow2(256, "square 2^(64*256)");
    test_square_pow2(1000, "square 2^(64*1000)");
    test_square_pow2(65536, "square 2^(64*65536)");
}

// ---- 中小规模随机对拍（ref_mul O(n²)）----

static void test_mul_random() {
    std::cout << "[mul random]\n";
    // 随机长度（brute 分发）
    for (int i = 0; i < 50; ++i) {
        const auto    a = rand_limbs_exact(1 + rng() % 250), b = rand_limbs_exact(1 + rng() % 250);
        const BigInt  A(a, rng() & 1), B(b, rng() & 1);
        const auto    exp = ref_mul(a, b);
        TEST_CHECK(to_limbs(A * B) == exp, "mul random brute");
    }
    // 边界与 FFT 内随机长度
    for (std::size_t n : {256u, 257u, 300u, 512u, 1000u}) {
        const auto   a = rand_limbs_exact(n), b = rand_limbs_exact(n);
        const BigInt A(a), B(b);
        const auto   exp = ref_mul(a, b);
        TEST_CHECK(to_limbs(A * B) == exp, "mul random boundary");
    }
    // 8192 limbs（ref_mul 约 6700 万次乘加，较慢，只做 1 次）
    {
        const auto   a = rand_limbs_exact(8192), b = rand_limbs_exact(8192);
        const BigInt A(a), B(b);
        const auto   exp = ref_mul(a, b);
        TEST_CHECK(to_limbs(A * B) == exp, "mul random 8192 limbs");
    }
    // 平方与交换律（库内自洽，大数）
    {
        const BigInt A(rand_limbs_exact(2048)), B(rand_limbs_exact(2048));
        TEST_CHECK(A * A == A * A, "square self");
        TEST_CHECK(A * B == B * A, "commutative");
        TEST_CHECK((A * B) * A == A * (B * A), "associative large");
    }
}

// ---- 大数除法 ----

// 2^(64k) - 1 整除 2^(64m) - 1（m | k）：商 = sum_{i=0}^{k/m-1} 2^(64*m*i)
static void test_div_pattern(std::size_t k, std::size_t m, const char* tag) {
    const BigInt a(LimbVec(k, UINT64_MAX));       // 2^(64k) - 1
    const BigInt b(LimbVec(m, UINT64_MAX));       // 2^(64m) - 1
    LimbVec      q_exp(k - m + 1, 0);             // 商最高位在 2^(64(k-m))
    for (std::size_t i = 0; i < k; i += m)
        q_exp[i] = 1;
    auto [Q, R] = a.divmod(b, RoundMode::Truncate);
    TEST_CHECK(to_limbs(Q) == q_exp && R.is_zero(), tag);
}

static void test_div_random() {
    std::cout << "[div random]\n";
    // 随机大除法对拍（二进制长除参考，规模折中）
    {
        const auto   a = rand_limbs_exact(900), b = rand_limbs_exact(400);
        const BigInt A(a, rng() & 1), B(b, rng() & 1);
        auto [q, r]  = ref_divmod(a, b);
        const int sQ = (A.sign() < 0) != (B.sign() < 0) ? -1 : 1;
        auto [Q, R]  = A.divmod(B, RoundMode::Truncate);
        TEST_CHECK(to_limbs(Q) == q && Q.sign() == sQ, "big divmod Q");
        TEST_CHECK(to_limbs(R) == r && R.sign() == (A.sign() < 0 ? -1 : 1), "big divmod R");
        TEST_CHECK(Q * B + R == A, "big divmod identity");
    }
    // 整除场景：a = q*b 精确整除
    for (int i = 0; i < 3; ++i) {
        const BigInt q(rand_limbs_exact(500), rng() & 1);
        const BigInt b(rand_limbs_exact(200), rng() & 1);
        const BigInt a = q * b;
        auto [Q, R] = a.divmod(b, RoundMode::Truncate);
        TEST_CHECK(Q == q && R.is_zero(), "exact division");
    }
    // 商为 1 的场景：a = b + r，0 < r < b
    for (int i = 0; i < 3; ++i) {
        const BigInt b(rand_limbs_exact(300), false);
        const BigInt r(rand_limbs_exact(100), false);
        if (r.is_zero() || r >= b)
            continue;
        const BigInt a = b + r;
        auto [Q, R] = a.divmod(b, RoundMode::Truncate);
        TEST_CHECK(Q == BigInt(1) && R == r, "quotient 1");
    }
}

// ---- 大数加减与位运算对拍 ----

static void test_add_sub_bitwise_large() {
    std::cout << "[add/sub/bitwise large]\n";
    for (int i = 0; i < 5; ++i) {
        const auto   a  = rand_limbs_exact(5000), b = rand_limbs_exact(3000);
        const bool   na = rng() & 1, nb = rng() & 1;
        const BigInt A(a, na), B(b, nb);
        // A+B：同号加、异号减（符号取大者）
        if (na == nb) {
            TEST_CHECK(to_limbs(A + B) == ref_add(a, b), "large add same sign");
        } else if (ref_cmp(a, b) >= 0) {
            TEST_CHECK(to_limbs(A + B) == ref_sub(a, b), "large add diff");
        } else {
            TEST_CHECK(to_limbs(A + B) == ref_sub(b, a), "large add diff 2");
        }
        // A-B = A+(-B)：异号加、同号减
        if (na != nb) {
            TEST_CHECK(to_limbs(A - B) == ref_add(a, b), "large sub diff sign");
        } else if (ref_cmp(a, b) >= 0) {
            TEST_CHECK(to_limbs(A - B) == ref_sub(a, b), "large sub same sign");
        } else {
            TEST_CHECK(to_limbs(A - B) == ref_sub(b, a), "large sub same sign 2");
        }
    }
    for (int i = 0; i < 3; ++i) {
        const auto   a = rand_limbs_exact(2000), b = rand_limbs_exact(2000);
        const BigInt A(a), B(b);
        LimbVec      e_and, e_or, e_xor;
        for (std::size_t j = 0; j < a.size(); ++j) {
            e_and.push_back(a[j] & b[j]);
            e_or.push_back(a[j] | b[j]);
            e_xor.push_back(a[j] ^ b[j]);
        }
        while (e_and.size() > 1 && e_and.back() == 0)
            e_and.pop_back();
        while (e_or.size() > 1 && e_or.back() == 0)
            e_or.pop_back();
        while (e_xor.size() > 1 && e_xor.back() == 0)
            e_xor.pop_back();
        TEST_CHECK(to_limbs(A & B) == e_and, "large and");
        TEST_CHECK(to_limbs(A | B) == e_or, "large or");
        TEST_CHECK(to_limbs(A ^ B) == e_xor, "large xor");
    }
}

// ---- get_pow_of_ten（FFT 倍增路径）----

static void test_get_pow_of_ten() {
    std::cout << "[get pow of ten]\n";
    for (uint32_t e : {0u, 1u, 19u, 100u, 1000u, 3000u}) {
        const std::string s = BigInt::get_pow_of_ten(e).to_string();
        TEST_CHECK(s.size() == static_cast<std::size_t>(e) + 1 && s[0] == '1'
                       && s.find_first_not_of('0', 1) == std::string::npos,
            "10^e string form");
    }
    // 大指数：10^50000（倍增 + FFT 乘法）
    {
        const std::string s = BigInt::get_pow_of_ten(50000).to_string();
        TEST_CHECK(s.size() == 50001 && s[0] == '1'
                       && s.find_first_not_of('0', 1) == std::string::npos,
            "10^50000 string form");
        // 与 (10^25000)^2 交叉验证（不同调用路径）
        const BigInt p = BigInt::get_pow_of_ten(25000);
        TEST_CHECK(BigInt::get_pow_of_ten(50000) == p * p, "10^50000 == (10^25000)^2");
    }
}

// ---- 可选 NTT 路径回归（默认跳过）----

static void test_ntt_optional() {
    if (std::getenv("BIGINT_TEST_NTT") == nullptr) {
        std::cout << "[ntt optional] skipped (set BIGINT_TEST_NTT=1 to enable)\n";
        return;
    }
    std::cout << "[ntt optional]\n";
    // 超出 FFT 上限 83886080 bits → NTT 分发（1310721 limbs * 64 bits）
    constexpr std::size_t K = 1310721;
    test_square_all_ones(K, "square all-ones k=1310721 (NTT)");
}

}  // namespace

auto main() -> int {
    test_mul_patterns();
    test_mul_random();
    test_div_pattern(4096, 256, "div 2^(64*4096)-1 / 2^(64*256)-1");
    test_div_pattern(65536, 1024, "div 2^(64*65536)-1 / 2^(64*1024)-1");
    test_div_random();
    test_add_sub_bitwise_large();
    test_get_pow_of_ten();
    test_ntt_optional();
    return bigint_test::summary("test_large");
}
