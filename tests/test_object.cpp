// BigInt/BigFloat 对象语义测试：拷贝构造/赋值独立性、移动语义（移动后源 reset 为 0
// 且对源调用任何公开 API 安全）、自移动赋值、reset/flip_sign/remove_sign/abs/operator bool。
#include "bigint/bigint.h"
#include "test_common.h"

namespace {

using bigint::BigFloat;
using bigint::BigInt;
using bigint::abs;
using bigint_test::rand_limbs;
using bigint_test::rand_limbs_exact;
using bigint_test::rng;

// 对对象调用一组公开 API，验证移动后的源对象状态安全（不崩溃、语义为 0）
static auto exercise(const BigInt& x) -> void {
    static_cast<void>(x.len());
    static_cast<void>(x.sign());
    static_cast<void>(x.is_zero());
    static_cast<void>(x.to_string());
    static_cast<void>(x.to_string(true));
    static_cast<void>(x.get_data());
    static_cast<void>(static_cast<bool>(x));
    static_cast<void>(x <=> 0);
    static_cast<void>(x == 0);
    static_cast<void>(x + x);
    static_cast<void>(x - x);
    static_cast<void>(x * x);
    static_cast<void>((x << 1) >> 1);
    static_cast<void>(x & x);
}

static auto exercise(const BigFloat& x) -> void {
    static_cast<void>(x.len());
    static_cast<void>(x.sign());
    static_cast<void>(x.is_zero());
    static_cast<void>(x.to_double());
    static_cast<void>(x.to_string());
    static_cast<void>(x.get_data());
    static_cast<void>(x.get_point_pos());
    if (!x.is_zero())  // 移动后的源为 0，reciprocal 会抛除零异常
        static_cast<void>(x.reciprocal(4));
    static_cast<void>(x + x);
    static_cast<void>(x - x);
}

// ---- 拷贝语义 ----

static void test_copy() {
    std::cout << "[copy]\n";
    for (int i = 0; i < 200; ++i) {
        const auto a  = rand_limbs(3);
        const bool na = rng() & 1;
        BigInt     src(a, na);
        BigInt     cpy(src);  // 拷贝构造
        BigInt     cpy2(BigInt(0));
        cpy2 = src;  // 拷贝赋值
        TEST_CHECK(cpy == src && cpy2 == src, "copy equal");
        // 修改副本不影响源
        cpy.flip_sign();
        cpy2 += BigInt(1);
        TEST_CHECK(src == BigInt(a, na), "copy independent");
    }
    // 深拷贝验证：共享 vector 内容不共享存储
    BigInt x(rand_limbs_exact(2), false), y(x);
    // NOLINTNEXTLINE(performance-unnecessary-copy-initialization) —— 故意测拷贝构造
    BigInt z = x;
    y.reset();
    TEST_CHECK(!z.is_zero(), "copy not aliased");
    // BigFloat 拷贝
    BigFloat f(rand_limbs(2), rng() & 1, -64);
    // NOLINTNEXTLINE(performance-unnecessary-copy-initialization) —— 故意测拷贝构造
    BigFloat g(f), h(0);
    h = f;
    TEST_CHECK(
        g.get_data() == f.get_data() && g.get_point_pos() == f.get_point_pos(), "BigFloat copy");
    TEST_CHECK(h.get_data() == f.get_data() && h.get_point_pos() == f.get_point_pos(),
        "BigFloat copy assign");
}

// ---- 移动语义 ----

static void test_move() {
    std::cout << "[move]\n";
    for (int i = 0; i < 200; ++i) {
        const auto a  = rand_limbs(3);
        const bool na = rng() & 1;
        // 移动构造
        BigInt src1(a, na);
        BigInt dst1(std::move(src1));
        TEST_CHECK(dst1 == BigInt(a, na), "move ctor value");
        // NOLINTNEXTLINE —— 故意验证移动后源的状态（应为 0）
        TEST_CHECK(src1.is_zero() && src1.sign() == 0 && src1.len() == 1, "moved-from is 0");
        // 故意对移动后源调用 API 验证安全性（clang-analyzer 的 moved-from 告警属预期行为）
        exercise(src1);  // NOLINT 对移动后源调用 API 安全
        // 移动赋值
        BigInt src2(a, na);
        BigInt dst2(999);
        dst2 = std::move(src2);
        TEST_CHECK(dst2 == BigInt(a, na), "move assign value");
        // NOLINTNEXTLINE —— 故意验证移动后源的状态（应为 0）
        TEST_CHECK(src2.is_zero(), "moved-from assign is 0");
        // 移动后源可继续使用
        src2 = BigInt(42);  // NOLINT 移动后源可重新赋值
        TEST_CHECK(src2 == BigInt(42), "moved-from reusable");
    }
    // 自移动赋值：对象不变
    {
        BigInt       self(rand_limbs(2), rng() & 1);
        const BigInt before = self;
        self                = std::move(self);
        TEST_CHECK(self == before, "self move assign no-op");
    }
    // BigFloat 移动
    {
        BigFloat   f(rand_limbs(2), rng() & 1, -64);
        const auto d0 = f.get_data();
        const auto p0 = f.get_point_pos();
        BigFloat   g(std::move(f));
        TEST_CHECK(g.get_data() == d0 && g.get_point_pos() == p0, "BigFloat move ctor");
        TEST_CHECK(f.is_zero() && f.sign() == 0, "BigFloat moved-from is 0");
        exercise(f);
        BigFloat h(7);
        h = std::move(g);
        TEST_CHECK(h.get_data() == d0 && h.get_point_pos() == p0, "BigFloat move assign");
        TEST_CHECK(g.is_zero(), "BigFloat moved-from assign is 0");
    }
    // 从 BigInt 移动构造 BigFloat：源被重置（offset 单位 bit，64 = 1 个 limb）
    {
        BigInt     x(rand_limbs_exact(2), false);
        const auto xv = x;
        BigFloat   f(std::move(x), 64);
        // f 的值 = xv × 2^64（整数），截断成 BigInt 精确验证
        TEST_CHECK(BigInt(f) == (xv << 64), "BigFloat from BigInt&&");
        TEST_CHECK(x.is_zero(), "BigInt source reset after move");
    }
}

// ---- reset / 符号操作 / abs ----

static void test_reset_and_sign_ops() {
    std::cout << "[reset and sign ops]\n";
    for (int i = 0; i < 200; ++i) {
        const auto a  = rand_limbs(2);
        const bool na = rng() & 1;
        BigInt     x(a, na);
        x.reset();
        TEST_CHECK(x.is_zero() && x.sign() == 0 && x.len() == 1, "reset -> 0");
        x           = BigInt(a, na);
        const int s = x.sign();
        x.flip_sign();
        TEST_CHECK(x.sign() == (s == 0 ? 0 : -s), "flip_sign");
        x.flip_sign();
        TEST_CHECK(x.sign() == s, "flip_sign twice");
        x.remove_sign();
        TEST_CHECK(x.sign() >= 0, "remove_sign");
        // abs 不修改原对象
        const BigInt orig(a, na);
        const BigInt ax = abs(orig);
        TEST_CHECK(ax == BigInt(a, false), "abs value");
        TEST_CHECK(orig == BigInt(a, na), "abs leaves source");
    }
    // 0 的符号操作
    BigInt z(0);
    z.flip_sign();
    TEST_CHECK(z.is_zero() && z.sign() == 0, "flip_sign on 0");
    // BigFloat 符号操作（BigFloat 不支持比较，用作差判断）
    BigFloat f(-12345);
    TEST_CHECK((abs(f) - BigFloat(12345)).is_zero(), "BigFloat abs");
    f.flip_sign();
    TEST_CHECK(f.sign() == 1, "BigFloat flip_sign");
    f.remove_sign();
    TEST_CHECK(f.sign() == 1, "BigFloat remove_sign on positive");
    // operator bool（仅 BigInt）
    TEST_CHECK(static_cast<bool>(BigInt(1)) && !static_cast<bool>(BigInt(0)), "bool BigInt");
}

}  // namespace

auto main() -> int {
    test_copy();
    test_move();
    test_reset_and_sign_ops();
    return bigint_test::summary("test_object");
}
