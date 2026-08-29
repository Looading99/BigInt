// BigFloat 全功能测试：double 往返（含子正规数/异常值）、构造（整型/字符串/limb/BigInt + offset）、
// 移位、加减/乘法（不同小数点位）与 limb 级独立参考对拍、mul(precision) 截断语义、
// round 四模式 × 两种相对基准（已知值 + 随机对拍）、reciprocal/inv、转 BigInt 四种舍入、字符串输出。
#include <bit>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <numbers>
#include <sstream>
#include <stdexcept>
#include <string>

#include "bigint/bigint.h"
#include "test_common.h"

namespace {

using bigint::BigFloat;
using bigint::BigInt;
using bigint::Digits;
using bigint::RoundMode;
using bigint::RoundRelativeTo;
using bigint_test::LimbVec;
using bigint_test::check_matches;
using bigint_test::check_ref_float;
using bigint_test::rand_limbs;
using bigint_test::ref_cmp;
using bigint_test::ref_float_add_sub;
using bigint_test::ref_float_mul;
using bigint_test::ref_float_round;
using bigint_test::ref_float_to_bigint_abs;
using bigint_test::ref_mul;
using bigint_test::RefFloat;
using bigint_test::rng;
using bigint_test::to_limbs;

// ---- double 往返 ----

static void test_double_roundtrip() {
    std::cout << "[double roundtrip]\n";
    const std::array vals{0.0,
        1.0,
        -1.0,
        0.5,
        2.0,
        std::numbers::pi,
        1e-300,
        1e300,
        5e-324,  // 最小子正规数
        2.2250738585072014e-308,
        1.7976931348623157e308,  // DBL_MAX
        0.1,
        -0.1,
        123456.789,
        1.5,
        0.75,
        -0.25};
    for (double d : vals) {
        BigFloat f(d);
        TEST_CHECK(std::bit_cast<uint64_t>(f.to_double()) == std::bit_cast<uint64_t>(d),
            "double roundtrip");
    }
    // 随机 bit 模式往返
    for (int i = 0; i < 2000; ++i) {
        uint64_t bits = rng();
        auto     d    = std::bit_cast<double>(bits);
        if (std::isnan(d) || std::isinf(d))
            continue;  // BigFloat 构造对 inf/nan 抛异常
        BigFloat f(d);
        TEST_CHECK(std::bit_cast<uint64_t>(f.to_double()) == bits, "double roundtrip random");
    }
    // inf/nan 构造抛异常
    bool threw = false;
    try {
        BigFloat f(std::numeric_limits<double>::infinity());
        static_cast<void>(f);
    } catch (const std::domain_error&) {
        threw = true;
    }
    TEST_CHECK(threw, "inf ctor throws");
    threw = false;
    try {
        BigFloat f(std::numeric_limits<double>::quiet_NaN());
        static_cast<void>(f);
    } catch (const std::domain_error&) {
        threw = true;
    }
    TEST_CHECK(threw, "nan ctor throws");
}

// ---- 构造（整型/字符串/limb/BigInt + offset）----

static void test_constructors() {
    std::cout << "[constructors]\n";
    // 整型与字符串
    TEST_CHECK(BigFloat(5).to_double() == 5.0, "int ctor");
    TEST_CHECK(BigFloat(-3).sign() == -1, "int ctor neg");
    TEST_CHECK(BigFloat("123").to_double() == 123.0, "string ctor");
    TEST_CHECK(BigFloat(BigInt(7)).to_double() == 7.0, "BigInt ctor");
    // limb + offset（offset 单位：位，负 = 缩小）
    {
        BigFloat f1(LimbVec{1}, false, -1);  // 0.5
        TEST_CHECK(f1.to_double() == 0.5 && f1.get_point_pos() == 1, "limb ctor 2^-1");
        BigFloat f2(LimbVec{1}, false, -64);  // 2^-64
        TEST_CHECK(f2.to_double() == 0x1p-64 && f2.get_point_pos() == 1, "limb ctor 2^-64");
        BigFloat f3(LimbVec{1, 1}, true, 0);  // -(2^64+1)
        TEST_CHECK(BigInt(f3) == -(BigInt("18446744073709551617")), "limb ctor neg");
        BigFloat f4(LimbVec{1}, false, 64);  // 2^64
        TEST_CHECK(BigInt(f4) == (BigInt(1) << 64), "limb ctor +64");
        BigFloat f5(LimbVec{0, 0}, false, 0);
        TEST_CHECK(f5.is_zero(), "all-zero limbs -> 0");
        // Digits 移动构造
        Digits d{5};
        BigFloat f6(std::move(d), true, 0);
        TEST_CHECK(BigInt(f6) == BigInt(-5), "digits move ctor");
        // 空 span
        BigFloat f7(std::span<const uint64_t>(), true, 0);
        TEST_CHECK(f7.is_zero(), "empty span -> 0");
    }
    // 带小数位的整型构造等价性：BigFloat(BigInt(x), -64) == x/2^64
    for (int i = 0; i < 100; ++i) {
        const auto   a  = rand_limbs(2);
        const bool   na = rng() & 1;
        const BigInt x(a, na);
        BigFloat     f(x, -64);
        if (x.is_zero()) {
            TEST_CHECK(f.is_zero(), "BigInt ctor zero");
            continue;
        }
        // 构造后 remove_tail_zero：低位 0 修剪，point 相应减小
        LimbVec exp = a;
        int64_t exp_point = 1;
        while (exp.size() > 1 && exp[0] == 0) {
            exp.erase(exp.begin());
            --exp_point;
        }
        TEST_CHECK(f.get_data() == exp && f.get_point_pos() == exp_point, "BigInt ctor offset");
    }
}

// ---- 移位 ----

static void test_shifts() {
    std::cout << "[shifts]\n";
    for (int i = 0; i < 300; ++i) {
        const auto     a   = rand_limbs(2);
        const bool     na  = rng() & 1;
        const int64_t  off = -static_cast<int64_t>(rng() % 200);  // 负 offset（小数）
        BigFloat       f(a, na, off);
        const int64_t  k = static_cast<int64_t>(rng() % 300) - 150;
        // 移位 == 构造时直接给 offset+k（两条不同路径，交叉验证）
        BigFloat g(a, na, off + k);
        BigFloat h = f;
        h <<= k;
        TEST_CHECK((h - g).is_zero(), "shift == direct offset");
        // >>= 是相反方向
        BigFloat h2 = h;
        h2 >>= k;
        TEST_CHECK((h2 - f).is_zero(), "shift roundtrip");
    }
    // 已知值
    BigFloat f(1);
    f <<= 10;
    TEST_CHECK(f.to_double() == 1024.0, "1 << 10");
    f >>= 10;
    TEST_CHECK(f.to_double() == 1.0, "1 >> 10");
    BigFloat g(1);
    g <<= -1;
    TEST_CHECK(g.to_double() == 0.5, "1 << -1 == 0.5");
    g <<= -63;
    TEST_CHECK(g.to_double() == 0x1p-64, "0.5 << -63 == 2^-64");
}

// ---- 加减（不同 point 位置，limb 级独立参考对拍）----

static void test_add_sub() {
    std::cout << "[add/sub]\n";
    // 点位置 0（整数）与 BigInt 对拍
    for (int i = 0; i < 200; ++i) {
        const auto   a = rand_limbs(2), b = rand_limbs(2);
        const bool   na = rng() & 1, nb = rng() & 1;
        const BigFloat fa(a, na), fb(b, nb);
        TEST_CHECK(BigInt(fa + fb) == BigInt(a, na) + BigInt(b, nb), "add vs BigInt");
        TEST_CHECK(BigInt(fa - fb) == BigInt(a, na) - BigInt(b, nb), "sub vs BigInt");
    }
    // 随机小数点位：limb 级参考（ref_float_add_sub）
    for (int i = 0; i < 1000; ++i) {
        const auto a = rand_limbs(2), b = rand_limbs(2);
        const bool na = rng() & 1, nb = rng() & 1;
        const int64_t off_a = -static_cast<int64_t>(rng() % 5) * 64;
        const int64_t off_b = -static_cast<int64_t>(rng() % 5) * 64;
        const BigFloat fa(a, na, off_a), fb(b, nb, off_b);
        check_ref_float(fa + fb, ref_float_add_sub(fa, fb, false), "float add random");
        check_ref_float(fa - fb, ref_float_add_sub(fa, fb, true), "float sub random");
        check_ref_float(fb - fa, ref_float_add_sub(fb, fa, true), "float sub swapped");
        // 与零/自身的边界
        check_ref_float(fa + BigFloat(0), ref_float_add_sub(fa, BigFloat(0), false), "add zero");
        check_ref_float(fa - fa, ref_float_add_sub(fa, fa, true), "sub self");
        check_ref_float(-fa + fa, ref_float_add_sub(-fa, fa, false), "neg add");
    }
    // 已知值（小数）
    {
        BigFloat a(BigInt(1), -64);  // 2^-64
        BigFloat b(BigInt(1), -64);
        TEST_CHECK(BigInt(a + b) == BigInt(0) && (a + b).get_point_pos() == 1
                       && (a + b).get_data()[0] == 2,
            "2^-64 + 2^-64");
        BigFloat c(1);  // 1.0
        TEST_CHECK(BigInt(c + b) == BigInt(1), "1 + 2^-64 truncates");
        const BigFloat s = c + b;  // 1 + 2^-64：data = {2^-64 部分, 1.0 部分}，point = 1
        const LimbVec  one_plus_eps{1, 1};
        TEST_CHECK(s.get_data() == one_plus_eps && s.get_point_pos() == 1, "1 + 2^-64 exact");
    }
}

// ---- 乘法 ----

// 模拟 mul_digits(a,b,precision) 的截断语义（输入截顶 precision 个 limb → 乘积 → 截顶，
// point 补偿总截断量），与 BigFloat::mul 内部表示对拍
static auto ref_mul_precision(const BigFloat& fa, const BigFloat& fb, std::size_t precision)
    -> RefFloat {
    LimbVec   ta(fa.get_data()), tb(fb.get_data());
    int64_t   point = fa.get_point_pos() + fb.get_point_pos();
    std::size_t off_a = ta.size() > precision ? ta.size() - precision : 0;
    std::size_t off_b = tb.size() > precision ? tb.size() - precision : 0;
    while (off_a < ta.size() && ta[off_a] == 0)
        ++off_a;
    while (off_b < tb.size() && tb[off_b] == 0)
        ++off_b;
    point -= static_cast<int64_t>(off_a + off_b);
    ta.erase(ta.begin(), ta.begin() + static_cast<int64_t>(off_a));
    tb.erase(tb.begin(), tb.begin() + static_cast<int64_t>(off_b));
    LimbVec c{0};
    if (!ta.empty() && !tb.empty()) {
        c                 = ref_mul(ta, tb);
        std::size_t off_r = c.size() > precision ? c.size() - precision : 0;
        while (off_r < c.size() && c[off_r] == 0)
            ++off_r;
        point -= static_cast<int64_t>(off_r);
        c.erase(c.begin(), c.begin() + static_cast<int64_t>(off_r));
        if (c.empty())
            c.push_back(0);
    }
    if (c.size() == 1 && c[0] == 0)
        return {.limbs = {0}, .point = 0, .sign = 0};
    // 尾零修剪（BigFloat::mul 构造后 remove_tail_zero）
    while (c.size() > 1 && c[0] == 0) {
        c.erase(c.begin());
        --point;
    }
    return {.limbs = std::move(c), .point = point, .sign = fa.sign() * fb.sign()};
}

static void test_mul() {
    std::cout << "[mul]\n";
    // operator* 精确乘法（limb 级参考）
    for (int i = 0; i < 500; ++i) {
        const auto a = rand_limbs(2), b = rand_limbs(2);
        const bool na = rng() & 1, nb = rng() & 1;
        const int64_t off_a = -static_cast<int64_t>(rng() % 4) * 64;
        const int64_t off_b = -static_cast<int64_t>(rng() % 4) * 64;
        const BigFloat fa(a, na, off_a), fb(b, nb, off_b);
        check_ref_float(fa * fb, ref_float_mul(fa, fb), "float mul random");
        // 交换律（库内）
        TEST_CHECK((fa * fb - fb * fa).is_zero(), "mul commutative");
    }
    // mul(precision) 截断（limb 级参考，含 point 验证）
    for (int i = 0; i < 500; ++i) {
        const auto a = rand_limbs(2), b = rand_limbs(2);
        const bool na = rng() & 1, nb = rng() & 1;
        const int64_t off_a = -static_cast<int64_t>(rng() % 3) * 64;
        const int64_t off_b = -static_cast<int64_t>(rng() % 3) * 64;
        const std::size_t precision = 1 + rng() % 4;
        const BigFloat fa(a, na, off_a), fb(b, nb, off_b);
        check_ref_float(
            BigFloat::mul(fa, fb, precision), ref_mul_precision(fa, fb, precision), "mul precision");
    }
    // 与 64 位整数混合（含 2 的幂移位路径）
    for (int i = 0; i < 300; ++i) {
        const auto   a  = rand_limbs(2);
        const bool   na = rng() & 1;
        const int64_t off = -static_cast<int64_t>(rng() % 3) * 64;
        const BigFloat fa(a, na, off);
        const auto     c = static_cast<int64_t>(rng());
        if (c == 0 || c == 1)
            continue;
        BigFloat exp = fa;
        exp *= c;
        // uint64 算术求 |c|（避免 -INT64_MIN 的 UB）；参考基于构造后的内部表示
        const auto uc = c < 0 ? -static_cast<uint64_t>(c) : static_cast<uint64_t>(c);
        const auto& da = fa.get_data();
        TEST_CHECK(exp.get_data() == ref_mul(da, {uc}) && exp.get_point_pos() == fa.get_point_pos()
                       && exp.sign() == fa.sign() * (c < 0 ? -1 : 1),
            "mul int64 random");
        TEST_CHECK((fa * c - exp).is_zero(), "mul int64 equals *=");
    }
    // 已知值：2 的幂乘法走移位分支
    {
        BigFloat f(BigInt(3));
        f *= 8;
        TEST_CHECK(BigInt(f) == BigInt(24), "3 * 8");
        BigFloat g(BigInt(5), -64);
        g *= 16;  // 5*2^-64 * 2^4 = 5*2^-60（移位路径：数据 ×2^4，point 不变）
        TEST_CHECK((g - BigFloat(BigInt(5), -60)).is_zero(), "5*2^-64 * 16");
        TEST_CHECK((g - g).is_zero(), "x - x");
    }
}

// ---- round ----

static void test_round() {
    std::cout << "[round]\n";
    // 已知值：14314.25（Point=0 舍到整数）
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
    // 二进制精确的小数（手工验算）
    {
        const auto rnd = [](double d, RoundMode mode, int64_t prec, RoundRelativeTo rel) {
            BigFloat x(d);
            x.round(mode, prec, rel);
            return BigInt(x);
        };
        // 0.5 = 2^-1，舍到整数（Point=0）
        TEST_CHECK(rnd(0.5, RoundMode::Ceil, 0, RoundRelativeTo::Point) == BigInt(1),
            "ceil 0.5 -> 1");
        TEST_CHECK(rnd(0.5, RoundMode::Floor, 0, RoundRelativeTo::Point) == BigInt(0),
            "floor 0.5 -> 0");
        TEST_CHECK(rnd(0.5, RoundMode::RoundHalfUp, 0, RoundRelativeTo::Point) == BigInt(1),
            "halfup 0.5 -> 1");
        TEST_CHECK(rnd(0.5, RoundMode::Truncate, 0, RoundRelativeTo::Point) == BigInt(0),
            "trunc 0.5 -> 0");
        TEST_CHECK(rnd(-0.5, RoundMode::Ceil, 0, RoundRelativeTo::Point) == BigInt(0),
            "ceil -0.5 -> 0");
        TEST_CHECK(rnd(-0.5, RoundMode::Floor, 0, RoundRelativeTo::Point) == BigInt(-1),
            "floor -0.5 -> -1");
        TEST_CHECK(rnd(-0.5, RoundMode::RoundHalfUp, 0, RoundRelativeTo::Point) == BigInt(-1),
            "halfup -0.5 -> -1");
        TEST_CHECK(rnd(0.25, RoundMode::RoundHalfUp, 0, RoundRelativeTo::Point) == BigInt(0),
            "halfup 0.25 -> 0");
        TEST_CHECK(rnd(-0.25, RoundMode::Ceil, 0, RoundRelativeTo::Point) == BigInt(0),
            "ceil -0.25 -> 0");
        TEST_CHECK(rnd(-0.25, RoundMode::Floor, 0, RoundRelativeTo::Point) == BigInt(-1),
            "floor -0.25 -> -1");
        // 1.5 / -1.5 舍入
        TEST_CHECK(rnd(1.5, RoundMode::RoundHalfUp, 0, RoundRelativeTo::Point) == BigInt(2),
            "halfup 1.5 -> 2");
        TEST_CHECK(rnd(-1.5, RoundMode::RoundHalfUp, 0, RoundRelativeTo::Point) == BigInt(-2),
            "halfup -1.5 -> -2");
        TEST_CHECK(rnd(-1.5, RoundMode::Floor, 0, RoundRelativeTo::Point) == BigInt(-2),
            "floor -1.5 -> -2");
        TEST_CHECK(rnd(-1.5, RoundMode::Ceil, 0, RoundRelativeTo::Point) == BigInt(-1),
            "ceil -1.5 -> -1");
    }
    // 随机对拍（limb 级参考，Significant/Point，含负精度与非法枚举）
    const std::array modes{RoundMode::Truncate, RoundMode::Floor, RoundMode::Ceil,
        RoundMode::RoundHalfUp};
    for (int i = 0; i < 1500; ++i) {
        const auto    a  = rand_limbs(3);
        const bool    na = rng() & 1;
        const int64_t off = -static_cast<int64_t>(rng() % 6) * 64;
        BigFloat      f(a, na, off);
        RoundMode     mode = modes[rng() % 4];
        if (rng() % 8 == 0)
            mode = static_cast<RoundMode>(99);  // NOLINT(clang-analyzer-optin.core.EnumCastOutOfRange) 非法 → Truncate
        RoundRelativeTo rel = (rng() & 1) ? RoundRelativeTo::Significant : RoundRelativeTo::Point;
        if (rng() % 8 == 0)
            rel = static_cast<RoundRelativeTo>(99);  // NOLINT(clang-analyzer-optin.core.EnumCastOutOfRange) 非法 → Significant
        const auto precision = static_cast<int64_t>(rng() % 10) - 2;  // [-2, 7]
        BigFloat      g = f;
        g.round(mode, precision, rel);
        check_ref_float(g,
            ref_float_round(f.get_data(), f.get_point_pos(), na, mode, precision, rel),
            "round random");
    }
    // 精度超长：无操作；精度 <= 0（全部舍掉）：归零
    {
        BigFloat f(123.5);
        const auto  before = f;
        f.round(RoundMode::Floor, 1000, RoundRelativeTo::Significant);
        TEST_CHECK(f.get_data() == before.get_data() && f.get_point_pos() == before.get_point_pos(),
            "huge precision no-op");
        BigFloat g(123.5);
        g.round(RoundMode::Floor, -5, RoundRelativeTo::Significant);
        TEST_CHECK(g.is_zero(), "negative precision -> 0");
    }
}

// ---- reciprocal / inv ----

static void test_reciprocal() {
    std::cout << "[reciprocal]\n";
    for (int i = 0; i < 100; ++i) {
        const auto    a = rand_limbs(2);
        if (ref_cmp(a, {0}) == 0)
            continue;
        const bool    na = rng() & 1;
        BigFloat      f(a, na, -static_cast<int64_t>(rng() % 3) * 64);
        const std::size_t prec = 2 + rng() % 6;
        BigFloat      inv  = f.reciprocal(prec);
        BigFloat      prod = BigFloat::mul(f, inv, prec + 4);
        BigFloat      err  = prod - BigFloat(1);
        // 误差应远小于 1 个 limb 的相对量级（乘积精度 prec+4 个 limb）
        TEST_CHECK(BigInt(err).len() <= 1, "reciprocal x*1/x ~= 1");
        // inv 别名与 reciprocal 等价（同路径计算，差必为 0）
        TEST_CHECK((f.inv(prec) - inv).is_zero(), "inv alias == reciprocal");
        // precision=0：使用输入精度
        TEST_CHECK((f.reciprocal(0) - f.reciprocal(f.len())).is_zero(), "precision 0 == len");
        // 负数倒数 = 正数倒数的相反数
        TEST_CHECK(((-f).reciprocal(prec) + inv).is_zero(), "reciprocal neg");
    }
    // 已知值
    {
        BigFloat two(2);
        BigFloat half = two.reciprocal(2);
        TEST_CHECK((BigFloat::mul(two, half, 3) - BigFloat(1)).is_zero(), "1/2");
        BigFloat f(BigInt(1), -64);  // 2^-64
        BigFloat r = f.reciprocal(2);  // 2^64
        TEST_CHECK(BigInt(r) == (BigInt(1) << 64), "reciprocal 2^-64 == 2^64");
        // 0 的倒数抛异常
        bool threw = false;
        try {
            BigFloat z(0);
            static_cast<void>(z.reciprocal(4));
        } catch (const std::domain_error&) {
            threw = true;
        }
        TEST_CHECK(threw, "reciprocal 0 throws");
    }
}

// ---- BigInt(BigFloat) 四种舍入 ----

static void test_to_bigint() {
    std::cout << "[to bigint]\n";
    const std::array modes{
        RoundMode::Truncate, RoundMode::Floor, RoundMode::Ceil, RoundMode::RoundHalfUp};
    for (int i = 0; i < 1000; ++i) {
        // f = (hi 段 << 64*lo_count) + lo 段（lo 段是小数部分）
        const auto   hi  = rand_limbs(2);
        const auto   lo  = rand_limbs(1 + rng() % 3);
        const bool   na  = rng() & 1;
        LimbVec      all = lo;
        all.insert(all.end(), hi.begin(), hi.end());
        const auto lo_count = static_cast<int64_t>(lo.size());
        BigFloat   f(all, na, -64 * lo_count);
        const RoundMode mode = modes[i % 4];
        const auto [abs, sign] = ref_float_to_bigint_abs(f, mode);
        check_matches(BigInt(f, mode), abs, sign, "BigInt(BigFloat) random");
        // 非法模式 = Truncate
        if (i % 16 == 0) {
            const auto [abs2, sign2] = ref_float_to_bigint_abs(f, RoundMode::Truncate);
            // NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange) —— 故意测非法枚举
            check_matches(BigInt(f, static_cast<RoundMode>(99)), abs2, sign2,
                "BigInt(BigFloat) invalid mode");
        }
    }
    // 已知值
    {
        BigFloat f(1.5);
        TEST_CHECK(BigInt(f, RoundMode::Truncate) == BigInt(1), "1.5 trunc -> 1");
        TEST_CHECK(BigInt(f, RoundMode::Floor) == BigInt(1), "1.5 floor -> 1");
        TEST_CHECK(BigInt(f, RoundMode::Ceil) == BigInt(2), "1.5 ceil -> 2");
        TEST_CHECK(BigInt(f, RoundMode::RoundHalfUp) == BigInt(2), "1.5 halfup -> 2");
        BigFloat g(-1.5);
        TEST_CHECK(BigInt(g, RoundMode::Truncate) == BigInt(-1), "-1.5 trunc -> -1");
        TEST_CHECK(BigInt(g, RoundMode::Floor) == BigInt(-2), "-1.5 floor -> -2");
        TEST_CHECK(BigInt(g, RoundMode::Ceil) == BigInt(-1), "-1.5 ceil -> -1");
        TEST_CHECK(BigInt(g, RoundMode::RoundHalfUp) == BigInt(-2), "-1.5 halfup -> -2");
        // |值| < 1 → 0
        BigFloat h(0.5);
        TEST_CHECK(BigInt(h).is_zero(), "0.5 -> 0");
        // 移动构造等价
        BigFloat m(2.5);
        TEST_CHECK(BigInt(std::move(m)) == BigInt(2), "BigInt(BigFloat&&)");
        TEST_CHECK(m.is_zero(), "moved BigFloat reset");
    }
}

// ---- 字符串输出 ----

static void test_to_string() {
    std::cout << "[to string]\n";
    TEST_CHECK(BigFloat(1).to_string() == "1.", "1 -> 1.");
    TEST_CHECK(BigFloat(1).to_string(2) == "1.00", "1 -> 1.00");
    TEST_CHECK(BigFloat(123).to_string() == "123.", "123 -> 123.");
    TEST_CHECK(BigFloat(0.5).to_string(1) == "0.5", "0.5");
    TEST_CHECK(BigFloat(-0.5).to_string(1) == "-0.5", "-0.5");
    TEST_CHECK(BigFloat(0.25).to_string(2) == "0.25", "0.25");
    TEST_CHECK(BigFloat(-1.5).to_string(1) == "-1.5", "-1.5");
    TEST_CHECK(BigFloat(1.5).to_string(1) == "1.5", "1.5");
    TEST_CHECK(BigFloat(0).to_string() == "0.", "0");
    TEST_CHECK(BigFloat(0).to_string(3) == "0.000", "0 -> 0.000");
    // 默认小数位数 = 二进制小数位数 * log10(2)
    TEST_CHECK(BigFloat(0.5).to_string() == "0.5000000000000000000", "0.5 default digits");
    // print 与 to_string 一致（direct / 非 direct）
    const BigFloat f(-1.5);
    std::ostringstream os1, os2;
    f.print(os1, 1, true);
    f.print(os2, 1, false);
    TEST_CHECK(os1.str() == "-1.5" && os2.str() == "-1.5", "print direct/buffered");
    // 流输出与 print 代理
    std::ostringstream os3;
    os3 << f;
    TEST_CHECK(os3.str() == f.to_string(), "operator<<");
    std::ostringstream os4;
    os4 << bigint::print(f, 1, true);
    TEST_CHECK(os4.str() == "-1.5", "print helper");
    // 保留精度语义：to_string 不用于往返（文档警告），但舍入到整数后应与 BigInt 一致
    BigFloat big(BigInt("12345678901234567890"));
    TEST_CHECK(big.to_string() == "12345678901234567890.", "big integer part");
    // 流格式不干净时 direct=false 的防御性：缓冲 + rdbuf 一次性倒出，
    // 不受外部流 hex/uppercase/setfill/setw 等标志影响（direct=true 无此保证）
    std::ostringstream os5;
    os5 << std::hex << std::uppercase << std::setfill('0');
    f.print(os5, 1, false);
    TEST_CHECK(os5.str() == f.to_string(1), "print buffered, dirty flags");
    std::ostringstream os6;
    os6 << std::setw(40) << std::setfill('*') << bigint::print(f, 1, false);
    TEST_CHECK(os6.str() == f.to_string(1), "print helper buffered, setw");
}

}  // namespace

auto main() -> int {
    test_double_roundtrip();
    test_constructors();
    test_shifts();
    test_add_sub();
    test_mul();
    test_round();
    test_reciprocal();
    test_to_bigint();
    test_to_string();
    return bigint_test::summary("test_bigfloat");
}
