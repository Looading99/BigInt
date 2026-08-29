// BigInt 构造与类型转换测试：
//   整型（含 128 位）/ constexpr / limb span / Digits 移动 / 十进制与十六进制字符串 / 流输入。
// 字符串解析与独立参考实现（ref_parse_dec/ref_parse_hex）随机对拍。
#include <cstdint>
#include <sstream>
#include <string>

#include "bigint/bigint.h"
#include "test_common.h"

namespace {

using bigint::BigInt;
using bigint::Digits;
using bigint::int128_t;
using bigint::uint128_t;
using bigint_test::LimbVec;
using bigint_test::check_matches;
using bigint_test::rand_limbs;
using bigint_test::ref_parse_dec;
using bigint_test::ref_parse_hex;
using bigint_test::rng;
using bigint_test::to_limbs;

// ---- 整型构造（64 位及以下）----

static void test_integral_constructors() {
    std::cout << "[integral constructors]\n";
    // 零与符号
    TEST_CHECK(BigInt(0).is_zero() && BigInt(0).sign() == 0, "0");
    TEST_CHECK(BigInt(1) == BigInt(1) && BigInt(1).sign() == 1, "1");
    TEST_CHECK(BigInt(-1).sign() == -1 && BigInt(-1) == BigInt(-1), "-1");
    // 64 位边界值
    TEST_CHECK(BigInt(UINT64_MAX) == BigInt("18446744073709551615"), "UINT64_MAX");
    TEST_CHECK(BigInt(INT64_MAX) == BigInt("9223372036854775807"), "INT64_MAX");
    TEST_CHECK(BigInt(INT64_MIN) == -BigInt("9223372036854775808"), "INT64_MIN");
    // 小类型
    TEST_CHECK(BigInt(uint8_t(255)) == BigInt(255), "uint8_t");
    TEST_CHECK(BigInt(int8_t(-128)) == BigInt(-128), "int8_t");
    TEST_CHECK(BigInt(uint32_t(4000000000u)) == BigInt("4000000000"), "uint32_t");
    TEST_CHECK(BigInt(int32_t(-2000000000)) == BigInt("-2000000000"), "int32_t");
    // 128 位
    const auto u128_max = ~uint128_t(0);
    TEST_CHECK(BigInt(u128_max) == BigInt("340282366920938463463374607431768211455"),
        "uint128_t max == 2^128-1");
    const auto i128_min = int128_t(1) << 127;
    TEST_CHECK(BigInt(i128_min) == -BigInt("170141183460469231731687303715884105728"),
        "int128_t min == -2^127");
    TEST_CHECK(BigInt(int128_t(-1)) == BigInt(-1), "int128_t -1");
    TEST_CHECK(BigInt(int128_t(0)) == BigInt(0), "int128_t 0");
    TEST_CHECK(BigInt(uint128_t(1) << 100) == (BigInt(1) << 100), "2^100");
}

// ---- limb span / Digits 移动构造 ----

static void test_limb_constructors() {
    std::cout << "[limb constructors]\n";
    // span 复制构造与 Digits 移动构造互相对拍
    for (int i = 0; i < 200; ++i) {
        const auto v   = rand_limbs(3);
        const bool neg = rng() & 1;
        BigInt     a(v, neg);
        Digits     d(v);
        BigInt     b(std::move(d), neg);
        TEST_CHECK(a == b, "span ctor == move ctor");
        // 前导 0 应在构造时被移除
        Digits d2(v);
        d2.push_back(0);
        BigInt c(d2, neg);
        TEST_CHECK(c == a, "leading zero trimmed");
    }
    // 边界：空输入、全 0 输入、负零
    TEST_CHECK(BigInt(std::span<const uint64_t>(), false).is_zero(), "empty span -> 0");
    TEST_CHECK(BigInt(std::span<const uint64_t>(), true).is_zero(), "empty span neg -> 0");
    TEST_CHECK(BigInt(LimbVec{0, 0}, true).is_zero() && BigInt(LimbVec{0, 0}, true).sign() == 0,
        "all-zero limbs -> 0");
    Digits z{0, 0};
    BigInt zz(std::move(z), true);
    TEST_CHECK(zz.is_zero() && zz.sign() == 0, "all-zero digits move -> 0");
    TEST_CHECK(BigInt(LimbVec{0}, true).is_zero(), "single zero limb neg -> 0");
    // 单/双 limb 已知值
    TEST_CHECK(
        BigInt(LimbVec{UINT64_MAX}, false) == BigInt("18446744073709551615"), "limb UINT64_MAX");
    TEST_CHECK(BigInt(LimbVec{1, 1}, true) == -(BigInt("18446744073709551617")), "two limbs neg");
    // 空 Digits 移动
    Digits empty;
    TEST_CHECK(BigInt(std::move(empty), true).is_zero(), "empty digits move -> 0");
}

// 生成随机解析输入：数字 + 随机垃圾字符（含 '-'）
static auto rand_string(std::size_t max_len, bool hex) -> std::string {
    const std::size_t n = rng() % (max_len + 1);
    std::string       s;
    s.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        const uint64_t r = rng() % 6;
        if (r == 0)
            s.push_back('-');
        else if (r == 1)
            s.push_back(hex ? "xyzXYZ"[rng() % 6] : "abcxyz"[rng() % 6]);
        else if (hex && r == 2)
            s.push_back("abcdefABCDEF"[rng() % 12]);
        else if (r <= 4)
            s.push_back(static_cast<char>('0' + rng() % 10));
        else
            s.push_back(static_cast<char>(rng() % 128));
    }
    return s;
}

// ---- 字符串构造（十进制）----

static void test_dec_string_constructors() {
    std::cout << "[dec string constructors]\n";
    // 已知值
    TEST_CHECK(to_limbs(BigInt("123456789012345678901234567890"))
                   == ref_parse_dec("123456789012345678901234567890"),
        "long dec string");
    TEST_CHECK(BigInt("-12345678901234567890") == -BigInt("12345678901234567890"), "neg dec");
    // 边界：空串、纯垃圾、只有负号、前导零、负零、+ 号
    TEST_CHECK(BigInt("").is_zero(), "empty string -> 0");
    TEST_CHECK(BigInt("abc").is_zero(), "no digits -> 0");
    TEST_CHECK(BigInt("-").is_zero() && BigInt("-").sign() == 0, "only '-' -> 0");
    TEST_CHECK(BigInt("000123") == BigInt(123), "leading zeros");
    TEST_CHECK(BigInt("-0") == BigInt(0) && BigInt("-0").sign() == 0, "-0 -> 0");
    TEST_CHECK(BigInt("+12") == BigInt(12), "'+' ignored");
    TEST_CHECK(BigInt("00-123") == BigInt(123), "sign after first digit ignored");
    TEST_CHECK(BigInt("12-34") == BigInt(1234), "sign after digits ignored");
    TEST_CHECK(BigInt("-1a2b3") == BigInt(-123), "garbage between digits");
    // 随机对拍（垃圾字符 + 符号；只有第一个合法数字前的 '-' 生效）
    for (int i = 0; i < 500; ++i) {
        const std::string s         = rand_string(60, false);
        const std::size_t first_dig = s.find_first_of("0123456789");
        const bool        neg       = first_dig != std::string::npos && s.find('-') < first_dig;
        BigInt            x(s);
        TEST_CHECK(to_limbs(x) == ref_parse_dec(s)
                       && x.sign() == (ref_parse_dec(s) == LimbVec{0} ? 0 : (neg ? -1 : 1)),
            "random dec");
    }
}

// ---- 字符串构造（十六进制）----

static void test_hex_string_constructors() {
    std::cout << "[hex string constructors]\n";
    // 已知值
    TEST_CHECK(BigInt("ff", true) == BigInt(255), "hex ff");
    TEST_CHECK(BigInt("FF", true) == BigInt(255), "hex FF upper");
    TEST_CHECK(BigInt("0x1F", true) == BigInt(31), "0x prefix ignored");
    TEST_CHECK(BigInt("-0x1f", true) == BigInt(-31), "negative hex");
    TEST_CHECK(BigInt("ffffffffffffffff", true) == BigInt("18446744073709551615"), "16 f");
    TEST_CHECK(BigInt("10000000000000000", true) == (BigInt(1) << 64), "2^64 in hex");
    TEST_CHECK(BigInt("xyz", true).is_zero(), "no hex digits -> 0");
    // 随机对拍
    for (int i = 0; i < 300; ++i) {
        const std::string s         = rand_string(40, true);
        const std::size_t first_dig = s.find_first_of("0123456789abcdefABCDEF");
        const bool        neg       = first_dig != std::string::npos && s.find('-') < first_dig;
        BigInt            x(s, true);
        TEST_CHECK(to_limbs(x) == ref_parse_hex(s)
                       && x.sign() == (ref_parse_hex(s) == LimbVec{0} ? 0 : (neg ? -1 : 1)),
            "random hex");
    }
    // hex 与 dec 一致性
    TEST_CHECK(BigInt("deadbeef", true) == BigInt("3735928559"), "deadbeef == 3735928559");
}

// ---- 流输入 ----

static void test_stream_input() {
    std::cout << "[stream input]\n";
    std::stringstream ss("12345 -678 9a");
    BigInt            a(0), b(0), c(0);
    ss >> a >> b;
    TEST_CHECK(a == BigInt(12345) && b == BigInt(-678), "stream dec input");
    // hex 标志
    std::stringstream hx("ff");
    hx >> std::hex >> c;
    TEST_CHECK(c == BigInt(255), "stream hex flag input");
    // 大数
    std::stringstream big("1234567890123456789012345678901234567890");
    BigInt            d(0);
    big >> d;
    TEST_CHECK(d == BigInt("1234567890123456789012345678901234567890"), "stream big input");
}

}  // namespace

auto main() -> int {
    test_integral_constructors();
    test_limb_constructors();
    test_dec_string_constructors();
    test_hex_string_constructors();
    test_stream_input();
    return bigint_test::summary("test_constructors");
}
