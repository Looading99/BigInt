// BigInt 字符串与流测试：to_string（dec/hex）与 print/operator<<（含流 hex 标志检测）、
// print 流式代理、随机往返、独立参考交叉验证、分治/暴力解析路径一致性、
// 大十进制串（>5000 位，走分治解析与输出）。
#include <array>
#include <cstdint>
#include <iomanip>
#include <sstream>
#include <string>

#include "bigint/bigint.h"
#include "test_common.h"

namespace {

using bigint::BigInt;
using bigint_test::rand_limbs;
using bigint_test::ref_parse_dec;
using bigint_test::ref_parse_hex;
using bigint_test::rng;
using bigint_test::to_limbs;

// ---- 已知值 ----

static void test_known_values() {
    std::cout << "[known values]\n";
    TEST_CHECK(BigInt(0).to_string() == "0", "0");
    TEST_CHECK(BigInt(-1).to_string() == "-1", "-1");
    TEST_CHECK(BigInt(123456789).to_string() == "123456789", "positive");
    BigInt p256(1);
    p256 <<= 256;
    TEST_CHECK(
        p256.to_string()
            == "115792089237316195423570985008687907853269984665640564039457584007913129639936",
        "2^256");
    TEST_CHECK(
        (p256 - 1).to_string()
            == "115792089237316195423570985008687907853269984665640564039457584007913129639935",
        "2^256-1");
    BigInt fact(1);
    for (int i = 2; i <= 30; ++i)
        fact *= BigInt(i);
    TEST_CHECK(fact.to_string() == "265252859812191058636308480000000", "30!");
    BigInt p10(1);
    p10 *= BigInt(10000000000000000000ull);
    p10 *= BigInt(10000000000000000000ull);
    TEST_CHECK(p10.to_string() == "100000000000000000000000000000000000000", "10^38");
    // hex 输出：无 0x 前缀、保留负号、小写
    TEST_CHECK(BigInt(255).to_string(true) == "ff", "hex 255");
    TEST_CHECK(BigInt(-255).to_string(true) == "-ff", "hex -255");
    TEST_CHECK(BigInt(0).to_string(true) == "0", "hex 0");
    TEST_CHECK((BigInt(1) << 64).to_string(true) == "10000000000000000", "hex 2^64");
    // 超大 hex
    const BigInt big = BigInt("123456789abcdef0123456789abcdef0", true);
    TEST_CHECK(big.to_string(true) == "123456789abcdef0123456789abcdef0", "big hex roundtrip");
}

// ---- 随机往返与独立参考交叉 ----

static void test_roundtrip() {
    std::cout << "[roundtrip]\n";
    for (int i = 0; i < 300; ++i) {
        const auto   a  = rand_limbs(3);
        const bool   na = rng() & 1;
        const BigInt A(a, na);
        TEST_CHECK(BigInt(A.to_string()) == A, "dec roundtrip");
        TEST_CHECK(BigInt(A.to_string(true), true) == A, "hex roundtrip");
        // 独立参考交叉：to_string 输出解析回去 == 原 limbs
        TEST_CHECK(ref_parse_dec(A.to_string()) == a, "to_string == ref_parse_dec");
        TEST_CHECK(ref_parse_hex(A.to_string(true)) == a, "hex to_string == ref_parse_hex");
    }
    // 较大规模（几十 limb）往返
    for (int i = 0; i < 30; ++i) {
        const auto   a  = rand_limbs(20);
        const bool   na = rng() & 1;
        const BigInt A(a, na);
        TEST_CHECK(BigInt(A.to_string()) == A, "big dec roundtrip");
        TEST_CHECK(BigInt(A.to_string(true), true) == A, "big hex roundtrip");
    }
}

// ---- print / operator<< / 流式代理 ----

static void test_print_and_stream() {
    std::cout << "[print and stream]\n";
    const std::array vals{BigInt(0),
        BigInt(1),
        BigInt(-1),
        BigInt(123456789),
        BigInt("123456789012345678901234567890"),
        -(BigInt(1) << 200)};
    for (const auto& x : vals) {
        // print 直接输出到流 == to_string
        std::ostringstream os1, os2, os3;
        x.print(os1, false, true);
        x.print(os2, true, true);
        x.print(os3, false, false);  // 非 direct（缓冲）
        TEST_CHECK(os1.str() == x.to_string(), "print direct dec");
        TEST_CHECK(os2.str() == x.to_string(true), "print direct hex");
        TEST_CHECK(os3.str() == x.to_string(), "print buffered");
        // operator<< 检测流 hex 标志
        std::ostringstream os4, os5;
        os4 << x;
        os5 << std::hex << x;
        TEST_CHECK(os4.str() == x.to_string(), "stream dec");
        TEST_CHECK(os5.str() == x.to_string(true), "stream hex flag");
        // print 流式代理：不检测流标志、支持 direct
        std::ostringstream os6, os7;
        os6 << std::hex << bigint::print(x, false, true);
        os7 << bigint::print(x, true, false);
        TEST_CHECK(os6.str() == x.to_string(), "print helper ignores flags");
        TEST_CHECK(os7.str() == x.to_string(true), "print helper hex");
    }
}

// ---- 流格式不干净时 direct=false 的防御性 ----
// print 的 direct=false 走内部干净缓冲，最后一次性 rdbuf 倒出，
// 应不受外部流的 hex/uppercase/setfill/setw 等格式标志影响。
// （direct=true 无此保证——文档要求调用方确保流标志干净。）

static void test_print_dirty_stream() {
    std::cout << "[print dirty stream]\n";
    const std::array vals{BigInt(0),
        BigInt(1),
        BigInt(-1),
        BigInt(123456789),
        BigInt("123456789012345678901234567890"),
        -(BigInt(1) << 200)};
    for (const auto& x : vals) {
        // 持续标志污染：hex + uppercase + setfill
        std::ostringstream os1;
        os1 << std::hex << std::uppercase << std::setfill('0');
        x.print(os1, false, false);
        TEST_CHECK(os1.str() == x.to_string(), "print dec buffered, dirty flags");
        // 宽度污染：setw + setfill
        std::ostringstream os2;
        os2 << std::setw(40) << std::setfill('*');
        x.print(os2, false, false);
        TEST_CHECK(os2.str() == x.to_string(), "print dec buffered, setw");
        // hex 输出 + 脏流（print_hex 内部自带标志管理，写入缓冲后倒出）
        std::ostringstream os3;
        os3 << std::oct << std::setfill('x');
        x.print(os3, true, false);
        TEST_CHECK(os3.str() == x.to_string(true), "print hex buffered, dirty flags");
        // 流式代理（默认 direct=false）同样防御
        std::ostringstream os4;
        os4 << std::hex << std::uppercase << std::setw(40) << bigint::print(x, false, false);
        TEST_CHECK(os4.str() == x.to_string(), "print helper buffered, dirty flags");
    }
}

// ---- 分治解析路径（DEC_STRING_BRUTE_THRESHOLDS 三配置一致性）----

static void test_divide_and_conquer_parsing() {
    std::cout << "[divide and conquer parsing]\n";
    const auto saved = BigInt::DEC_STRING_BRUTE_THRESHOLDS;
    // 生成 6000 位随机数字串（> 默认阈值 5000，走分治路径）
    std::string s;
    s.reserve(6000);
    s.push_back(static_cast<char>('1' + rng() % 9));
    for (int i = 1; i < 6000; ++i)
        s.push_back(static_cast<char>('0' + rng() % 10));
    BigInt by_default(s);
    BigInt::DEC_STRING_BRUTE_THRESHOLDS = {0, 0};  // 全分治
    BigInt by_dac(s);
    BigInt::DEC_STRING_BRUTE_THRESHOLDS = {SIZE_MAX, SIZE_MAX};  // 全暴力
    BigInt by_brute(s);
    BigInt::DEC_STRING_BRUTE_THRESHOLDS = saved;  // 恢复
    TEST_CHECK(by_dac == by_default && by_brute == by_default, "three parse paths agree");
    // 分治路径 roundtrip
    TEST_CHECK(by_default.to_string() == s, "6000-digit roundtrip");
    // 已知模式：'9'*6000 == 10^6000 - 1（与 get_pow_of_ten 交叉验证）
    const std::string nines(6000, '9');
    TEST_CHECK(BigInt(nines) == BigInt::get_pow_of_ten(6000) - BigInt(1), "9*6000 == 10^6000-1");
    // 1 后接 6000 个 0
    const std::string pow10 = "1" + std::string(6000, '0');
    TEST_CHECK(BigInt(pow10) == BigInt::get_pow_of_ten(6000), "10^6000 parse");
    // 大 hex 串（6000 hex 位 = 24000 bit）
    std::string h;
    h.reserve(6000);
    h.push_back(static_cast<char>("123456789abcdef"[rng() % 15]));
    for (int i = 1; i < 6000; ++i)
        h.push_back(static_cast<char>("0123456789abcdef"[rng() % 16]));
    BigInt hx(h, true);
    TEST_CHECK(hx.to_string(true) == h, "6000-hex-digit roundtrip");
}

}  // namespace

auto main() -> int {
    test_known_values();
    test_roundtrip();
    test_print_and_stream();
    test_print_dirty_stream();
    test_divide_and_conquer_parsing();
    return bigint_test::summary("test_string");
}
