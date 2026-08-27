// 测量每个 digit_bits 能保证精度的最大输入总比特数（最坏情况），并做随机回归。
//
// 用法：
//   test_fft_digit_bits            # 先做最坏情况边界测量，再做随机回归（需表已回填）
//   test_fft_digit_bits measure    # 仅最坏情况边界测量，输出可直接回填 mul.h 的表
//   test_fft_digit_bits regress    # 仅随机回归（自动选择 digit_bits，需表已回填）
//
// 说明：fft::mul 的 digit_bits 参数可覆盖自动选择，故单次编译即可测出全部结果，
// 无需宏定义多次编译。最坏情况输入为“每 B 位数字都取 2^B-1”，卷积系数取到
// 理论最大值 m*(2^B-1)^2，据此得到保守可保证精度的边界；另附随机输入回归。
// 内部表示为基数 2^64 的 limb 数组，最坏情况即前 nbits 位全 1。
#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <iostream>
#include <random>
#include <span>
#include <string>
#include <vector>


#include "bigint/bigint_base.h"
#include "bigint/mul.h"


using bigint::DIGIT_BITS;
using bigint::Digits;
using bigint::mul::fft::MAX_DIGIT_BITS;
using bigint::mul::fft::MAX_FFT_LEN;
using bigint::mul::fft::MIN_DIGIT_BITS;
using Vec64 = std::vector<uint64_t>;

// 去掉尾部 0，统一长度后再比较
static void normalize(Vec64& v) {
    while (v.size() > 1 && v.back() == 0) {
        v.pop_back();
    }
}

// 2^(B*m)-1 的 64 位 limb 表示（小端）：前 B*m 位全部取 1（最坏情况）
static auto ones_limbs(int B, std::size_t m) -> Vec64 {
    const auto nbits = static_cast<std::size_t>(B) * m;
    const std::size_t nd = (nbits + DIGIT_BITS - 1) / DIGIT_BITS;
    Vec64             d(nd, UINT64_MAX);
    const std::size_t rem = nbits % DIGIT_BITS;
    if (rem != 0) {
        d.back() = (uint64_t(1) << rem) - 1;
    }
    return d;
}

// 最坏情况：两个 operand 各 m 个 2^B-1 数字相乘，强制 FFT(digit_bits=B) 与 NTT 参考比对
static auto worst_case_ok(int B, std::size_t m) -> bool {
    const Vec64 a64   = ones_limbs(B, m);
    Vec64       r_fft = bigint::mul::fft::mul(a64, a64, B);
    Vec64       r_ref = bigint::mul::ntt::mul(a64, a64);  // NTT 参考（64 位块接口）
    normalize(r_fft);
    normalize(r_ref);
    return r_fft == r_ref;
}

// 对给定 B 求最大通过 m（每 operand 的 B 位数字个数）：先探测上限，否则指数探测+二分
static auto max_passing_m(int B) -> std::size_t {
    const std::size_t m_cap = MAX_FFT_LEN / 2;  // 保证变换长度 N <= MAX_FFT_LEN
    if (worst_case_ok(B, m_cap)) {
        return m_cap;
    }
    std::size_t lo = 1, hi = 1;
    while (hi <= m_cap && worst_case_ok(B, hi)) {
        lo = hi;
        hi = std::min(m_cap, hi * 2);
    }
    std::size_t ans = lo, l = lo + 1, r = hi;
    while (l <= r) {
        const std::size_t mid = l + (r - l) / 2;
        if (worst_case_ok(B, mid)) {
            ans = mid;
            l   = mid + 1;
        } else {
            r = mid - 1;
        }
    }
    return ans;
}

static void run_measure() {
    std::cout << "MAX_FFT_LEN=" << MAX_FFT_LEN << " (layer=" << std::countr_zero(MAX_FFT_LEN)
              << ")\n";
    std::array<std::size_t, MAX_DIGIT_BITS + 1> max_total_bits{};
    std::cout << "--- per digit_bits max total bits (worst case) ---\n";
    for (int B = MIN_DIGIT_BITS; B <= MAX_DIGIT_BITS; ++B) {
        const std::size_t m_max = max_passing_m(B);
        const std::size_t t_max = 2 * m_max * static_cast<std::size_t>(B);
        max_total_bits[B]       = t_max;
        std::cout << "B=" << B << "  m_max=" << m_max << "  max_total_bits=" << t_max << "\n";
    }
    std::cout << "--- copy-paste table for mul.h ---\n";
    std::cout << "constexpr std::array<std::size_t, MAX_DIGIT_BITS + 1> "
                 "MAX_TOTAL_BITS_FOR_DIGIT_BITS = {\n";
    for (int i = 0; i <= MAX_DIGIT_BITS; ++i) {
        std::cout << "    " << max_total_bits[i];
        if (i != MAX_DIGIT_BITS) {
            std::cout << ",";
        }
        std::cout << "  // digit_bits=" << i << "\n";
    }
    std::cout << "};\n";
}

static void run_regress() {
    if (bigint::mul::fft::MAX_TOTAL_BITS == 0) {
        std::cout << "regress skipped: MAX_TOTAL_BITS_FOR_DIGIT_BITS 尚未回填\n";
        return;
    }
    std::mt19937_64 rng(0x9e3779b97f4a7c15ull);
    std::size_t     cases = 0, fails = 0;

    auto rand_digits = [&](std::size_t n) {
        Digits d(n);
        for (auto& x : d) {
            x = rng();
        }
        return d;
    };
    auto check = [&](const Digits& a, const Digits& b) {
        Vec64 r_fft = bigint::mul::fft::mul(a, b);  // 自动选择 digit_bits
        Vec64 r_ntt = bigint::mul::ntt::mul(a, b);  // NTT 参考
        normalize(r_fft);
        normalize(r_ntt);
        ++cases;
        if (r_fft != r_ntt) {
            ++fails;
            std::cout << "MISMATCH n_a=" << a.size() << " n_b=" << b.size() << "\n";
        }
    };

    // fft::mul 自动选择，跨各规模（覆盖多个 digit_bits 区间；上限须在 FFT 容量 MAX_TOTAL_BITS 内）
    for (std::size_t n : {200ull, 500ull, 1000ull, 3000ull, 8000ull, 20000ull, 25000ull}) {
        for (int rep = 0; rep < 4; ++rep) {
            check(rand_digits(n), rand_digits(n));
        }
    }
    // mul_digits 全链路（brute/FFT/NTT 分发）vs NTT 参考
    for (std::size_t n : {100ull, 1000ull, 10000ull, 40000ull}) {
        for (int rep = 0; rep < 3; ++rep) {
            Digits a = rand_digits(n), b = rand_digits(n);
            Digits r_mul = bigint::mul::mul_digits(a, b);
            Vec64  r_ntt = bigint::mul::ntt::mul(a, b);
            normalize(r_mul);
            normalize(r_ntt);
            ++cases;
            if (r_mul != r_ntt) {
                ++fails;
                std::cout << "MUL_DIGITS MISMATCH n=" << n << "\n";
            }
        }
    }
    // 强制各 B 的随机输入正确性（小规模，各 B 均在其安全规模内）
    for (int B = MIN_DIGIT_BITS; B <= MAX_DIGIT_BITS; ++B) {
        const std::size_t n = 64;
        for (int rep = 0; rep < 4; ++rep) {
            const Digits a = rand_digits(n), b = rand_digits(n);
            Vec64        r_fft = bigint::mul::fft::mul(a, b, B);
            Vec64        r_ntt = bigint::mul::ntt::mul(a, b);  // NTT 参考
            normalize(r_fft);
            normalize(r_ntt);
            ++cases;
            if (r_fft != r_ntt) {
                ++fails;
                std::cout << "FORCED B=" << B << " MISMATCH\n";
            }
        }
    }
    std::cout << "regress: cases=" << cases << " fails=" << fails << "\n";
}

auto main(int argc, char** argv) -> int {
    const std::span<char*> args(argv, static_cast<std::size_t>(argc));
    const std::string      mode       = args.size() > 1 ? std::string(args[1]) : "";
    const bool             do_measure = mode.empty() || mode == "measure";
    const bool             do_regress = mode.empty() || mode == "regress";
    if (do_measure) {
        run_measure();
    }
    if (do_regress) {
        run_regress();
    }
    return 0;
}
