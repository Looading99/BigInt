// 解决形如 $ a=\sum_{i=1}^{\infty} \frac{P(i)}{R(i)} \prod_{j=1}^{i}\frac{R(j)}{Q(j)} $
// 的求任意精度的无理数 a 的问题
// 方法：分治合并
// P(l, r) = P(l, m) * Q(m + 1, r) + R(l, m) * P(m + 1, r)
// Q(l, r) = Q(l, m) * Q(m + 1, r)
// R(l, r) = R(l, m) * R(m + 1, r)
// 最终 a = P(1, r) / Q(1, r)

#include <bit>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <utility>
#include <vector>


#include "bigint/bigint.h"


using namespace bigint;

template<typename lambda_P, typename lambda_Q, typename lambda_R>
    requires std::is_invocable_r_v<BigInt, lambda_P, std::size_t>
             && std::is_invocable_r_v<BigInt, lambda_Q, std::size_t>
             && std::is_invocable_r_v<BigInt, lambda_R, std::size_t>
auto bst(std::size_t r, lambda_P P, lambda_Q Q, lambda_R R) -> std::pair<BigFloat, BigFloat> {
    struct Entry {
        BigInt val_P;
        BigInt val_Q;
        BigInt val_R;
        bool   valid{};
    };

    std::vector<Entry> segs;
    segs.reserve(std::bit_width(r));
    for (std::size_t i = 1; i <= r; ++i) {
        BigInt val_P(P(i)), val_Q(Q(i)), val_R(R(i));
        bool   moved = false;
        for (auto& seg : segs) {
            if (!seg.valid) {
                moved = seg.valid = true;
                seg.val_P         = std::move(val_P);
                seg.val_Q         = std::move(val_Q);
                seg.val_R         = std::move(val_R);
                break;
            }
            seg.valid = false;
            val_P     = seg.val_P * val_Q + seg.val_R * val_P;
            val_Q     = val_Q * seg.val_Q;
            val_R     = val_R * seg.val_R;
        }
        if (!moved) {
            segs.emplace_back(std::move(val_P), std::move(val_Q), std::move(val_R), true);
        }
    }

    BigInt val_P(0), val_Q(0), val_R(0);
    bool   moved = false;
    for (auto& seg : segs) {
        if (seg.valid) {
            if (!moved) {
                moved = true;
                val_P = std::move(seg.val_P);
                val_Q = std::move(seg.val_Q);
                val_R = std::move(seg.val_R);
            } else {
                val_P = seg.val_P * val_Q + seg.val_R * val_P;
                val_Q = val_Q * seg.val_Q;
                val_R = val_R * seg.val_R;
            }
        }
    }
    return {BigFloat(std::move(val_P)), BigFloat(std::move(val_Q))};
}

auto main() -> int {
    init_thread_pool(8);
    // 示例：计算 e
    auto P = [](std::size_t x) { return BigInt(1); };
    auto Q = [](std::size_t x) { return BigInt(x); };
    auto R = [](std::size_t x) { return BigInt(1); };

    std::size_t dec_digits = 0;
    std::cout << "input target e precision(decimal digits): ";
    std::cin >> dec_digits;
    if (dec_digits < 2) {
        dec_digits = 2;
    }

    double      cur_dec_precision = 0;
    std::size_t r                 = 0;
    while (cur_dec_precision < static_cast<double>(dec_digits)) {
        ++r;
        cur_dec_precision += std::log10(r + 1);
    }

    auto t1 = std::chrono::high_resolution_clock::now();

    auto [res_P, res_Q] = bst(r + 1, P, Q, R);

    std::size_t inner_precision =
        std::ceil(static_cast<double>(dec_digits + 1) / DIGIT_BITS * std::log2(10));
    inner_precision = inner_precision * 3 / 2;  // 保证精度足够

    auto e = BigFloat(1) + res_P * res_Q.reciprocal(inner_precision);

    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "calucation finished in "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << "ms.\n";

    std::ofstream out("e.txt", std::ios::out);
    out << print(e, dec_digits, true) << '\n';
    out.close();
    std::cout << "output has been written to e.txt.\n";
}
