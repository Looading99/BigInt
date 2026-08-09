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
    requires std::is_invocable_r_v<BigInt, lambda_P, size_type>
             && std::is_invocable_r_v<BigInt, lambda_Q, size_type>
             && std::is_invocable_r_v<BigInt, lambda_R, size_type>
auto bst(size_type r, lambda_P P, lambda_Q Q, lambda_R R) -> std::pair<BigFloat, BigFloat> {
    struct Entry {
        BigInt val_P;
        BigInt val_Q;
        BigInt val_R;
        bool   valid{};
    };

    std::vector<Entry> segs;
    segs.reserve(std::bit_width(r));
    for (size_type i = 1; i <= r; ++i) {
        BigInt val_P = P(i), val_Q = Q(i), val_R = R(i);
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
            segs.emplace_back(val_P, val_Q, val_R, true);
        }
    }
    BigInt &res_P = segs.back().val_P, &res_Q = segs.back().val_Q, &res_R = segs.back().val_R;
    for (size_type i = segs.size() - 1; i > 0;) {
        --i;
        auto& seg = segs[i];
        if (seg.valid) {
            res_P = res_P * seg.val_Q + res_R * seg.val_P;
            res_Q = res_Q * seg.val_Q;
            res_R = res_R * seg.val_R;
        }
    }
    return {std::move(res_P), std::move(res_Q)};
}

auto main() -> int {
    // 示例：计算 e
    auto P = [](size_type x) { return 1; };
    auto Q = [](size_type x) { return x; };
    auto R = [](size_type x) { return 1; };

    size_type dec_digits = 0;
    std::cout << "input target e precision(decimal digits): ";
    std::cin >> dec_digits;
    if (dec_digits < 2) {
        dec_digits = 2;
    }

    double    cur_dec_precision = 0;
    size_type r                 = 0;
    while (cur_dec_precision < static_cast<double>(dec_digits)) {
        ++r;
        cur_dec_precision += std::log10(r + 1);
    }

    auto t1 = std::chrono::high_resolution_clock::now();

    auto [res_P, res_Q] = bst(r + 1, P, Q, R);

    size_type reciprocal_precision =
        std::ceil(static_cast<double>(dec_digits + 1) / DIGIT_BITS * std::log2(10));
    reciprocal_precision *= 2;  // 多迭代一次

    auto e = BigFloat(1) + res_P * res_Q.reciprocal(reciprocal_precision);

    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "calucation finished in "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << "ms.\n";

    std::ofstream out("e.txt", std::ios::out);
    e.print(out, dec_digits, true);
    out << '\n';
    out.close();
    std::cout << "output has been written to e.txt.\n";
}
