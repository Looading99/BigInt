#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>


#include "bigint/bigint.h"


using namespace bigint;

auto bst(size_type r) -> std::pair<BigInt, BigInt> {
    BigInt p0("13591409");
    BigInt q0("10939058860032000");

    struct Entry {
        BigInt val_P;
        BigInt val_Q;
        BigInt val_R;
        bool   valid{};
    };

    std::vector<Entry> segs;
    segs.reserve(std::bit_width(r));
    for (size_type i = 1; i <= r; ++i) {
        BigInt val_Q(q0);
        val_Q *= i;
        val_Q *= i;
        val_Q *= i;

        BigInt val_R(2 * i - 1);
        val_R *= 6 * i - 1;
        val_R *= 6 * i - 5;

        p0 += 545140134;
        BigInt val_P(p0 * val_R);
        if (i % 2) {
            val_P.flip_sign();
        }

        bool moved = false;
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
    size_type dec_digits = 0;
    std::cout << "input target pi precision(decimal digits): ";
    std::cin >> dec_digits;
    if (dec_digits < 2) {
        dec_digits = 2;
    }

    auto inner_precision = static_cast<size_type>(
        std::ceil(static_cast<double>(dec_digits + 1) / DIGIT_BITS * std::log2(10)));
    inner_precision = inner_precision * 3 / 2;  // 保证精度足够


    auto t1 = std::chrono::high_resolution_clock::now();

    auto [P, Q] = bst(dec_digits / 14 + 5);

    BigFloat pi(Q);
    pi *= 4270934400;

    // 计算 1/sqrt(10005)
    BigFloat s(1.0 / std::sqrt(10005));
    {
        const BigFloat a(1.5);
        size_type      cur_prec = 52 / DIGIT_BITS;
        while (cur_prec < inner_precision) {
            cur_prec *= 2;
            s = BigFloat::mul(s, a - ((10005 * BigFloat::mul(s, s, cur_prec)) >> 1), cur_prec);
        }
    }
    pi = BigFloat::mul(pi, s, inner_precision);
    pi = BigFloat::mul(pi, BigFloat(13591409 * Q + P).reciprocal(inner_precision), inner_precision);


    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "calucation finished in "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << "ms.\n";

    std::ofstream out("pi.txt", std::ios::out);
    pi.print(out, dec_digits, true);
    out << '\n';
    out.close();
    std::cout << "output has been written to pi.txt.\n";
}
