#include <algorithm>
#include <bit>
#include <cstring>
#include <span>
#include <utility>
#include <vector>


#include "bigint/bigint_base.h"
#include "bigint/mul.h"


namespace bigint::mul::ssa {

//  计算 C = A + B mod (2^m+1)，必须保证 A 和 B < 2^m+1，C.size() >= m/64+1 且 m 是 64 的倍数
static inline void mod_add(
    std::span<const uint64_t> A, std::span<const uint64_t> B, std::span<uint64_t> C, size_type m) {
    size_type w = m / 64;
    if (!(C.size() >= w + 1 && m % 64 == 0))
        unreachable();
    utils::add(A, B, C.size() == w + 1 ? C : C.subspan(0, w + 1));
    if (C[w]) {  // C >= 2^m
        size_type i = 0;
        for (; i < w; ++i) {
            if (C[i] != 0) {
                break;
            }
        }
        if (i == w && C[w] == 1)  // C == 2^m
            return;
        // C > 2^m
        std::memset(&C[0], -1, i * sizeof(uint64_t));
        --C[i];
        --C[w];
    }
}

// 要求与 mod_add 相同
static inline void mod_sub(
    std::span<const uint64_t> A, std::span<const uint64_t> B, std::span<uint64_t> C, size_type m) {
    size_type w = m / 64;
    if (!(C.size() >= w + 1 && m % 64 == 0))
        unreachable();
    bool borrow = utils::sub(A, B, C.size() == w + 1 ? C : C.subspan(0, w + 1));
    if (borrow) {
        utils::add1(C);
        ++C[w];
    }
}

// 计算 A = -A mod (2^m+1)，必须保证 A < 2^m+1 且 A.size() >= m/64+1 且 m 是 64 的倍数
static void mod_neg(std::span<uint64_t> A, size_type m) {
    size_type w = m / 64;
    if (!(A.size() >= w + 1 && m % 64 == 0))
        unreachable();
    // A = 2^m 时结果为 1
    if (A[w] == 1) {
        A[0] = 1;
        std::fill(A.begin() + 1, A.begin() + static_cast<int64_t>(w) + 1, 0);
        return;
    }
    // 0 < A < 2^m 时等价于计算 2^m+1 - A = 2^m-1 - A + 2
    size_type i = 0;
    for (; i < w; ++i) {
        if (A[i] != 0) {
            break;
        }
    }
    if (i == w) {  // A == 0
        return;
    }
    // A > 0
    if (i > 0) {
        std::fill(A.begin(), A.begin() + static_cast<int64_t>(i), uint64_t(-1));
    }
    for (; i < w; ++i) {
        A[i] = uint64_t(-1) - A[i];
    }
    bool carry = add_overflow<uint64_t>(A[0], 2, A[0]);
    if (carry) {
        utils::add1(A.subspan(1));
    }
}

// 计算 A = A mod (2^m+1)，m 必须是 64 的倍数
static void mod(std::span<uint64_t> A, size_type m) {
    if (!(m % 64 == 0))
        unreachable();
    size_type w = m / 64;
    if (A.size() <= w) {
        return;
    }
    // 将 A 每 w 个数分割成 k 块
    size_type k = ceil_div(A.size(), w);
    // A mod = A_0 - A_1 + A_2 - A_3 + ...。
    // 迭代 R = (A_i - R) mod (2^m+1) for i from 1 to k-1，初始值为 A_0
    // R 有 w + 1 位，注意 i = 1 时 R 最高位与 A_1 重叠，读取时跳过即可
    auto R = A.subspan(0, w + 1);
    for (size_type i = 1; i < k; ++i) {
        auto A_i = i < k - 1 ? A.subspan(i * w, w) : A.subspan(i * w);
        mod_sub(A_i, i == 1 ? R.subspan(0, w) : R, R, m);
    }
    // 偶数个块时得到的是 -A 的模，取负即可
    if (k % 2 == 0) {
        mod_neg(R, m);
    }
}

// 计算 B = A * 2^p mod (2^m+1)，必须保证 A < 2^m+1 且 B.size() >= m/64+1 且 m 是 64 的倍数 且
// tmp.size() >= 2*m/64，A 和 B 的起始地址可以相同或不同，A 长度不足自动补 0
static void mod_mul_pow_of_two(std::span<const uint64_t> A, std::span<uint64_t> B, uint64_t p,
    size_type m, std::span<uint64_t> tmp) {
    size_type w = m / 64;
    if (!(B.size() >= w + 1 && m % 64 == 0 && tmp.size() >= 2 * w))
        unreachable();
    if (A.size() > w + 1)
        A = A.subspan(0, w + 1);
    if (B.size() > w + 1)
        B = B.subspan(0, w + 1);
    p &= (2 * m - 1);
    bool neg = false;
    if (p >= m) {
        neg = true;
        p -= m;
    }
    if (p == 0) {
        if (A.data() != B.data()) {
            std::memcpy(B.data(), A.data(), A.size() * sizeof(uint64_t));
        }
        if (A.size() < w + 1) {
            std::memset(&B[A.size()], 0, (B.size() - A.size()) * sizeof(uint64_t));
        }
    } else {
        auto      shl_A      = tmp.subspan(0, 2 * w);
        size_type word_shift = p / 64, bit_shift = p % 64;
        std::memset(&shl_A[0], 0, word_shift * sizeof(uint64_t));
        std::memcpy(&shl_A[word_shift], A.data(), A.size() * sizeof(uint64_t));
        if (word_shift + A.size() < 2 * w)
            std::memset(&shl_A[word_shift + A.size()],
                0,
                (shl_A.size() - word_shift - A.size()) * sizeof(uint64_t));
        if (bit_shift) {
            for (size_type i = std::min(word_shift + A.size(), 2 * w - 1); i > word_shift; --i) {
                shl_A[i] = (shl_A[i] << bit_shift) | (shl_A[i - 1] >> (64 - bit_shift));
            }
            shl_A[word_shift] <<= bit_shift;
        }
        mod_sub(shl_A.subspan(0, w), shl_A.subspan(w), B, m);
    }
    if (neg) {
        mod_neg(B, m);
    }
}

// 要求与 mod_mul_pow_of_two 相同
static inline void mod_div_pow_of_two(std::span<const uint64_t> A, std::span<uint64_t> B,
    uint64_t p, size_type m, std::span<uint64_t> tmp) {
    mod_mul_pow_of_two(A, B, 2 * m - (p & (2 * m - 1)), m, tmp);
}

// 计算 A 模 2^m+1 的 NTT，必须保证 A[i] < 2^m+1 且 A[i].size() == m/64+1
// 且 A.size() == m 是 2 的幂且为 64 的倍数 且 tmp.size() >= 3*m/64+1
static void ntt(
    std::vector<std::vector<uint64_t>>& A, size_type m, bool rev, std::span<uint64_t> tmp) {
    size_type w = m / 64;
    if (!(m == A.size() && m != 0 && std::popcount(m) == 1 && m % 64 == 0
            && tmp.size() >= 3 * w + 1))
        unreachable();
    // 位逆序置换
    for (size_type i = 1, j = m >> 1; i < m - 1; ++i) {
        if (i < j)
            std::swap(A[i], A[j]);
        size_type k = m >> 1;
        while (j >= k) {
            j -= k;
            k >>= 1;
        }
        j += k;
    }
    auto b = tmp.subspan(2 * w, w + 1);
    for (size_type len = 2; len <= m; len *= 2) {
        size_type half     = len / 2;
        size_type step_exp = 2 * m / len;
        if (rev)
            step_exp = 2 * m - step_exp;
        for (size_type i = 0; i < m; i += len) {
            size_type cur_exp = 0;
            for (size_type j = 0; j < half; ++j) {
                size_type idx_a = i + j, idx_b = idx_a + half;
                if (!(A[idx_a].size() == w + 1 && A[idx_b].size() == w + 1))
                    unreachable();
                mod_mul_pow_of_two(A[idx_b], b, cur_exp, m, tmp);
                mod_sub(A[idx_a], b, A[idx_b], m);
                mod_add(A[idx_a], b, A[idx_a], m);
                cur_exp = (cur_exp + step_exp) & (2 * m - 1);
            }
        }
    }
    if (rev) {
        for (auto& a : A) {
            mod_div_pow_of_two(a, a, std::countr_zero(m), m, tmp);
        }
    }
}

// m 必须是 2 的幂
static constexpr auto calc_slice(size_type m) -> std::pair<size_type, size_type> {
    // N = sqrt(2*m)
    size_type N = 1ull << (std::countr_zero(m) + 1) / 2;
    size_type K = m / N;
    // 防溢出条件：2^N>=2N*2^(2K) => N-2K>=log2(2N) => N-log2(N)-1>=2K
    while (N - std::countr_zero(static_cast<uint64_t>(N)) - 1 < 2 * K) {
        N *= 2;
        K /= 2;
    }
    return {N, K};
}

// 计算 A*B mod (2^m+1)，必须保证 A 和 B < 2^m+1 且 m 是 2 的幂 且 tmp.size() >= 3*N/64+1
static auto mod_mul(std::span<const uint64_t> A, std::span<const uint64_t> B, size_type m,
    std::span<uint64_t> tmp) -> std::vector<uint64_t> {
    size_type w = m / 64;
    if (A.size() >= w + 1 && A[w] == 1) {
        std::vector<uint64_t> res(B.begin(), B.end());
        res.resize(w + 1);
        mod_neg(res, m);
        return res;
    } else if (B.size() >= w + 1 && B[w] == 1) {
        std::vector<uint64_t> res(A.begin(), A.end());
        res.resize(w + 1);
        mod_neg(res, m);
        return res;
    }

    if ((A.size() + B.size()) * 64 < MIN_TOTAL_BITS) {
        auto res = _mul(A, B);
        mod(res, m);
        res.resize(w + 1);
        return res;
    }

    auto [N, K] = calc_slice(m);

    size_type                          n = N / 64, k = K / 64;
    std::vector<std::vector<uint64_t>> A_slice(N), B_slice(N);

    for (size_type i = 0; i < N; ++i) {
        A_slice[i].resize(n + 1);
        if (i * k >= A.size())
            continue;
        auto A_i = A.subspan(i * k, std::min(k, A.size() - i * k));
        mod_mul_pow_of_two(A_i, A_slice[i], i, N, tmp);
    }
    ntt(A_slice, N, false, tmp);

    for (size_type i = 0; i < N; ++i) {
        B_slice[i].resize(n + 1);
        if (i * k >= B.size())
            continue;
        auto B_i = B.subspan(i * k, std::min(k, B.size() - i * k));
        mod_mul_pow_of_two(B_i, B_slice[i], i, N, tmp);
    }
    ntt(B_slice, N, false, tmp);

    for (size_type i = 0; i < N; ++i) {
        auto C_i = mod_mul(A_slice[i], B_slice[i], N, tmp);
        C_i.resize(n + 1);
        A_slice[i] = std::move(C_i);
    }
    B_slice.clear();
    ntt(A_slice, N, true, tmp);

    for (size_type i = 1; i < N; ++i) {
        mod_div_pow_of_two(A_slice[i], A_slice[i], i, N, tmp);
    }

    std::vector<uint64_t> res_pos((N - 1) * k + n + 1), res_neg = res_pos;
    // 通过比较 A[i] 与 (2^N+1)/2 的大小判断符号，
    // A[i] < (2^N+1)/2 （即 A[i] <= 2^(N-1)）时是非负数，
    // 否则是负数，真实值为 -(2^N+1-A[i])
    auto check_is_neg = [](const std::vector<uint64_t>& a, size_type _N) -> bool {
        // 2^(N-1) : N/64-1 位 0 + 最高位 1 << 63
        size_type i = a.size() - 1;
        for (; i >= _N / 64; --i) {
            if (a[i]) {
                return true;
            }
        }
        if (a[i] != 1ull << 63) {
            return a[i] > 1ull << 63;
        }
        while (i > 0) {
            --i;
            if (a[i]) {
                return true;
            }
        }
        return false;
    };
    for (size_type i = 0; i < N; ++i) {
        bool is_neg = check_is_neg(A_slice[i], N);
        if (is_neg) {
            mod_neg(A_slice[i], N);
        }
        std::span<uint64_t> target(
            (is_neg ? res_neg : res_pos).begin() + static_cast<int64_t>(i * k), n + 1);
        utils::add(A_slice[i], target, target);
    }
    mod(res_pos, m);
    mod(res_neg, m);
    res_pos.resize(w + 1);
    res_neg.resize(w + 1);
    mod_sub(res_pos, res_neg, res_pos, m);
    return res_pos;
}

auto mul(std::span<const uint64_t> A, std::span<const uint64_t> B) -> std::vector<uint64_t> {
    size_type total_bits = (A.size() + B.size()) * 64;
    if (!(total_bits >= MIN_TOTAL_BITS))
        unreachable();
    size_type             m = std::bit_ceil(total_bits);
    std::vector<uint64_t> tmp(calc_slice(m).first * 3 + 1);
    auto                  res = mod_mul(A, B, m, tmp);
    utils::remove_leading_zero(res);
    return res;
}

}  // namespace bigint::mul::ssa
