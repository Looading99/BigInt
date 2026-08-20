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
        if (i > 0)
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

// 计算 B = A * 2^p mod (2^m+1)，必须保证 A < 2^m+1 且 B.size() >= m/64+1
// 且 m 是 2 的幂且为 64 的倍数 且 temp.size() >= 2*m/64，
// A 和 B 的起始地址可以相同或不同，A 长度不足自动补 0
static void mod_mul_pow_of_two(std::span<const uint64_t> A, std::span<uint64_t> B, uint64_t p,
    size_type m, std::span<uint64_t> temp) {
    size_type w = m / 64;
    if (!(B.size() >= w + 1 && m != 0 && std::popcount(m) == 1 && m % 64 == 0
            && temp.size() >= 2 * w))
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
        auto      shl_A      = temp.subspan(0, 2 * w);
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
    uint64_t p, size_type m, std::span<uint64_t> temp) {
    mod_mul_pow_of_two(A, B, 2 * m - (p & (2 * m - 1)), m, temp);
}

// 计算模 2^m+1 的 NTT，必须保证 ptrs.size() == m 是 2 的幂且为 64 的倍数，
// slices.size() == m*(m/64+1)，temp.size() >= 3*m/64+1
static void ntt(std::span<uint64_t> ptrs, std::span<uint64_t> slices, size_type m, bool rev,
    std::span<uint64_t> temp) {
    size_type w = m / 64;
    if (!(m == ptrs.size() && m != 0 && std::popcount(m) == 1 && m % 64 == 0
            && slices.size() == m * (w + 1) && temp.size() >= 3 * w + 1))
        unreachable();
    // 位逆序置换
    for (size_type i = 1, j = m >> 1; i < m - 1; ++i) {
        if (i < j)
            std::swap(ptrs[i], ptrs[j]);
        size_type k = m >> 1;
        while (j >= k) {
            j -= k;
            k >>= 1;
        }
        j += k;
    }
    auto mul_temp = temp.subspan(0, 2 * w);
    auto b        = temp.subspan(2 * w, w + 1);
    for (size_type len = 2; len <= m; len *= 2) {
        size_type half     = len / 2;
        size_type step_exp = 2 * m >> std::countr_zero(len);
        if (rev)
            step_exp = 2 * m - step_exp;
        for (size_type i = 0; i < m; i += len) {
            size_type cur_exp = 0;
            for (size_type j = 0; j < half; ++j) {
                auto a0 = slices.subspan(ptrs[i + j], w + 1);
                auto b0 = slices.subspan(ptrs[i + j + half], w + 1);
                mod_mul_pow_of_two(b0, b, cur_exp, m, mul_temp);
                mod_sub(a0, b, b0, m);
                mod_add(a0, b, a0, m);
                cur_exp = (cur_exp + step_exp) & (2 * m - 1);
            }
        }
    }
    if (rev) {
        for (size_type i = 0; i < m; ++i) {
            auto a = slices.subspan(i * (w + 1), w + 1);
            mod_div_pow_of_two(a, a, std::countr_zero(m), m, temp);
        }
    }
}

// m 必须是 2 的幂
static constexpr auto calc_slice(size_type m) -> std::pair<size_type, size_type> {
    // N = sqrt(2*m)
    size_type N = 1ull << (std::countr_zero(m) + 1) / 2;
    size_type K = m / N;
    // 防溢出条件：2^N>=2N*2^(2K) => N-2K>=log2(2N) => N-log2(N)-1>=2K
    while (N - std::countr_zero(N) - 1 < 2 * K) {
        N *= 2;
        K /= 2;
    }
    return {N, K};
}

static constexpr auto ptrs_size(size_type N) -> size_type {
    return N;
}

static constexpr auto slices_size(size_type N) -> size_type {
    return N * (N / 64 + 1);
}

static constexpr auto temp_size(size_type N) -> size_type {
    return N / 64 * 3 + 1;
}

static constexpr auto acc_size(size_type N, size_type K) -> size_type {
    return (N - 1) * K / 64 + N / 64 + 1;
}

static constexpr auto ssa_size(size_type m) -> size_type {
    // 递归终点为 (A.size() + B.size()) * 64 < MIN_TOTAL_BITS，
    // 且递归过程总是满足 A.size() 和 B.size() <= m / 64 + 1 (仅第一层小于，其余层等于)
    if (2 * (m + 64) < MIN_TOTAL_BITS) {
        return 0;
    }
    auto [N, K] = calc_slice(m);
    return (ptrs_size(N) + slices_size(N)) * 2 + std::max(temp_size(N), ssa_size(N));
}

// 计算 C = A*B mod (2^m+1)，必须保证 A 和 B < 2^m+1 且 C.size() >= m/64 + 1 且 m 是 2 的幂
static void mod_mul(std::span<const uint64_t> A, std::span<const uint64_t> B, std::span<uint64_t> C,
    size_type m, std::span<uint64_t> buffer) {
    size_type w = m / 64;

    auto copy_and_fill_zero = [w](std::span<const uint64_t> src, std::span<uint64_t> dst) {
        memcpy(dst.data(), src.data(), std::min(src.size(), w + 1) * sizeof(uint64_t));
        if (src.size() < w + 1)
            memset(&dst[src.size()], 0, (w + 1 - src.size()) * sizeof(uint64_t));
    };

    if (A.size() >= w + 1 && A[w] == 1) {
        if (B.data() != C.data())
            copy_and_fill_zero(B, C);
        mod_neg(C, m);
        return;
    }
    if (B.size() >= w + 1 && B[w] == 1) {
        if (A.data() != C.data())
            copy_and_fill_zero(A, C);
        mod_neg(C, m);
        return;
    }
    if ((A.size() + B.size()) * 64 < MIN_TOTAL_BITS) {
        auto res = _mul(A, B);
        mod(res, m);
        copy_and_fill_zero(res, C);
        return;
    }

    auto [N, K] = calc_slice(m);

    size_type n = N / 64, k = K / 64;
    size_type buffer_offset = 0;

    auto A_ptrs = buffer.subspan(buffer_offset, ptrs_size(N));
    buffer_offset += A_ptrs.size();

    auto A_slices = buffer.subspan(buffer_offset, slices_size(N));
    buffer_offset += A_slices.size();

    auto B_ptrs = buffer.subspan(buffer_offset, ptrs_size(N));
    buffer_offset += B_ptrs.size();

    auto B_slices = buffer.subspan(buffer_offset, slices_size(N));
    buffer_offset += B_slices.size();

    auto temp         = buffer.subspan(buffer_offset, temp_size(N));
    auto child_buffer = buffer.subspan(buffer_offset);
    // 复用 B_ptrs + B_slices + temp 的空间，可以证明空间足够
    auto acc_pos = buffer.subspan(A_ptrs.size() + A_slices.size(), acc_size(N, K));
    auto acc_neg = buffer.subspan(A_ptrs.size() + A_slices.size() + acc_pos.size(), acc_size(N, K));

    for (size_type i = 0; i < N; ++i) {
        A_ptrs[i] = B_ptrs[i] = i * (n + 1);
    }

    {
        size_type i = 0;
        for (; i < N && i * k < A.size(); ++i) {
            auto a   = A_slices.subspan(i * (n + 1), n + 1);
            auto A_i = A.subspan(i * k, std::min(k, A.size() - i * k));
            mod_mul_pow_of_two(A_i, a, i, N, temp);
        }
        if (i < N) {
            std::memset(
                &A_slices[i * (n + 1)], 0, (A_slices.size() - i * (n + 1)) * sizeof(uint64_t));
        }
        ntt(A_ptrs, A_slices, N, false, temp);

        for (i = 0; i < N && i * k < B.size(); ++i) {
            auto b   = B_slices.subspan(i * (n + 1), n + 1);
            auto B_i = B.subspan(i * k, std::min(k, B.size() - i * k));
            mod_mul_pow_of_two(B_i, b, i, N, temp);
        }
        if (i < N) {
            std::memset(
                &B_slices[i * (n + 1)], 0, (B_slices.size() - i * (n + 1)) * sizeof(uint64_t));
        }
        ntt(B_ptrs, B_slices, N, false, temp);
    }

    for (size_type i = 0; i < N; ++i) {
        auto a = A_slices.subspan(A_ptrs[i], n + 1);
        auto b = B_slices.subspan(B_ptrs[i], n + 1);
        mod_mul(a, b, a, N, child_buffer);
    }

    ntt(A_ptrs, A_slices, N, true, temp);
    // 两次位逆序置换后 A_ptrs 变回连续的

    for (size_type i = 1; i < N; ++i) {
        auto a = A_slices.subspan(i * (n + 1), n + 1);
        mod_div_pow_of_two(a, a, i, N, temp);
    }

    // 通过比较 A[i] 与 (2^N+1)/2 的大小判断符号，
    // A[i] < (2^N+1)/2 （即 A[i] <= 2^(N-1)）时是非负数，
    // 否则是负数，真实值为 -(2^N+1-A[i])
    auto check_is_neg = [](std::span<const uint64_t> a, size_type _N) -> bool {
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
    std::memset(acc_pos.data(), 0, acc_pos.size() * sizeof(uint64_t));
    std::memset(acc_neg.data(), 0, acc_neg.size() * sizeof(uint64_t));
    for (size_type i = 0; i < N; ++i) {
        auto a      = A_slices.subspan(i * (n + 1), n + 1);
        bool is_neg = check_is_neg(a, N);
        if (is_neg) {
            mod_neg(a, N);
        }
        auto target = (is_neg ? acc_neg : acc_pos).subspan(i * k, n + 1);
        utils::add(a, target, target);
    }
    mod(acc_pos, m);
    mod(acc_neg, m);
    mod_sub(acc_pos, acc_neg, C, m);
}

auto mul(std::span<const uint64_t> A, std::span<const uint64_t> B) -> std::vector<uint64_t> {
    static thread_local std::vector<uint64_t> buffer;

    size_type total_bits = (A.size() + B.size()) * 64;
    if (!(total_bits >= MIN_TOTAL_BITS))
        unreachable();

    size_type m = std::bit_ceil(total_bits);

    if (size_type buffer_required = ssa_size(m); buffer_required > buffer.size()) {
        buffer.clear();
        buffer.shrink_to_fit();
        buffer.resize(buffer_required);
    }

    std::vector<uint64_t> res(m / 64 + 1);
    mod_mul(A, B, res, m, buffer);
    utils::remove_leading_zero(res);
    return res;
}

}  // namespace bigint::mul::ssa
