#include <bit>
#include <cmath>
#include <immintrin.h>
#include <ranges>
#include <span>


#include "bigint/aligned_allocator.h"
#include "bigint/bigint_base.h"
#include "bigint/mul.h"


namespace bigint::mul::fft {

static void reverse(std::span<double> v_re, std::span<double> v_im) {
    const size_type n = v_re.size();
    if (n != v_im.size() || n < 4 || std::popcount(n) != 1 || std::countr_zero(n) % 2 != 0) {
        unreachable();
    } else if (n == 4) {
        return;
    }
    const size_type k0 = (n >> 1) | (n >> 2);
    for (size_type i = 1, j = n >> 2; i < n - 1; ++i) {
        if (i < j) {
            std::swap(v_re[i], v_re[j]);
            std::swap(v_im[i], v_im[j]);
        }
        size_type k = k0;
        while (j >= k) {
            j -= k;
            k >>= 2;
        }
        j += size_type(1) << std::countr_zero(k);
    }
}

// 输入必须是 4 的幂
static constexpr auto log4(size_type n) -> size_type {
    if (n == 0 || std::popcount(n) != 1 || std::countr_zero(n) % 2 != 0) {
        unreachable();
    }
    return std::countr_zero(n) / 2;
}

static inline void complex_mul_simd(
    __m256d a_re, __m256d a_im, __m256d b_re, __m256d b_im, __m256d& res_re, __m256d& res_im) {
    res_re = _mm256_fmsub_pd(a_re, b_re, _mm256_mul_pd(a_im, b_im));
    res_im = _mm256_fmadd_pd(a_re, b_im, _mm256_mul_pd(a_im, b_re));
}

// 层数转旋转因子数组的起始下标
// 从第 2 层开始，层内块长为 block = 4^layer，
// 块内 A 无需旋转，B C D 各需 sub = block / 4 个旋转因子，
// 故每层旋转因子数为 3 * 4^(layer-1)，
// 从 2 到 layer - 1 求和得 4^(layer-1) - 4
static constexpr auto layer_to_W_idx(size_type layer) -> size_type {
    if (layer < 2) {
        unreachable();
    }
    return (size_type(1) << (2 * (layer - 1))) - 4;
}

// 旋转因子表
alignas(32) static const auto W = [] {
    std::array<std::array<double, layer_to_W_idx(MAX_FFT_LAYER + 1)>, 2> res{};

    for (size_type layer = 2; layer <= MAX_FFT_LAYER; ++layer) {
        size_type base_idx = layer_to_W_idx(layer);
        size_type block = size_type(1) << (layer * 2), sub = block / 4;
        res[0][base_idx + 0 * sub] = res[0][base_idx + 1 * sub] = res[0][base_idx + 2 * sub] = 1;
        res[1][base_idx + 0 * sub] = res[1][base_idx + 1 * sub] = res[1][base_idx + 2 * sub] = 0;
        for (size_type k = 1; k < sub; ++k) {
            for (size_type t = 0; t < 3; ++t) {
                double    theta = 2 * std::numbers::pi / static_cast<double>(block)
                                  * static_cast<double>(k * (t + 1));
                size_type idx   = base_idx + t * sub + k;
                res[0][idx]     = std::cos(theta);
                res[1][idx]     = -std::sin(theta);
            }
        }
    }

    return res;
}();

// SIMD 基-4 fft，输入长度必须是 4 的幂且小于等于 MAX_FFT_LEN ，输入数组必须 32 字节对齐
template<bool rev> static void fft(std::span<double> v_re, std::span<double> v_im) {
    // simd 指令别名
    auto m_add   = [](__m256d a, __m256d b) -> __m256d { return _mm256_add_pd(a, b); };
    auto m_sub   = [](__m256d a, __m256d b) -> __m256d { return _mm256_sub_pd(a, b); };
    auto m_mul   = [](__m256d a, __m256d b) -> __m256d { return _mm256_mul_pd(a, b); };
    auto m_load  = [](const double* p) -> __m256d { return _mm256_load_pd(p); };
    auto m_store = [](double* p, __m256d a) -> void { _mm256_store_pd(p, a); };

    const size_type n = v_re.size();
    if (n != v_im.size() || n < 4 || std::popcount(n) != 1 || std::countr_zero(n) % 2 != 0) {
        unreachable();
    }
    const size_type total_layer = log4(n);
    if (total_layer > MAX_FFT_LAYER) {
        unreachable();
    }
    reverse(v_re, v_im);
    for (size_type i = 0; i < n; i += 4) {
        double A_re = v_re[i + 0] + v_re[i + 1] + v_re[i + 2] + v_re[i + 3];
        double A_im = v_im[i + 0] + v_im[i + 1] + v_im[i + 2] + v_im[i + 3];
        double B_re = v_re[i + 0] + v_im[i + 1] - v_re[i + 2] - v_im[i + 3];
        double B_im = v_im[i + 0] - v_re[i + 1] - v_im[i + 2] + v_re[i + 3];
        double C_re = v_re[i + 0] - v_re[i + 1] + v_re[i + 2] - v_re[i + 3];
        double C_im = v_im[i + 0] - v_im[i + 1] + v_im[i + 2] - v_im[i + 3];
        double D_re = v_re[i + 0] - v_im[i + 1] - v_re[i + 2] + v_im[i + 3];
        double D_im = v_im[i + 0] + v_re[i + 1] - v_im[i + 2] - v_re[i + 3];
        if constexpr (rev) {
            std::swap(B_re, D_re);
            std::swap(B_im, D_im);
        }
        v_re[i + 0] = A_re, v_im[i + 0] = A_im;
        v_re[i + 1] = B_re, v_im[i + 1] = B_im;
        v_re[i + 2] = C_re, v_im[i + 2] = C_im;
        v_re[i + 3] = D_re, v_im[i + 3] = D_im;
    }
    for (size_type layer = 2; layer <= total_layer; ++layer) {
        size_type block = 1 << (layer * 2), sub = block / 4;

        size_type base_W_idx = layer_to_W_idx(layer);
        for (size_type k = sub; k < block; k += 4) {
            size_type W_idx = base_W_idx + k - sub;
            __m256d   w_re = m_load(&W[0][W_idx]), w_im = m_load(&W[1][W_idx]);
            if constexpr (rev) {
                w_im = _mm256_xor_pd(w_im, _mm256_set1_pd(-0.0));
            }
            for (size_type i = 0; i < n; i += block) {
                double *p_re = &v_re[i + k], *p_im = &v_im[i + k];
                __m256d re = m_load(p_re), im = m_load(p_im);
                __m256d res_re, res_im;
                complex_mul_simd(re, im, w_re, w_im, res_re, res_im);
                m_store(p_re, res_re);
                m_store(p_im, res_im);
            }
        }

        for (size_type i = 0; i < n; i += block) {
            for (size_type k = 0; k < sub; k += 4) {
                size_type idx_A = i + k, idx_B = idx_A + sub, idx_C = idx_B + sub,
                          idx_D = idx_C + sub;

                __m256d A_re = m_load(&v_re[idx_A]), A_im = m_load(&v_im[idx_A]);
                __m256d B_re = m_load(&v_re[idx_B]), B_im = m_load(&v_im[idx_B]);
                __m256d C_re = m_load(&v_re[idx_C]), C_im = m_load(&v_im[idx_C]);
                __m256d D_re = m_load(&v_re[idx_D]), D_im = m_load(&v_im[idx_D]);

                __m256d nA_re = m_add(m_add(A_re, B_re), m_add(C_re, D_re));
                __m256d nA_im = m_add(m_add(A_im, B_im), m_add(C_im, D_im));
                __m256d nB_re = m_sub(m_add(A_re, B_im), m_add(C_re, D_im));
                __m256d nB_im = m_sub(m_sub(A_im, B_re), m_sub(C_im, D_re));
                __m256d nC_re = m_add(m_sub(A_re, B_re), m_sub(C_re, D_re));
                __m256d nC_im = m_add(m_sub(A_im, B_im), m_sub(C_im, D_im));
                __m256d nD_re = m_sub(m_sub(A_re, B_im), m_sub(C_re, D_im));
                __m256d nD_im = m_sub(m_add(A_im, B_re), m_add(C_im, D_re));
                if constexpr (rev) {
                    std::swap(nB_re, nD_re);
                    std::swap(nB_im, nD_im);
                }

                m_store(&v_re[idx_A], nA_re), m_store(&v_im[idx_A], nA_im);
                m_store(&v_re[idx_B], nB_re), m_store(&v_im[idx_B], nB_im);
                m_store(&v_re[idx_C], nC_re), m_store(&v_im[idx_C], nC_im);
                m_store(&v_re[idx_D], nD_re), m_store(&v_im[idx_D], nD_im);
            }
        }
    }

    if constexpr (rev) {
        __m256d inv_n = _mm256_set1_pd(1.0 / static_cast<double>(n));
        for (size_type i = 0; i < n; i += 4) {
            m_store(&v_re[i], m_mul(m_load(&v_re[i]), inv_n));
            m_store(&v_im[i], m_mul(m_load(&v_im[i]), inv_n));
        }
    }
}

template<typename Alloc>
static void int_vec_to_double(std::span<const uint64_t> X, std::vector<double, Alloc>& d_X) {
    d_X.clear();
    d_X.reserve(X.size() * 64 / FFT_DIGIT_BITS + 1);
    uint64_t tmp      = 0;
    int      tmp_bits = 0;
    for (auto digit : X) {
        int cur_digit_bits = 64;
        if (tmp_bits > 0) {
            int digit_bits = FFT_DIGIT_BITS - tmp_bits;
            d_X.emplace_back(((digit & ((1ull << digit_bits) - 1)) << tmp_bits) | tmp);
            digit >>= digit_bits;
            cur_digit_bits -= digit_bits;
            tmp      = 0;
            tmp_bits = 0;
        }
        while (cur_digit_bits >= FFT_DIGIT_BITS) {
            d_X.emplace_back(digit & ((1ull << FFT_DIGIT_BITS) - 1));
            digit >>= FFT_DIGIT_BITS;
            cur_digit_bits -= FFT_DIGIT_BITS;
        }
        if (cur_digit_bits > 0) {
            tmp      = digit;
            tmp_bits = cur_digit_bits;
        }
    }
    if (tmp_bits > 0) {
        d_X.emplace_back(tmp);
    }
}

auto mul(std::span<const uint64_t> A, std::span<const uint64_t> B) -> std::vector<uint64_t> {
    size_type tail_zero_A = 0, tail_zero_B = 0, leading_zero_A = 0, leading_zero_B = 0;
    for (auto digit : A) {
        if (digit != 0) {
            break;
        }
        ++tail_zero_A;
    }
    for (auto digit : B) {
        if (digit != 0) {
            break;
        }
        ++tail_zero_B;
    }
    if (tail_zero_A == A.size() || tail_zero_B == B.size()) {
        return {0};
    }
    for (auto digit : std::views::reverse(A)) {
        if (digit != 0) {
            break;
        }
        ++leading_zero_A;
    }
    for (auto digit : std::views::reverse(B)) {
        if (digit != 0) {
            break;
        }
        ++leading_zero_B;
    }

    std::vector<double, AlignedAllocator<double>> d_A, d_B;
    int_vec_to_double(A.subspan(tail_zero_A, A.size() - tail_zero_A - leading_zero_A), d_A);
    int_vec_to_double(B.subspan(tail_zero_B, B.size() - tail_zero_B - leading_zero_B), d_B);

    size_type fft_new_size = std::bit_ceil(d_A.size() + d_B.size() - 1);
    if (std::countr_zero(fft_new_size) % 2 == 1) {
        fft_new_size *= 2;
    }
    d_A.resize(fft_new_size), d_B.resize(fft_new_size);

    fft<false>(d_A, d_B);
    for (size_type i = 0; i < fft_new_size; i += 4) {
        __m256d re = _mm256_load_pd(&d_A[i]), im = _mm256_load_pd(&d_B[i]);
        __m256d n_re = _mm256_mul_pd(_mm256_add_pd(re, im), _mm256_sub_pd(re, im));
        __m256d n_im = _mm256_mul_pd(re, im);
        n_im         = _mm256_add_pd(n_im, n_im);
        _mm256_store_pd(&d_A[i], n_re);
        _mm256_store_pd(&d_B[i], n_im);
    }
    fft<true>(d_A, d_B);

    std::vector<uint64_t> res;
    res.resize(tail_zero_A + tail_zero_B);
    size_type leading_zero_res = 0;
    for (auto digit : std::views::reverse(d_B)) {
        if (std::llround(digit) != 0) {
            break;
        }
        ++leading_zero_res;
    }
    uint128_t digit          = 0;
    int       cur_digit_bits = 0;
    size_type i = 0, max_i = d_B.size() - leading_zero_res;
    while (i < max_i) {
        while (cur_digit_bits < 64 && i < max_i) {
            digit += static_cast<uint128_t>(std::llround(d_B[i]) / 2) << cur_digit_bits;
            cur_digit_bits += FFT_DIGIT_BITS;
            ++i;
        }
        while (cur_digit_bits >= 64) {
            res.emplace_back(digit);
            digit >>= 64;
            cur_digit_bits -= 64;
        }
    }
    if (digit) {
        do {  // NOLINT
            res.emplace_back(digit);
            digit >>= 64;
        } while (digit);
    } else {
        utils::remove_leading_zero(res);
    }

    return res;
}

}  // namespace bigint::mul::fft
