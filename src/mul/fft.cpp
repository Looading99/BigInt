#include <bit>
#include <cmath>
#include <immintrin.h>
#include <numbers>
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
        // j += size_type(1) << std::countr_zero(k);
        j += k & (k >> 1);
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

static inline void complex_square_simd(__m256d re, __m256d im, __m256d& n_re, __m256d& n_im) {
    n_re      = _mm256_mul_pd(_mm256_add_pd(re, im), _mm256_sub_pd(re, im));
    __m256d t = _mm256_mul_pd(re, im);
    n_im      = _mm256_add_pd(t, t);
}

static inline void transpose_4x4d(__m256d& r0, __m256d& r1, __m256d& r2, __m256d& r3) {
    __m256d t0 = _mm256_unpacklo_pd(r0, r1);  // [r0[0], r1[0], r0[2], r1[2]]
    __m256d t1 = _mm256_unpackhi_pd(r0, r1);  // [r0[1], r1[1], r0[3], r1[3]]
    __m256d t2 = _mm256_unpacklo_pd(r2, r3);  // [r2[0], r3[0], r2[2], r3[2]]
    __m256d t3 = _mm256_unpackhi_pd(r2, r3);  // [r2[1], r3[1], r2[3], r3[3]]

    r0 = _mm256_permute2f128_pd(t0, t2, 0x20);  // [r0[0], r1[0], r2[0], r3[0]]
    r1 = _mm256_permute2f128_pd(t1, t3, 0x20);  // [r0[1], r1[1], r2[1], r3[1]]
    r2 = _mm256_permute2f128_pd(t0, t2, 0x31);  // [r0[2], r1[2], r2[2], r3[2]]
    r3 = _mm256_permute2f128_pd(t1, t3, 0x31);  // [r0[3], r1[3], r2[3], r3[3]]
}

// 每层预计算 3 个象限（B/C/D）的旋转因子种子与步长，总大小 O(MAX_FFT_LAYER) = O(log n)
// 第 layer 层块长 block = 4^layer、sub = block / 4，
// 正变换（rev=false）时位置 p = (t+1) * sub + m（t=0,1,2 象限，m ∈ [0, sub)）的旋转因子为
//   ω^{(t+1)m}，其中 ω = e^(-2πi/block)。
// SIMD 一次处理 4 个连续旋转因子（m, m+1, m+2, m+3）：
//   种子 seed_t = [1, ω^{t+1}, ω^{2(t+1)}, ω^{3(t+1)}]（m 从 0 起）
//   步长 step_t = ω^{4(t+1)}（每处理完一组 m += 4 后整体相乘生成下一组）
// 逆变换只需对 seed/step 虚部取负（共轭），见 fft<rev> 内的 _mm256_xor_pd。
struct alignas(32) LayerTwiddles {
    std::array<std::array<double, 4>, 3> seed_re;
    std::array<std::array<double, 4>, 3> seed_im;
    std::array<double, 3>                step_re;
    std::array<double, 3>                step_im;
};

// std::cos / std::sin 不是 constexpr，因此用运行期静态初始化
alignas(32) static const auto layer_twiddles = [] {
    std::array<LayerTwiddles, MAX_FFT_LAYER + 1> res{};
    for (size_type layer = 2; layer <= MAX_FFT_LAYER; ++layer) {
        const size_type block = 1ull << (layer * 2);
        const double    theta = 2 * std::numbers::pi / static_cast<double>(block);
        for (size_type t = 0; t < 3; ++t) {
            const auto mul = static_cast<double>(t + 1);
            for (size_type j = 0; j < 4; ++j) {
                res[layer].seed_re[t][j] = std::cos(theta * mul * static_cast<double>(j));
                res[layer].seed_im[t][j] = -std::sin(theta * mul * static_cast<double>(j));
            }
            res[layer].step_re[t] = std::cos(theta * mul * 4.0);
            res[layer].step_im[t] = -std::sin(theta * mul * 4.0);
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
    if (n != v_im.size() || n == 0)
        unreachable();
    if (n == 1)
        return;
    if (n > MAX_FFT_LEN || std::popcount(n) != 1 || std::countr_zero(n) % 2 != 0)
        unreachable();
    if (n == 4) {
        size_type i = 0;

        double A_add_C_re = v_re[i + 0] + v_re[i + 2], A_add_C_im = v_im[i + 0] + v_im[i + 2];
        double B_add_D_re = v_re[i + 1] + v_re[i + 3], B_add_D_im = v_im[i + 1] + v_im[i + 3];
        double nA_re = A_add_C_re + B_add_D_re, nA_im = A_add_C_im + B_add_D_im;
        double nC_re = A_add_C_re - B_add_D_re, nC_im = A_add_C_im - B_add_D_im;
        double A_sub_C_re = v_re[i + 0] - v_re[i + 2], A_sub_C_im = v_im[i + 0] - v_im[i + 2];
        double B_sub_D_re = v_re[i + 1] - v_re[i + 3], B_sub_D_im = v_im[i + 1] - v_im[i + 3];
        double nB_re = A_sub_C_re + B_sub_D_im, nB_im = A_sub_C_im - B_sub_D_re;
        double nD_re = A_sub_C_re - B_sub_D_im, nD_im = A_sub_C_im + B_sub_D_re;
        if constexpr (rev) {
            std::swap(nB_re, nD_re);
            std::swap(nB_im, nD_im);
            nA_re /= 4, nB_re /= 4, nC_re /= 4, nD_re /= 4;
            nA_im /= 4, nB_im /= 4, nC_im /= 4, nD_im /= 4;
        }
        v_re[i + 0] = nA_re, v_im[i + 0] = nA_im;
        v_re[i + 1] = nB_re, v_im[i + 1] = nB_im;
        v_re[i + 2] = nC_re, v_im[i + 2] = nC_im;
        v_re[i + 3] = nD_re, v_im[i + 3] = nD_im;
        return;
    }
    const size_type total_layer = log4(n);
    reverse(v_re, v_im);
    for (size_type i = 0; i < n; i += 4 * 4) {
        size_type idx_A = i + 4 * 0, idx_B = i + 4 * 1, idx_C = i + 4 * 2, idx_D = i + 4 * 3;

        __m256d A_re = m_load(&v_re[idx_A]), B_re = m_load(&v_re[idx_B]),
                C_re = m_load(&v_re[idx_C]), D_re = m_load(&v_re[idx_D]);
        __m256d A_im = m_load(&v_im[idx_A]), B_im = m_load(&v_im[idx_B]),
                C_im = m_load(&v_im[idx_C]), D_im = m_load(&v_im[idx_D]);
        transpose_4x4d(A_re, B_re, C_re, D_re);
        transpose_4x4d(A_im, B_im, C_im, D_im);

        __m256d A_add_C_re = m_add(A_re, C_re), A_add_C_im = m_add(A_im, C_im);
        __m256d B_add_D_re = m_add(B_re, D_re), B_add_D_im = m_add(B_im, D_im);
        __m256d nA_re = m_add(A_add_C_re, B_add_D_re), nA_im = m_add(A_add_C_im, B_add_D_im);
        __m256d nC_re = m_sub(A_add_C_re, B_add_D_re), nC_im = m_sub(A_add_C_im, B_add_D_im);

        __m256d A_sub_C_re = m_sub(A_re, C_re), A_sub_C_im = m_sub(A_im, C_im);
        __m256d B_sub_D_re = m_sub(B_re, D_re), B_sub_D_im = m_sub(B_im, D_im);
        __m256d nB_re = m_add(A_sub_C_re, B_sub_D_im), nB_im = m_sub(A_sub_C_im, B_sub_D_re);
        __m256d nD_re = m_sub(A_sub_C_re, B_sub_D_im), nD_im = m_add(A_sub_C_im, B_sub_D_re);
        if constexpr (rev) {
            std::swap(nB_re, nD_re);
            std::swap(nB_im, nD_im);
        }

        transpose_4x4d(nA_re, nB_re, nC_re, nD_re);
        transpose_4x4d(nA_im, nB_im, nC_im, nD_im);
        m_store(&v_re[idx_A], nA_re), m_store(&v_re[idx_B], nB_re);
        m_store(&v_re[idx_C], nC_re), m_store(&v_re[idx_D], nD_re);
        m_store(&v_im[idx_A], nA_im), m_store(&v_im[idx_B], nB_im);
        m_store(&v_im[idx_C], nC_im), m_store(&v_im[idx_D], nD_im);
    }
    for (size_type layer = 2; layer <= total_layer; ++layer) {
        size_type block = 1ull << (layer * 2), sub = block / 4;

        // 完全按输入数组顺序扫描（i 外层/m 内层），每块重新生成 3 个象限的旋转因子链，
        // 不预计算缓冲
        const auto& twl = layer_twiddles[layer];
        for (size_type i = 0; i < n; i += block) {
            for (size_type t = 0; t < 3; ++t) {
                __m256d w_re = m_load(twl.seed_re[t].data()), w_im = m_load(twl.seed_im[t].data());
                __m256d step_re = _mm256_set1_pd(twl.step_re[t]),
                        step_im = _mm256_set1_pd(twl.step_im[t]);
                if constexpr (rev) {  // 逆变换取共轭
                    w_im    = _mm256_xor_pd(w_im, _mm256_set1_pd(-0.0));
                    step_im = _mm256_xor_pd(step_im, _mm256_set1_pd(-0.0));
                }
                const size_type pos = (t + 1) * sub;
                for (size_type m = 0; m < sub; m += 4) {
                    double *p_re = &v_re[i + pos + m], *p_im = &v_im[i + pos + m];
                    __m256d res_re, res_im;
                    complex_mul_simd(m_load(p_re), m_load(p_im), w_re, w_im, res_re, res_im);
                    m_store(p_re, res_re);
                    m_store(p_im, res_im);
                    complex_mul_simd(w_re, w_im, step_re, step_im, w_re, w_im);  // 生成下一组
                }
            }
        }

        for (size_type i = 0; i < n; i += block) {
            for (size_type m = 0; m < sub; m += 4) {
                size_type idx_A = i + m, idx_B = idx_A + sub, idx_C = idx_B + sub,
                          idx_D = idx_C + sub;

                __m256d A_re = m_load(&v_re[idx_A]), B_re = m_load(&v_re[idx_B]),
                        C_re = m_load(&v_re[idx_C]), D_re = m_load(&v_re[idx_D]);
                __m256d A_im = m_load(&v_im[idx_A]), B_im = m_load(&v_im[idx_B]),
                        C_im = m_load(&v_im[idx_C]), D_im = m_load(&v_im[idx_D]);

                __m256d A_add_C_re = m_add(A_re, C_re), A_add_C_im = m_add(A_im, C_im);
                __m256d B_add_D_re = m_add(B_re, D_re), B_add_D_im = m_add(B_im, D_im);
                __m256d nA_re = m_add(A_add_C_re, B_add_D_re),
                        nA_im = m_add(A_add_C_im, B_add_D_im);
                __m256d nC_re = m_sub(A_add_C_re, B_add_D_re),
                        nC_im = m_sub(A_add_C_im, B_add_D_im);

                __m256d A_sub_C_re = m_sub(A_re, C_re), A_sub_C_im = m_sub(A_im, C_im);
                __m256d B_sub_D_re = m_sub(B_re, D_re), B_sub_D_im = m_sub(B_im, D_im);
                __m256d nB_re = m_add(A_sub_C_re, B_sub_D_im),
                        nB_im = m_sub(A_sub_C_im, B_sub_D_re);
                __m256d nD_re = m_sub(A_sub_C_re, B_sub_D_im),
                        nD_im = m_add(A_sub_C_im, B_sub_D_re);
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
        }
        for (size_type i = 0; i < n; i += 4) {
            m_store(&v_im[i], m_mul(m_load(&v_im[i]), inv_n));
        }
    }
}

template<typename Alloc>
static void vec64_to_double(
    std::span<const uint64_t> X, std::vector<double, Alloc>& d_X, int digit_bits) {
    d_X.clear();
    uint64_t tmp      = 0;
    int      tmp_bits = 0;
    for (auto digit : X) {
        int cur_digit_bits = 64;
        if (tmp_bits > 0) {
            int bits = digit_bits - tmp_bits;
            d_X.emplace_back(((digit & ((1ull << bits) - 1)) << tmp_bits) | tmp);
            digit >>= bits;
            cur_digit_bits -= bits;
            tmp      = 0;
            tmp_bits = 0;
        }
        while (cur_digit_bits >= digit_bits) {
            d_X.emplace_back(digit & ((1ull << digit_bits) - 1));
            digit >>= digit_bits;
            cur_digit_bits -= digit_bits;
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

auto mul(std::span<const uint64_t> A, std::span<const uint64_t> B, int digit_bits)
    -> std::vector<uint64_t> {
    bool a_is_b = A.data() == B.data();

    size_type tail_zero_A = 0, tail_zero_B = 0, leading_zero_A = 0, leading_zero_B = 0;
    for (auto digit : A) {
        if (digit != 0)
            break;
        ++tail_zero_A;
    }
    if (a_is_b) {
        tail_zero_B = tail_zero_A;
    } else {
        for (auto digit : B) {
            if (digit != 0)
                break;
            ++tail_zero_B;
        }
    }
    if (tail_zero_A == A.size() || tail_zero_B == B.size()) {
        return {0};
    }
    for (auto digit : std::views::reverse(A)) {
        if (digit != 0)
            break;
        ++leading_zero_A;
    }
    if (a_is_b) {
        leading_zero_B = leading_zero_A;
    } else {
        for (auto digit : std::views::reverse(B)) {
            if (digit != 0)
                break;
            ++leading_zero_B;
        }
    }

    A = A.subspan(tail_zero_A, A.size() - tail_zero_A - leading_zero_A);
    B = B.subspan(tail_zero_B, B.size() - tail_zero_B - leading_zero_B);

    // 实际输入总比特数（裁剪后按 64 位块计数），digit_bits=0 时自动查表选择
    if (digit_bits == 0) {
        digit_bits = digit_bits_for_total_bits((A.size() + B.size()) * 64);
    }

    std::vector<double, AlignedAllocator<double>> v_re, v_im;
    size_type fft_new_size = pow4ceil(ceil_div<size_type>(A.size() * 64, digit_bits)
                                      + ceil_div<size_type>(B.size() * 64, digit_bits) - 1);
    v_re.reserve(fft_new_size);
    vec64_to_double(A, v_re, digit_bits);
    v_re.resize(fft_new_size);
    if (a_is_b) {
        v_im = v_re;
    } else {
        v_im.reserve(fft_new_size);
        vec64_to_double(B, v_im, digit_bits);
        v_im.resize(fft_new_size);
    }


    fft<false>(v_re, v_im);
    for (size_type i = 0; i < fft_new_size; i += 4) {
        __m256d re = _mm256_load_pd(&v_re[i]), im = _mm256_load_pd(&v_im[i]);
        complex_square_simd(re, im, re, im);
        _mm256_store_pd(&v_re[i], re);
        _mm256_store_pd(&v_im[i], im);
    }
    fft<true>(v_re, v_im);

    std::vector<uint64_t> res;
    res.resize(tail_zero_A + tail_zero_B);
    size_type leading_zero_res = 0;
    for (auto digit : std::views::reverse(v_im)) {
        if (std::llround(digit) != 0) {
            break;
        }
        ++leading_zero_res;
    }
    uint128_t digit          = 0;
    int       cur_digit_bits = 0;
    size_type i = 0, max_i = v_im.size() - leading_zero_res;
    while (i < max_i) {
        while (cur_digit_bits < 64 && i < max_i) {
            digit += static_cast<uint128_t>(std::llround(v_im[i]) / 2) << cur_digit_bits;
            cur_digit_bits += digit_bits;
            ++i;
        }
        while (cur_digit_bits >= 64) {
            res.emplace_back(digit);
            digit >>= 64;
            cur_digit_bits -= 64;
        }
    }
    if (digit) {
        do {
            res.emplace_back(digit);
            digit >>= 64;
        } while (digit);
    } else {
        utils::remove_leading_zero(res);
    }

    return res;
}

}  // namespace bigint::mul::fft
