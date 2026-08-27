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

// SIMD 指令别名
static const auto m_add   = [](__m256d a, __m256d b) -> __m256d { return _mm256_add_pd(a, b); };
static const auto m_sub   = [](__m256d a, __m256d b) -> __m256d { return _mm256_sub_pd(a, b); };
static const auto m_mul   = [](__m256d a, __m256d b) -> __m256d { return _mm256_mul_pd(a, b); };
static const auto m_load  = [](const double* p) -> __m256d { return _mm256_load_pd(p); };
static const auto m_store = [](double* p, __m256d a) -> void { _mm256_store_pd(p, a); };

static inline void complex_mul_simd(
    __m256d a_re, __m256d a_im, __m256d b_re, __m256d b_im, __m256d& res_re, __m256d& res_im) {
    res_re = _mm256_fmsub_pd(a_re, b_re, m_mul(a_im, b_im));
    res_im = _mm256_fmadd_pd(a_re, b_im, m_mul(a_im, b_re));
}

static inline void complex_square_simd(__m256d re, __m256d im, __m256d& n_re, __m256d& n_im) {
    n_re      = m_mul(m_add(re, im), m_sub(re, im));
    __m256d t = m_mul(re, im);
    n_im      = m_add(t, t);
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

// 每层预计算 3 个象限（B/C/D）的旋转因子种子与步长，总大小 O(MAX_FFT_LAYER/2) = O(log n)
// 第 layer 层块长 len = 4^layer、sub = block / 4，
// 正变换时位置 p = (t+1) * sub + m（t=0,1,2 象限，m ∈ [0, sub)）的旋转因子为
//   ω^{(t+1)m}，其中 ω = e^(-2πi/block)。
// SIMD 一次处理 4 个连续旋转因子（m, m+1, m+2, m+3）：
//   种子 seed_t = [1, ω^{t+1}, ω^{2(t+1)}, ω^{3(t+1)}]（m 从 0 起）
//   步长 step_t = ω^{4(t+1)}（每处理完一组 m += 4 后整体相乘生成下一组）
// 逆变换只需对 seed/step 虚部取负（共轭）。
struct alignas(32) LayerTwiddles {
    std::array<std::array<double, 4>, 3> seed_re;
    std::array<std::array<double, 4>, 3> seed_im;
    std::array<double, 3>                step_re;
    std::array<double, 3>                step_im;
};

// std::cos / std::sin 不是 constexpr，因此用运行期静态初始化
alignas(32) static const auto layer_twiddles = [] {
    // 基-4 层最大层数 = MAX_FFT_LAYER/2（基-2 层数 MAX_FFT_LAYER = log2(MAX_FFT_LEN)）
    std::array<LayerTwiddles, MAX_FFT_LAYER / 2 + 1> res{};
    for (std::size_t layer = 2; layer <= MAX_FFT_LAYER / 2; ++layer) {
        auto&  twl   = res[layer];
        double theta = 2 * std::numbers::pi / static_cast<double>(1ull << (layer * 2));
        for (int t = 0; t < 3; ++t) {
            const auto mul = static_cast<double>(t + 1);
            for (int j = 0; j < 4; ++j) {
                twl.seed_re[t][j] = std::cos(theta * mul * j);
                twl.seed_im[t][j] = -std::sin(theta * mul * j);
            }
            twl.step_re[t] = std::cos(theta * mul * 4.0);
            twl.step_im[t] = -std::sin(theta * mul * 4.0);
        }
    }
    return res;
}();

struct alignas(32) R2LayerTwiddles {
    std::array<double, 4> seed_re, seed_im;
    double                step_re, step_im;
};

// 基-2 FFT 的奇数层（2*4^k 顶层）旋转因子表，层 5 到 <= MAX_FFT_LAYER 的最大奇数
// 索引 (layer-1)/2，表大小 (MAX_FFT_LAYER+1)/2
alignas(32) static const auto r2_odd_layer_twiddles = [] {
    std::array<R2LayerTwiddles, (MAX_FFT_LAYER + 1) / 2> res{};
    for (std::size_t layer = 5; layer <= MAX_FFT_LAYER; layer += 2) {
        auto&  twl   = res[(layer - 1) / 2];
        double theta = 2 * std::numbers::pi / static_cast<double>(1ull << layer);
        twl.seed_re  = {1.0, std::cos(theta), std::cos(2 * theta), std::cos(3 * theta)};
        twl.seed_im  = {0.0, -std::sin(theta), -std::sin(2 * theta), -std::sin(3 * theta)};
        twl.step_re  = std::cos(4 * theta);
        twl.step_im  = -std::sin(4 * theta);
    }
    return res;
}();

/**
 * DIF FFT：输入自然序，输出混合逆序。
 * DIT IFFT：输入混合逆序，输出自然序，无最后的缩放。
 * 混合逆序：长度为 4^k 时是基-4数字逆序，
 *           长度为 2*4^k 时先把偶索引放在前半，奇索引放在后半，
 *           再对前后分别进行基-4数字逆序。
 * 保证 IDIT(DIF(x)) / len(x) == x。
 */

// 长度 <= 8 ，不能完全由 SIMD 处理的基础情况
template<bool is_dif> static void fft_basecase(std::span<double> v_re, std::span<double> v_im) {
    // 4 点 DFT / IDFT
    constexpr auto dft4 = [](std::span<double> _v_re, std::span<double> _v_im) constexpr {
        double A_add_C_re = _v_re[0] + _v_re[2], A_add_C_im = _v_im[0] + _v_im[2];
        double B_add_D_re = _v_re[1] + _v_re[3], B_add_D_im = _v_im[1] + _v_im[3];
        double nA_re = A_add_C_re + B_add_D_re, nA_im = A_add_C_im + B_add_D_im;
        double nC_re = A_add_C_re - B_add_D_re, nC_im = A_add_C_im - B_add_D_im;
        double A_sub_C_re = _v_re[0] - _v_re[2], A_sub_C_im = _v_im[0] - _v_im[2];
        double B_sub_D_re = _v_re[1] - _v_re[3], B_sub_D_im = _v_im[1] - _v_im[3];
        double nB_re = A_sub_C_re + B_sub_D_im, nB_im = A_sub_C_im - B_sub_D_re;
        double nD_re = A_sub_C_re - B_sub_D_im, nD_im = A_sub_C_im + B_sub_D_re;
        if constexpr (!is_dif) {  // IDFT 相比 DFT 只需交换 B 和 D
            std::swap(nB_re, nD_re);
            std::swap(nB_im, nD_im);
        }
        _v_re[0] = nA_re, _v_im[0] = nA_im;
        _v_re[1] = nB_re, _v_im[1] = nB_im;
        _v_re[2] = nC_re, _v_im[2] = nC_im;
        _v_re[3] = nD_re, _v_im[3] = nD_im;
    };

    const std::size_t n = v_re.size();
    if (n != v_im.size())
        unreachable();
    switch (n) {
    case 1: return;
    case 2: {
        double A_re = v_re[0], B_re = v_re[1];
        double A_im = v_im[0], B_im = v_im[1];
        v_re[0] = A_re + B_re, v_re[1] = A_re - B_re;
        v_im[0] = A_im + B_im, v_im[1] = A_im - B_im;
        return;
    }
    case 4: {
        dft4(v_re, v_im);
        return;
    }
    case 8: {
        constexpr double w0   = std::numbers::sqrt2 / 2.0;
        const __m256d    w_re = _mm256_set_pd(-w0, 0, w0, 1);
        if constexpr (is_dif) {
            const __m256d w_im = _mm256_set_pd(-w0, -1, -w0, 0);

            __m256d A_re = m_load(&v_re[0]), B_re = m_load(&v_re[4]);
            __m256d A_im = m_load(&v_im[0]), B_im = m_load(&v_im[4]);
            __m256d nA_re = m_add(A_re, B_re), nA_im = m_add(A_im, B_im);
            __m256d nB_re = m_sub(A_re, B_re), nB_im = m_sub(A_im, B_im);
            complex_mul_simd(nB_re, nB_im, w_re, w_im, nB_re, nB_im);
            m_store(&v_re[0], nA_re), m_store(&v_re[4], nB_re);
            m_store(&v_im[0], nA_im), m_store(&v_im[4], nB_im);
            dft4(v_re.subspan(0, 4), v_im.subspan(0, 4));
            dft4(v_re.subspan(4, 4), v_im.subspan(4, 4));
        } else {
            const __m256d w_im = _mm256_set_pd(w0, 1, w0, 0);

            dft4(v_re.subspan(0, 4), v_im.subspan(0, 4));
            dft4(v_re.subspan(4, 4), v_im.subspan(4, 4));
            __m256d A_re = m_load(&v_re[0]), B_re = m_load(&v_re[4]);
            __m256d A_im = m_load(&v_im[0]), B_im = m_load(&v_im[4]);
            complex_mul_simd(B_re, B_im, w_re, w_im, B_re, B_im);
            __m256d nA_re = m_add(A_re, B_re), nA_im = m_add(A_im, B_im);
            __m256d nB_re = m_sub(A_re, B_re), nB_im = m_sub(A_im, B_im);
            m_store(&v_re[0], nA_re), m_store(&v_re[4], nB_re);
            m_store(&v_im[0], nA_im), m_store(&v_im[4], nB_im);
        }
        return;
    }
    default: unreachable();
    }
}

static void dif_fft_mixed(std::span<double> v_re, std::span<double> v_im) {
    const std::size_t n = v_re.size();
    if (n != v_im.size() || n == 0 || std::popcount(n) != 1)
        unreachable();
    if (n <= 8) {
        fft_basecase<true>(v_re, v_im);
        return;
    }
    std::size_t len = n;
    if (std::countr_zero(n) % 2) {
        std::size_t   half    = n / 2;
        std::size_t   i_mid   = half / 2;  // 中间组起始位置 = n/4
        auto&         twl     = r2_odd_layer_twiddles[(std::countr_zero(n) - 1) / 2];
        const __m256d seed_re = m_load(twl.seed_re.data()), seed_im = m_load(twl.seed_im.data());
        const __m256d step_re = _mm256_set1_pd(twl.step_re), step_im = _mm256_set1_pd(twl.step_im);
        // 前半从 seed_first 递推；后半复用 seed_first 乘 -i 得中间组种子（ω^{n/4}=-i，精确），
        // 两段最坏递推步数均 n/16 减半，缓解误差累积，且不额外存储
        for (std::size_t seg = 0; seg < 2; ++seg) {
            __m256d w_re, w_im;
            if (seg == 0) {
                w_re = seed_re;
                w_im = seed_im;
            } else {  // seed_mid = -i * seed_first：交换实虚、虚部取负
                w_re = seed_im;
                w_im = _mm256_xor_pd(seed_re, _mm256_set1_pd(-0.0));
            }
            const std::size_t i0 = seg == 1 ? i_mid : 0, i1 = seg == 1 ? half : i_mid;
            for (std::size_t i = i0; i < i1; i += 4) {
                __m256d A_re = m_load(&v_re[i]), B_re = m_load(&v_re[half + i]);
                __m256d A_im = m_load(&v_im[i]), B_im = m_load(&v_im[half + i]);
                __m256d nA_re = m_add(A_re, B_re), nA_im = m_add(A_im, B_im);
                __m256d nB_re = m_sub(A_re, B_re), nB_im = m_sub(A_im, B_im);
                complex_mul_simd(nB_re, nB_im, w_re, w_im, nB_re, nB_im);
                m_store(&v_re[i], nA_re), m_store(&v_re[half + i], nB_re);
                m_store(&v_im[i], nA_im), m_store(&v_im[half + i], nB_im);
                complex_mul_simd(w_re, w_im, step_re, step_im, w_re, w_im);  // 推进下一组
            }
        }
        len = half;
    }
    for (std::size_t layer = std::countr_zero(len) / 2, sub = len / 4; layer > 1; --layer) {
        for (std::size_t i = 0; i < n; i += len) {
            for (std::size_t m = 0; m < sub; m += 4) {
                std::size_t idx_A = i + m, idx_B = idx_A + sub, idx_C = idx_B + sub,
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

                m_store(&v_re[idx_A], nA_re), m_store(&v_im[idx_A], nA_im);
                m_store(&v_re[idx_B], nB_re), m_store(&v_im[idx_B], nB_im);
                m_store(&v_re[idx_C], nC_re), m_store(&v_im[idx_C], nC_im);
                m_store(&v_re[idx_D], nD_re), m_store(&v_im[idx_D], nD_im);
            }
        }
        const auto& twl = layer_twiddles[layer];
        for (std::size_t i = 0; i < n; i += len) {
            for (std::size_t t = 0; t < 3; ++t) {
                __m256d w_re = m_load(twl.seed_re[t].data()), w_im = m_load(twl.seed_im[t].data());
                __m256d step_re       = _mm256_set1_pd(twl.step_re[t]),
                        step_im       = _mm256_set1_pd(twl.step_im[t]);
                const std::size_t pos = i + (t + 1) * sub;
                for (std::size_t m = 0; m < sub; m += 4) {
                    double *p_re = &v_re[pos + m], *p_im = &v_im[pos + m];
                    __m256d res_re, res_im;
                    complex_mul_simd(m_load(p_re), m_load(p_im), w_re, w_im, res_re, res_im);
                    m_store(p_re, res_re), m_store(p_im, res_im);
                    complex_mul_simd(w_re, w_im, step_re, step_im, w_re, w_im);  // 生成下一组
                }
            }
        }
        len = sub, sub /= 4;
    }
    for (std::size_t i = 0; i < n; i += 4 * 4) {
        std::size_t idx_A = i + 4 * 0, idx_B = i + 4 * 1, idx_C = i + 4 * 2, idx_D = i + 4 * 3;

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

        transpose_4x4d(nA_re, nB_re, nC_re, nD_re);
        transpose_4x4d(nA_im, nB_im, nC_im, nD_im);
        m_store(&v_re[idx_A], nA_re), m_store(&v_im[idx_A], nA_im);
        m_store(&v_re[idx_B], nB_re), m_store(&v_im[idx_B], nB_im);
        m_store(&v_re[idx_C], nC_re), m_store(&v_im[idx_C], nC_im);
        m_store(&v_re[idx_D], nD_re), m_store(&v_im[idx_D], nD_im);
    }
}

static void dit_ifft_mixed(std::span<double> v_re, std::span<double> v_im) {
    const std::size_t n = v_re.size();
    if (n != v_im.size() || n == 0 || std::popcount(n) != 1)
        unreachable();
    if (n <= 8) {
        fft_basecase<false>(v_re, v_im);
        return;
    }
    for (std::size_t i = 0; i < n; i += 4 * 4) {
        std::size_t idx_A = i + 4 * 0, idx_B = i + 4 * 1, idx_C = i + 4 * 2, idx_D = i + 4 * 3;

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
        __m256d nD_re = m_add(A_sub_C_re, B_sub_D_im), nD_im = m_sub(A_sub_C_im, B_sub_D_re);
        __m256d nB_re = m_sub(A_sub_C_re, B_sub_D_im), nB_im = m_add(A_sub_C_im, B_sub_D_re);

        transpose_4x4d(nA_re, nB_re, nC_re, nD_re);
        transpose_4x4d(nA_im, nB_im, nC_im, nD_im);
        m_store(&v_re[idx_A], nA_re), m_store(&v_im[idx_A], nA_im);
        m_store(&v_re[idx_B], nB_re), m_store(&v_im[idx_B], nB_im);
        m_store(&v_re[idx_C], nC_re), m_store(&v_im[idx_C], nC_im);
        m_store(&v_re[idx_D], nD_re), m_store(&v_im[idx_D], nD_im);
    }
    const std::size_t total_r4_layers = std::countr_zero(n) / 2;
    for (std::size_t layer = 2, len = 16, sub = 4; layer <= total_r4_layers; ++layer) {
        const auto& twl = layer_twiddles[layer];
        for (std::size_t i = 0; i < n; i += len) {
            for (std::size_t t = 0; t < 3; ++t) {
                __m256d w_re = m_load(twl.seed_re[t].data()),
                        w_im = _mm256_xor_pd(m_load(twl.seed_im[t].data()), _mm256_set1_pd(-0.0));
                __m256d step_re       = _mm256_set1_pd(twl.step_re[t]),
                        step_im       = _mm256_set1_pd(-twl.step_im[t]);
                const std::size_t pos = i + (t + 1) * sub;
                for (std::size_t m = 0; m < sub; m += 4) {
                    double *p_re = &v_re[pos + m], *p_im = &v_im[pos + m];
                    __m256d res_re, res_im;
                    complex_mul_simd(m_load(p_re), m_load(p_im), w_re, w_im, res_re, res_im);
                    m_store(p_re, res_re), m_store(p_im, res_im);
                    complex_mul_simd(w_re, w_im, step_re, step_im, w_re, w_im);
                }
            }
        }
        for (std::size_t i = 0; i < n; i += len) {
            for (std::size_t m = 0; m < sub; m += 4) {
                std::size_t idx_A = i + m, idx_B = idx_A + sub, idx_C = idx_B + sub,
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
                __m256d nD_re = m_add(A_sub_C_re, B_sub_D_im),
                        nD_im = m_sub(A_sub_C_im, B_sub_D_re);
                __m256d nB_re = m_sub(A_sub_C_re, B_sub_D_im),
                        nB_im = m_add(A_sub_C_im, B_sub_D_re);

                m_store(&v_re[idx_A], nA_re), m_store(&v_im[idx_A], nA_im);
                m_store(&v_re[idx_B], nB_re), m_store(&v_im[idx_B], nB_im);
                m_store(&v_re[idx_C], nC_re), m_store(&v_im[idx_C], nC_im);
                m_store(&v_re[idx_D], nD_re), m_store(&v_im[idx_D], nD_im);
            }
        }
        sub = len, len *= 4;
    }
    if (std::countr_zero(n) % 2) {  // n = 2*4^k
        std::size_t   half    = n / 2;
        std::size_t   i_mid   = half / 2;
        auto&         twl     = r2_odd_layer_twiddles[(std::countr_zero(n) - 1) / 2];
        const __m256d seed_re = m_load(twl.seed_re.data()), seed_im = m_load(twl.seed_im.data());
        const __m256d step_re = _mm256_set1_pd(twl.step_re), step_im = _mm256_set1_pd(-twl.step_im);
        for (std::size_t seg = 0; seg < 2; ++seg) {
            __m256d w_re, w_im;
            if (seg == 0) {
                w_re = seed_re;
                w_im = _mm256_xor_pd(seed_im, _mm256_set1_pd(-0.0));
            } else {  // seed_mid_inv = i * conj(seed_first)：交换实虚（无符号翻转）
                w_re = seed_im;
                w_im = seed_re;
            }
            const std::size_t i0 = seg == 1 ? i_mid : 0, i1 = seg == 1 ? half : i_mid;
            for (std::size_t i = i0; i < i1; i += 4) {
                __m256d A_re = m_load(&v_re[i]), B_re = m_load(&v_re[half + i]);
                __m256d A_im = m_load(&v_im[i]), B_im = m_load(&v_im[half + i]);
                complex_mul_simd(B_re, B_im, w_re, w_im, B_re, B_im);
                __m256d nA_re = m_add(A_re, B_re), nA_im = m_add(A_im, B_im);
                __m256d nB_re = m_sub(A_re, B_re), nB_im = m_sub(A_im, B_im);
                m_store(&v_re[i], nA_re), m_store(&v_re[half + i], nB_re);
                m_store(&v_im[i], nA_im), m_store(&v_im[half + i], nB_im);
                complex_mul_simd(w_re, w_im, step_re, step_im, w_re, w_im);
            }
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

    std::size_t tail_zero_A = 0, tail_zero_B = 0, leading_zero_A = 0, leading_zero_B = 0;
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
    std::size_t                                   fft_new_size =
        std::bit_ceil(ceil_div<std::size_t>(A.size() * 64, digit_bits)
                      + ceil_div<std::size_t>(B.size() * 64, digit_bits) - 1);
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

    dif_fft_mixed(v_re, v_im);
    for (std::size_t i = 0; i < fft_new_size; i += 4) {
        __m256d re = m_load(&v_re[i]), im = m_load(&v_im[i]);
        complex_square_simd(re, im, re, im);
        m_store(&v_re[i], re), m_store(&v_im[i], im);
    }
    dit_ifft_mixed(v_re, v_im);

    std::vector<uint64_t> res;
    res.resize(tail_zero_A + tail_zero_B);
    std::size_t leading_zero_res = 0;
    for (auto digit : std::views::reverse(v_im)) {
        if (std::llround(digit) != 0) {
            break;
        }
        ++leading_zero_res;
    }
    uint128_t    digit          = 0;
    int          cur_digit_bits = 0;
    std::size_t  i = 0, max_i = v_im.size() - leading_zero_res;
    const double scale = 1.0 / static_cast<double>(2 * fft_new_size);
    while (i < max_i) {
        while (cur_digit_bits < 64 && i < max_i) {
            digit += static_cast<uint128_t>(std::llround(v_im[i] * scale)) << cur_digit_bits;
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
