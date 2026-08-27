#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <immintrin.h>
#include <mutex>
#include <thread>
#include <variant>
#include <vector>


#include "bigint/mul.h"
#include "bigint/ntt_base.h"


namespace bigint {

namespace mul::ntt_multithread {

class NTTThreadPool {
public:
    struct BaseTask {
    protected:
        std::size_t total, block_size;
    };

    // 交织式多模数 NTT
    struct IMMNTTTask : BaseTask {
        std::span<uint32_t> v;
        std::size_t         len;
        bool                rev;

        static constexpr std::size_t THRESHOLD           = 1024;
        static constexpr std::size_t BLOCK_THREAD_FACTOR = 4;

        void init(std::size_t total_threads) {
            if (total == 0) {
                total      = v.size() / 2 / ntt::NUM_PRIMES;
                block_size = std::max(1ull, total / (total_threads * BLOCK_THREAD_FACTOR));
            }
        }

        auto run(std::size_t cur_block) -> bool {
            std::size_t start = cur_block * block_size, end = std::min(start + block_size, total);
            if (start >= end) {
                return false;
            }
            const std::size_t half = len / 2;
            const __m128i     simd_step =
                ntt::mont::vec_to_simd(ntt::steps[rev][std::countr_zero(half)]);
            __m128i simd_cur = ntt::mont::simd_one;
            if (std::size_t j0 = start & (half - 1); j0 != 0) {
                simd_cur = ntt::mont::fast_pow(simd_step, j0);
            }
            for (std::size_t cnt = start; cnt < end; ++cnt) {
                // 等价于 j = cnt % half, i = (cnt / half) * len
                std::size_t j = cnt & (half - 1), i = (cnt - j) << 1;
                if (j == 0) {
                    simd_cur = ntt::mont::simd_one;
                }
                std::size_t idx_a = (i + j) * ntt::NUM_PRIMES;
                std::size_t idx_b = (i + j + half) * ntt::NUM_PRIMES;
                __m128i simd_a = ntt::mont::vec_to_simd({v[idx_a + 0], v[idx_a + 1], v[idx_a + 2]});
                __m128i simd_b = ntt::mont::simd_mul(
                    simd_cur, ntt::mont::vec_to_simd({v[idx_b + 0], v[idx_b + 1], v[idx_b + 2]}));
                // __m256i simd_add_sub;
                // ntt::mont::simd_add_and_sub(simd_a, simd_b, simd_add_sub);
                // __m128i simd_a_add_b = _mm256_castsi256_si128(simd_add_sub);
                // __m128i simd_a_sub_b = _mm256_extracti128_si256(simd_add_sub, 1);
                __m128i simd_a_add_b = ntt::mont::simd_add(simd_a, simd_b);
                __m128i simd_a_sub_b = ntt::mont::simd_sub(simd_a, simd_b);
                ntt::mont::simd_store3(&v[idx_a], simd_a_add_b);
                ntt::mont::simd_store3(&v[idx_b], simd_a_sub_b);
                simd_cur = ntt::mont::simd_mul(simd_cur, simd_step);
            }
            return end != total;
        }
    };

    struct IMMMulNumTask : BaseTask {
        std::span<uint32_t>                   v;
        std::array<uint32_t, ntt::NUM_PRIMES> nums;

        static constexpr std::size_t THRESHOLD           = 2048;
        static constexpr std::size_t BLOCK_THREAD_FACTOR = 4;

        void init(std::size_t total_threads) {
            if (total == 0) {
                total      = v.size() / ntt::NUM_PRIMES;
                block_size = std::max(1ull, total / (total_threads * BLOCK_THREAD_FACTOR));
            }
        }

        auto run(std::size_t cur_block) -> bool {
            std::size_t start = cur_block * block_size, end = std::min(start + block_size, total);
            if (start >= end) {
                return false;
            }
            __m128i simd_nums = ntt::mont::vec_to_simd(nums);
            for (std::size_t i = start * ntt::NUM_PRIMES; i < end * ntt::NUM_PRIMES;
                i += ntt::NUM_PRIMES) {
                __m128i simd_a = ntt::mont::vec_to_simd({v[i + 0], v[i + 1], v[i + 2]});
                __m128i simd_r = ntt::mont::simd_mul(simd_a, simd_nums);
                ntt::mont::simd_store3(&v[i], simd_r);
            }
            return end != total;
        }
    };

    struct IMMMulVecTask : BaseTask {
        std::span<uint32_t>       v1;
        std::span<const uint32_t> v2;

        static constexpr std::size_t THRESHOLD           = 2048;
        static constexpr std::size_t BLOCK_THREAD_FACTOR = 4;

        void init(std::size_t total_threads) {
            if (total == 0) {
                total      = v1.size() / ntt::NUM_PRIMES;
                block_size = std::max(1ull, total / (total_threads * BLOCK_THREAD_FACTOR));
            }
        }

        auto run(std::size_t cur_block) -> bool {
            std::size_t start = cur_block * block_size, end = std::min(start + block_size, total);
            if (start >= end) {
                return false;
            }
            for (std::size_t i = start * ntt::NUM_PRIMES; i < end * ntt::NUM_PRIMES;
                i += ntt::NUM_PRIMES) {
                __m128i simd_a = ntt::mont::vec_to_simd({v1[i + 0], v1[i + 1], v1[i + 2]});
                __m128i simd_b = ntt::mont::vec_to_simd({v2[i + 0], v2[i + 1], v2[i + 2]});
                __m128i simd_r = ntt::mont::simd_mul(simd_a, simd_b);
                ntt::mont::simd_store3(&v1[i], simd_r);
            }
            return end != total;
        }
    };

    struct CRTMergeTask : BaseTask {
        std::span<uint32_t> v;

        static constexpr std::size_t THRESHOLD           = 1024;
        static constexpr std::size_t BLOCK_THREAD_FACTOR = 4;

        void init(std::size_t total_threads) {
            if (total == 0) {
                total      = v.size() / ntt::NUM_PRIMES;
                block_size = std::max(1ull, total / (total_threads * BLOCK_THREAD_FACTOR));
            }
        }

        auto run(std::size_t cur_block) -> bool {
            std::size_t start = cur_block * block_size, end = std::min(start + block_size, total);
            if (start >= end) {
                return false;
            }
            for (std::size_t i = start * ntt::NUM_PRIMES; i < end * ntt::NUM_PRIMES;
                i += ntt::NUM_PRIMES) {
                uint128_t x = ntt::garner_merge(v[i + 0], v[i + 1], v[i + 2]);
                v[i + 0]    = x & DIGIT_MASK;
                x >>= DIGIT_BITS;
                v[i + 1] = x & DIGIT_MASK;
                x >>= DIGIT_BITS;
                v[i + 2] = x & DIGIT_MASK;
            }
            return end != total;
        }
    };

    using p_Task = std::variant<IMMNTTTask*, IMMMulNumTask*, IMMMulVecTask*, CRTMergeTask*>;

private:
    p_Task                    p_cur_task_;
    std::atomic<std::size_t>  cur_block_;
    std::mutex                mtx;
    std::vector<std::jthread> threads_;
    uint32_t                  total_threads_;
    std::atomic<uint32_t>     running_count_;
    std::atomic<bool>         have_work_, all_finish_, stop_;
    bool                      is_valid_pool_;

    static void wait_with_spin_then_block(std::atomic<bool>& flag) {
        using clock = std::chrono::steady_clock;
        constexpr std::chrono::milliseconds SPIN_TIMEOUT{5};
        constexpr std::size_t               CHECK_INTERVAL = 32;

        const auto spin_deadline = clock::now() + SPIN_TIMEOUT;
        do {
            std::this_thread::yield();
            for (std::size_t i = 0; i < CHECK_INTERVAL; ++i) {
                if (flag.load(std::memory_order_acquire)) {
                    return;
                }
                _mm_pause();
            }
        } while (clock::now() < spin_deadline);

        while (!flag.load(std::memory_order_acquire)) {
            flag.wait(false, std::memory_order_acquire);
        }
    }

public:
    static auto is_valid_thread_count(uint32_t n) -> bool { return n != 0 && n != 1; }

    NTTThreadPool(uint32_t n)
        : total_threads_(n)
        , running_count_(n)
        , have_work_(false)
        , all_finish_(false)
        , stop_(false)
        , is_valid_pool_(is_valid_thread_count(n)) {

        if (!is_valid_pool_) {
            return;
        }

        auto worker = [&]() {
            while (true) {
                wait_with_spin_then_block(have_work_);

                if (stop_.load(std::memory_order_acquire)) {
                    break;
                }

                while (std::visit(
                    [&](auto p_task) -> bool {
                        return p_task->run(cur_block_.fetch_add(1, std::memory_order_relaxed));
                    },
                    p_cur_task_)) {}

                running_count_.fetch_sub(1, std::memory_order_release);
                wait_with_spin_then_block(all_finish_);

                running_count_.fetch_add(1, std::memory_order_release);
            }
        };

        for (uint32_t i = 0; i < total_threads_; ++i) {
            threads_.emplace_back(worker);
        }
    }

    ~NTTThreadPool() {
        if (is_valid_pool_) {
            stop_.store(true, std::memory_order_release);
            have_work_.store(true, std::memory_order_release);
            have_work_.notify_all();
        }
    }

    NTTThreadPool(const NTTThreadPool&)  = delete;
    NTTThreadPool(NTTThreadPool&&)       = delete;
    auto operator=(const NTTThreadPool&) = delete;
    auto operator=(NTTThreadPool&&)      = delete;

    void run_task(p_Task p_task) {
        std::lock_guard<std::mutex> lock(mtx);

        p_cur_task_ = p_task;
        std::visit([&](auto ptr_task) { ptr_task->init(total_threads_); }, p_cur_task_);

        cur_block_.store(0, std::memory_order_relaxed);
        have_work_.store(true, std::memory_order_release);
        have_work_.notify_all();

        while (running_count_.load(std::memory_order_acquire) != 0) {
            _mm_pause();
        }

        have_work_.store(false, std::memory_order_relaxed);
        all_finish_.store(true, std::memory_order_release);
        all_finish_.notify_all();

        while (running_count_.load(std::memory_order_acquire) != total_threads_) {
            _mm_pause();
        }

        all_finish_.store(false, std::memory_order_relaxed);
    }

    [[nodiscard]] auto is_valid() const -> bool { return is_valid_pool_; }
};

static auto get_pool(uint32_t n = 0) -> NTTThreadPool& {
    static NTTThreadPool pool(n != 0 ? n : std::thread::hardware_concurrency());
    return pool;
}

static void imm_mul_num(std::span<uint32_t> v, std::array<uint32_t, ntt::NUM_PRIMES> nums) {
    const std::size_t n = v.size();
    if (n == 0 || n % 3 != 0) {
        unreachable();
    }
    auto& pool = get_pool();
    if (n < ntt::NUM_PRIMES * NTTThreadPool::IMMMulNumTask::THRESHOLD || !pool.is_valid()) {
        __m128i simd_nums = ntt::mont::vec_to_simd(nums);
        for (std::size_t i = 0; i < n; i += ntt::NUM_PRIMES) {
            __m128i simd_a = ntt::mont::vec_to_simd({v[i + 0], v[i + 1], v[i + 2]});
            __m128i simd_r = ntt::mont::simd_mul(simd_a, simd_nums);
            ntt::mont::simd_store3(&v[i], simd_r);
        }
    } else {
        NTTThreadPool::IMMMulNumTask task({}, v, nums);
        pool.run_task(&task);
    }
}

static void imm_mul_vec(std::span<uint32_t> v1, std::span<const uint32_t> v2) {
    const std::size_t n = v1.size();
    if (n == 0 || n % 3 != 0 || n != v2.size()) {
        unreachable();
    }
    auto& pool = get_pool();
    if (n < ntt::NUM_PRIMES * NTTThreadPool::IMMMulNumTask::THRESHOLD || !pool.is_valid()) {
        for (std::size_t i = 0; i < n; i += ntt::NUM_PRIMES) {
            __m128i simd_a = ntt::mont::vec_to_simd({v1[i + 0], v1[i + 1], v1[i + 2]});
            __m128i simd_b = ntt::mont::vec_to_simd({v2[i + 0], v2[i + 1], v2[i + 2]});
            __m128i simd_r = ntt::mont::simd_mul(simd_a, simd_b);
            ntt::mont::simd_store3(&v1[i], simd_r);
        }
    } else {
        NTTThreadPool::IMMMulVecTask task({}, v1, v2);
        pool.run_task(&task);
    }
}

// Montgomery 域的交织式多模数 NTT (interleaved multi-modulus NTT)，
// 对下标模 NUM_PRIMES 同余类分别应用模数为 P[i] 的 NTT，
// 务必保证传入的数组长度是 2 的幂 * NUM_PRIMES 。
static void imm_ntt(std::span<uint32_t> v, bool rev) {
    std::size_t n = v.size();
    if (n == ntt::NUM_PRIMES) {
        return;
    }
    if (n % ntt::NUM_PRIMES != 0 || ntt::is_invalid_ntt_len(n / ntt::NUM_PRIMES)) {
        unreachable();
    }
    n /= ntt::NUM_PRIMES;
    ntt::multi_bit_swap(v);
    auto& pool = get_pool();
    if (n < 2 * NTTThreadPool::IMMNTTTask::THRESHOLD || !pool.is_valid()) {
        for (std::size_t len = 2; len <= n; len *= 2) {
            std::size_t   half = len / 2;
            const __m128i simd_step =
                ntt::mont::vec_to_simd(ntt::steps[rev][std::countr_zero(half)]);
            for (std::size_t i = 0; i < n; i += len) {
                __m128i simd_cur = ntt::mont::simd_one;
                for (std::size_t j = 0; j < half; ++j) {
                    std::size_t idx_a = (i + j) * ntt::NUM_PRIMES;
                    std::size_t idx_b = (i + j + half) * ntt::NUM_PRIMES;
                    __m128i     simd_a =
                        ntt::mont::vec_to_simd({v[idx_a + 0], v[idx_a + 1], v[idx_a + 2]});
                    __m128i simd_b = ntt::mont::simd_mul(simd_cur,
                        ntt::mont::vec_to_simd({v[idx_b + 0], v[idx_b + 1], v[idx_b + 2]}));
                    // __m256i simd_add_sub;
                    // ntt::mont::simd_add_and_sub(simd_a, simd_b, simd_add_sub);
                    // __m128i simd_a_add_b = _mm256_castsi256_si128(simd_add_sub);
                    // __m128i simd_a_sub_b = _mm256_extracti128_si256(simd_add_sub, 1);
                    __m128i simd_a_add_b = ntt::mont::simd_add(simd_a, simd_b);
                    __m128i simd_a_sub_b = ntt::mont::simd_sub(simd_a, simd_b);
                    ntt::mont::simd_store3(&v[idx_a], simd_a_add_b);
                    ntt::mont::simd_store3(&v[idx_b], simd_a_sub_b);
                    simd_cur = ntt::mont::simd_mul(simd_cur, simd_step);
                }
            }
        }
    } else {
        NTTThreadPool::IMMNTTTask task({}, v, 2, rev);
        for (task.len = 2; task.len <= n; task.len *= 2) {
            pool.run_task(&task);
        }
    }
    if (rev) {
        imm_mul_num(v, ntt::inv_pow_of_two[std::countr_zero(n) - 1]);
    }
}

// 利用中国剩余定理从 imm_ntt 逆变换结果中复原出真实数据，
// 拆成 3 个 DIGIT_BITS 位二进制数写回原位，
// 调用者应当保证这样的拆分是安全的。
void crt_merge(std::span<uint32_t> v) {
    const std::size_t n = v.size();
    if (n == 0 || n % 3 != 0) {
        unreachable();
    }
    auto& pool = get_pool();
    if (n < ntt::NUM_PRIMES * NTTThreadPool::CRTMergeTask::THRESHOLD || !pool.is_valid()) {
        for (std::size_t i = 0; i < n; i += ntt::NUM_PRIMES) {
            uint128_t x = ntt::garner_merge(v[i + 0], v[i + 1], v[i + 2]);
            v[i + 0]    = x & DIGIT_MASK;
            x >>= DIGIT_BITS;
            v[i + 1] = x & DIGIT_MASK;
            x >>= DIGIT_BITS;
            v[i + 2] = x & DIGIT_MASK;
        }
    } else {
        NTTThreadPool::CRTMergeTask task{{}, v};
        pool.run_task(&task);
    }
}

}  // namespace mul::ntt_multithread

void init_thread_pool(uint32_t n) {
    mul::ntt_multithread::get_pool(n == 0 ? 1 : n);
}

namespace mul::ntt {

// 从 crt_merge 结果中复原位权并进位
// 其位权为: 0, 1, 2 | 1, 2, 3 | 2, 3, 4 | ...
// 设 i 为位权，j 为下标，有 i = j / 3 + j % 3
// 反推出 j = 3 * i, 3 * (i - 1) + 1, 3 * (i - 2) + 2
// 即 j = 3 * i, 3 * i - 2, 3 * i - 4
static void trim(std::vector<uint32_t>& v) {
    std::size_t new_size_ntt = v.size(), new_size = new_size_ntt / ntt::NUM_PRIMES;
    if (new_size != 1) {
        uint32_t digit = v[1] + v[3 * 1];
        v[1]           = digit & NTT_DIGIT_MASK;
        uint32_t carry = digit >> NTT_DIGIT_BITS;
        for (std::size_t i = 2, j = i * ntt::NUM_PRIMES; i < new_size; ++i, j += ntt::NUM_PRIMES) {
            digit = v[j - 4] + v[j - 2] + v[j] + carry;
            v[i]  = digit & NTT_DIGIT_MASK;
            carry = digit >> NTT_DIGIT_BITS;
        }
        digit           = v[new_size_ntt - 4] + v[new_size_ntt - 2] + carry;
        v[new_size]     = digit & NTT_DIGIT_MASK;
        carry           = digit >> NTT_DIGIT_BITS;
        digit           = v[new_size_ntt + 3 - 4] + carry;
        v[new_size + 1] = digit & NTT_DIGIT_MASK;
        carry           = digit >> NTT_DIGIT_BITS;
        v[new_size + 2] = carry;
    }
    std::size_t i = new_size == 1 ? 2 : new_size + 2;
    while (v[i] == 0) {
        --i;
    }
    v.resize(i + 1);
}

auto mul(std::span<const uint64_t> A, std::span<const uint64_t> B) -> std::vector<uint64_t> {
    bool a_is_b = A.data() == B.data();

    auto vec64_to_vec32_and_repeat = [](std::span<const uint64_t> src,
                                         std::vector<uint32_t>&   dst,
                                         int                      digit_bits,
                                         int                      k) -> void {
        uint32_t    mask     = (1u << digit_bits) - 1;
        uint128_t   tmp      = 0;
        int         tmp_bits = 0;
        std::size_t i = 0, n = src.size();
        while (i < n) {
            if (tmp_bits < digit_bits && i < n) {
                tmp |= static_cast<uint128_t>(src[i]) << tmp_bits;
                tmp_bits += 64;
                ++i;
            }
            while (tmp_bits >= digit_bits) {
                for (int j = 0; j < k; ++j) {
                    dst.emplace_back(tmp & mask);
                }
                tmp >>= digit_bits;
                tmp_bits -= digit_bits;
            }
        }
        while (tmp) {
            for (int j = 0; j < k; ++j) {
                dst.emplace_back(tmp & mask);
            }
            tmp >>= digit_bits;
        }
    };

    std::size_t           new_size =
                              std::bit_ceil(ceil_div<std::size_t>(A.size() * 64, NTT_DIGIT_BITS)
                                            + ceil_div<std::size_t>(B.size() * 64, NTT_DIGIT_BITS) - 1),
                          new_size_ntt = new_size * 3;
    std::vector<uint32_t> vec_a;
    vec_a.reserve(new_size_ntt);
    vec64_to_vec32_and_repeat(A, vec_a, NTT_DIGIT_BITS, NUM_PRIMES);
    vec_a.resize(new_size_ntt);
    ntt_multithread::imm_mul_num(vec_a, mont::R_sq);
    ntt_multithread::imm_ntt(vec_a, false);

    if (a_is_b) {
        ntt_multithread::imm_mul_vec(vec_a, vec_a);
    } else {
        std::vector<uint32_t> vec_b;
        vec_b.reserve(new_size_ntt);
        vec64_to_vec32_and_repeat(B, vec_b, NTT_DIGIT_BITS, NUM_PRIMES);
        vec_b.resize(new_size_ntt);
        ntt_multithread::imm_mul_num(vec_b, mont::R_sq);
        ntt_multithread::imm_ntt(vec_b, false);
        ntt_multithread::imm_mul_vec(vec_a, vec_b);
    }

    ntt_multithread::imm_ntt(vec_a, true);
    ntt_multithread::imm_mul_num(vec_a, {1, 1, 1});
    ntt_multithread::crt_merge(vec_a);
    trim(vec_a);

    std::vector<uint64_t> res;
    res.reserve(ceil_div<std::size_t>(vec_a.size() * NTT_DIGIT_BITS, 64));
    {
        uint128_t   tmp      = 0;
        int         tmp_bits = 0;
        std::size_t i = 0, n = vec_a.size();
        while (i < n) {
            while (tmp_bits < 64 && i < n) {
                tmp |= static_cast<uint128_t>(vec_a[i]) << tmp_bits;
                tmp_bits += NTT_DIGIT_BITS;
                ++i;
            }
            if (tmp_bits >= 64) {
                res.emplace_back(tmp);
                tmp_bits -= 64;
                tmp >>= 64;
            }
        }
        if (tmp) {
            res.emplace_back(tmp);
        } else {
            utils::remove_leading_zero(res);
        }
    }

    return res;
}

}  // namespace mul::ntt

}  // namespace bigint
