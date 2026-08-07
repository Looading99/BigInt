#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <immintrin.h>
#include <memory>
#include <mutex>
#include <thread>
#include <variant>
#include <vector>


#include "bigint/ntt.h"
#include "bigint/ntt_multithread.h"


namespace bigint {

namespace ntt_multithread {

class NTTThreadPool {
public:
    struct BaseTask {
    protected:
        size_type total, block_size;
    };

    // 交织式多模数 NTT
    struct IMMNTTTask : BaseTask {
        Digits*   p_v;
        size_type len;
        bool      rev;

        static constexpr size_type THRESHOLD           = 512;
        static constexpr size_type BLOCK_THREAD_FACTOR = 4;

        void init(size_type total_threads) {
            if (total == 0) {
                total      = p_v->size() / 2 / ntt::NUM_PRIMES;
                block_size = std::max(1ull, total / (total_threads * BLOCK_THREAD_FACTOR));
            }
        }

        auto run(size_type cur_block) -> bool {
            size_type start = cur_block * block_size, end = std::min(start + block_size, total);
            if (start >= end) {
                return false;
            }
            auto&           v    = *p_v;
            const size_type half = len / 2;
            const __m128i   simd_step =
                ntt::mont::vec_to_simd(ntt::steps[rev][std::countr_zero(half)]);
            __m128i simd_cur = ntt::mont::simd_one;
            if (size_type j0 = start & (half - 1); j0 != 0) {
                simd_cur = ntt::mont::fast_pow(simd_step, j0);
            }
            for (size_type cnt = start; cnt < end; ++cnt) {
                // 等价于 j = cnt % half, i = (cnt / half) * len
                size_type j = cnt & (half - 1), i = (cnt - j) << 1;
                if (j == 0) {
                    simd_cur = ntt::mont::simd_one;
                }
                size_type idx_a = (i + j) * ntt::NUM_PRIMES;
                size_type idx_b = (i + j + half) * ntt::NUM_PRIMES;
                __m128i simd_a = ntt::mont::vec_to_simd({v[idx_a + 0], v[idx_a + 1], v[idx_a + 2]});
                __m128i simd_b = ntt::mont::simd_mul(
                    simd_cur, ntt::mont::vec_to_simd({v[idx_b + 0], v[idx_b + 1], v[idx_b + 2]}));
                // __m256i simd_add_sub = ntt::mont::simd_add_and_sub(simd_a, simd_b);
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
        Digits*                               p_v;
        std::array<uint32_t, ntt::NUM_PRIMES> nums;

        static constexpr size_type THRESHOLD           = 1024;
        static constexpr size_type BLOCK_THREAD_FACTOR = 4;

        void init(size_type total_threads) {
            if (total == 0) {
                total      = p_v->size() / ntt::NUM_PRIMES;
                block_size = std::max(1ull, total / (total_threads * BLOCK_THREAD_FACTOR));
            }
        }

        auto run(size_type cur_block) -> bool {
            auto&     v     = *p_v;
            size_type start = cur_block * block_size, end = std::min(start + block_size, total);
            if (start >= end) {
                return false;
            }
            __m128i simd_nums = ntt::mont::vec_to_simd(nums);
            for (size_type i = start * ntt::NUM_PRIMES; i < end * ntt::NUM_PRIMES;
                i += ntt::NUM_PRIMES) {
                __m128i simd_a = ntt::mont::vec_to_simd({v[i + 0], v[i + 1], v[i + 2]});
                __m128i simd_r = ntt::mont::simd_mul(simd_a, simd_nums);
                ntt::mont::simd_store3(&v[i], simd_r);
            }
            return end != total;
        }
    };

    struct IMMMulVecTask : BaseTask {
        Digits*       p_v;
        const Digits* p_v2;

        static constexpr size_type THRESHOLD           = 1024;
        static constexpr size_type BLOCK_THREAD_FACTOR = 4;

        void init(size_type total_threads) {
            if (total == 0) {
                total      = p_v->size() / ntt::NUM_PRIMES;
                block_size = std::max(1ull, total / (total_threads * BLOCK_THREAD_FACTOR));
            }
        }

        auto run(size_type cur_block) -> bool {
            auto&       v     = *p_v;
            const auto& v2    = *p_v2;
            size_type   start = cur_block * block_size, end = std::min(start + block_size, total);
            if (start >= end) {
                return false;
            }
            for (size_type i = start * ntt::NUM_PRIMES; i < end * ntt::NUM_PRIMES;
                i += ntt::NUM_PRIMES) {
                __m128i simd_a = ntt::mont::vec_to_simd({v[i + 0], v[i + 1], v[i + 2]});
                __m128i simd_b = ntt::mont::vec_to_simd({v2[i + 0], v2[i + 1], v2[i + 2]});
                __m128i simd_r = ntt::mont::simd_mul(simd_a, simd_b);
                ntt::mont::simd_store3(&v[i], simd_r);
            }
            return end != total;
        }
    };

    struct CRTMergeTask : BaseTask {
        Digits* p_v;

        static constexpr size_type THRESHOLD           = 1024;
        static constexpr size_type BLOCK_THREAD_FACTOR = 4;

        void init(size_type total_threads) {
            if (total == 0) {
                total      = p_v->size() / ntt::NUM_PRIMES;
                block_size = std::max(1ull, total / (total_threads * BLOCK_THREAD_FACTOR));
            }
        }

        auto run(size_type cur_block) -> bool {
            auto&     v     = *p_v;
            size_type start = cur_block * block_size, end = std::min(start + block_size, total);
            if (start >= end) {
                return false;
            }
            for (size_type i = start * ntt::NUM_PRIMES; i < end * ntt::NUM_PRIMES;
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
    std::atomic<size_type>    cur_block_;
    std::mutex                mtx;
    std::vector<std::jthread> threads_;
    uint32_t                  total_threads_;
    std::atomic<uint32_t>     ready_count_, running_count_;
    std::atomic<bool>         have_work_, all_finish_, stop_;
    bool                      is_valid_pool_;

    using clock = std::chrono::steady_clock;
    static constexpr std::chrono::milliseconds SPIN_TIMEOUT{100};

    static void wait_with_spin_then_block(std::atomic<bool>& flag) {
        const auto spin_deadline = clock::now() + SPIN_TIMEOUT;
        while (clock::now() < spin_deadline) {
            if (flag.load(std::memory_order_acquire)) {
                return;
            }
            std::this_thread::yield();
        }
        while (!flag.load(std::memory_order_acquire)) {
            flag.wait(false, std::memory_order_acquire);
        }
    }

public:
    static auto is_valid_thread_count(uint32_t n) -> bool { return n != 0 && n != 1; }

    NTTThreadPool(uint32_t n)
        : total_threads_(n)
        , ready_count_(0)
        , running_count_(0)
        , have_work_(false)
        , all_finish_(false)
        , stop_(false) {

        if (!is_valid_thread_count(n)) {
            is_valid_pool_ = false;
            return;
        }
        is_valid_pool_ = true;

        auto worker = [&]() {
            while (true) {
                ready_count_.fetch_add(1, std::memory_order_release);
                wait_with_spin_then_block(have_work_);

                ready_count_.fetch_sub(1, std::memory_order_relaxed);

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

        running_count_.store(total_threads_, std::memory_order_relaxed);
        cur_block_.store(0, std::memory_order_relaxed);
        have_work_.store(true, std::memory_order_release);
        have_work_.notify_all();

        while (running_count_.load(std::memory_order_acquire) != 0) {
            std::this_thread::yield();
        }

        have_work_.store(false, std::memory_order_relaxed);
        all_finish_.store(true, std::memory_order_release);
        all_finish_.notify_all();

        while (ready_count_.load(std::memory_order_acquire) != total_threads_) {
            std::this_thread::yield();
        }

        all_finish_.store(false, std::memory_order_release);
    }

    [[nodiscard]] auto is_valid() const -> bool { return is_valid_pool_; }
};

static auto get_pool(uint32_t n = 0) -> NTTThreadPool& {
    static std::unique_ptr<NTTThreadPool> p_pool;
    static std::once_flag                 once;
    std::call_once(once, [n] {
        p_pool = std::make_unique<NTTThreadPool>(n != 0 ? n : std::thread::hardware_concurrency());
    });
    return *p_pool;
}

void imm_ntt(Digits& v, bool rev) {
    size_type n = v.size();
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
        for (size_type len = 2; len <= n; len *= 2) {
            size_type     half = len / 2;
            const __m128i simd_step =
                ntt::mont::vec_to_simd(ntt::steps[rev][std::countr_zero(half)]);
            for (size_type i = 0; i < n; i += len) {
                __m128i simd_cur = ntt::mont::simd_one;
                for (size_type j = 0; j < half; ++j) {
                    size_type idx_a = (i + j) * ntt::NUM_PRIMES;
                    size_type idx_b = (i + j + half) * ntt::NUM_PRIMES;
                    __m128i   simd_a =
                        ntt::mont::vec_to_simd({v[idx_a + 0], v[idx_a + 1], v[idx_a + 2]});
                    __m128i simd_b = ntt::mont::simd_mul(simd_cur,
                        ntt::mont::vec_to_simd({v[idx_b + 0], v[idx_b + 1], v[idx_b + 2]}));
                    // __m256i simd_add_sub = ntt::mont::simd_add_and_sub(simd_a, simd_b);
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
        NTTThreadPool::IMMNTTTask task({}, &v, 2, rev);
        for (task.len = 2; task.len <= n; task.len *= 2) {
            pool.run_task(&task);
        }
    }
    if (rev) {
        imm_mul_num(v, ntt::inv_pow_of_two[std::countr_zero(n) - 1]);
    }
}

void imm_mul_num(Digits& v, std::array<uint32_t, ntt::NUM_PRIMES> nums) {
    const size_type n = v.size();
    if (n == 0 || n % 3 != 0) {
        unreachable();
    }
    auto& pool = get_pool();
    if (n < ntt::NUM_PRIMES * NTTThreadPool::IMMMulNumTask::THRESHOLD || !pool.is_valid()) {
        __m128i simd_nums = ntt::mont::vec_to_simd(nums);
        for (size_type i = 0; i < n; i += ntt::NUM_PRIMES) {
            __m128i simd_a = ntt::mont::vec_to_simd({v[i + 0], v[i + 1], v[i + 2]});
            __m128i simd_r = ntt::mont::simd_mul(simd_a, simd_nums);
            ntt::mont::simd_store3(&v[i], simd_r);
        }
    } else {
        NTTThreadPool::IMMMulNumTask task({}, &v, nums);
        pool.run_task(&task);
    }
}

void imm_mul_vec(Digits& v, const Digits& v2) {
    const size_type n = v.size();
    if (n == 0 || n % 3 != 0 || n != v2.size()) {
        unreachable();
    }
    auto& pool = get_pool();
    if (n < ntt::NUM_PRIMES * NTTThreadPool::IMMMulNumTask::THRESHOLD || !pool.is_valid()) {
        for (size_type i = 0; i < n; i += ntt::NUM_PRIMES) {
            __m128i simd_a = ntt::mont::vec_to_simd({v[i + 0], v[i + 1], v[i + 2]});
            __m128i simd_b = ntt::mont::vec_to_simd({v2[i + 0], v2[i + 1], v2[i + 2]});
            __m128i simd_r = ntt::mont::simd_mul(simd_a, simd_b);
            ntt::mont::simd_store3(&v[i], simd_r);
        }
    } else {
        NTTThreadPool::IMMMulVecTask task({}, &v, &v2);
        pool.run_task(&task);
    }
}

void crt_merge(Digits& v) {
    const size_type n = v.size();
    if (n == 0 || n % 3 != 0) {
        unreachable();
    }
    auto& pool = get_pool();
    if (n < ntt::NUM_PRIMES * NTTThreadPool::CRTMergeTask::THRESHOLD || !pool.is_valid()) {
        for (size_type i = 0; i < n; i += ntt::NUM_PRIMES) {
            uint128_t x = ntt::garner_merge(v[i + 0], v[i + 1], v[i + 2]);
            v[i + 0]    = x & DIGIT_MASK;
            x >>= DIGIT_BITS;
            v[i + 1] = x & DIGIT_MASK;
            x >>= DIGIT_BITS;
            v[i + 2] = x & DIGIT_MASK;
        }
    } else {
        NTTThreadPool::CRTMergeTask task{{}, &v};
        pool.run_task(&task);
    }
}

}  // namespace ntt_multithread


void init_thread_pool(uint32_t n) {
    ntt_multithread::get_pool(n == 0 ? 1 : n);
}


namespace ntt {

// 从 crt_merge 结果中复原位权并进位
// 其位权为: 0, 1, 2 | 1, 2, 3 | 2, 3, 4 | ...
// 设 i 为位权，j 为下标，有 i = j / 3 + j % 3
// 反推出 j = 3 * i, 3 * (i - 1) + 1, 3 * (i - 2) + 2
// 即 j = 3 * i, 3 * i - 2, 3 * i - 4
static void trim(Digits& v) {
    size_type new_size_ntt = v.size(), new_size = new_size_ntt / ntt::NUM_PRIMES;
    if (new_size != 1) {
        uint32_t digit = v[1] + v[3 * 1];
        v[1]           = digit & DIGIT_MASK;
        uint32_t carry = digit >> DIGIT_BITS;
        for (size_type i = 2, j = i * ntt::NUM_PRIMES; i < new_size; ++i, j += ntt::NUM_PRIMES) {
            digit = v[j - 4] + v[j - 2] + v[j] + carry;
            v[i]  = digit & DIGIT_MASK;
            carry = digit >> DIGIT_BITS;
        }
        digit           = v[new_size_ntt - 4] + v[new_size_ntt - 2] + carry;
        v[new_size]     = digit & DIGIT_MASK;
        carry           = digit >> DIGIT_BITS;
        digit           = v[new_size_ntt + 3 - 4] + carry;
        v[new_size + 1] = digit & DIGIT_MASK;
        carry           = digit >> DIGIT_BITS;
        v[new_size + 2] = carry;
    }
    size_type i = new_size == 1 ? 2 : new_size + 2;
    while (v[i] == 0) {
        --i;
    }
    v.resize(i + 1);
}

auto ntt_mul(const Digits& a, const Digits& b, size_type output_precision,
    int64_t* p_result_point_pos) -> Digits {

    if (a.size() == 0 || b.size() == 0) {
        unreachable();
    }

    bool a_is_b = std::addressof(a) == std::addressof(b);

    size_type offset_a    = get_mul_offset(a, output_precision);
    size_type offset_b    = a_is_b ? offset_a : get_mul_offset(b, output_precision);
    size_type tail_zero_a = 0, tail_zero_b = 0;
    while (a[offset_a + tail_zero_a] == 0) {
        ++tail_zero_a;
    }
    if (a_is_b) {
        tail_zero_b = tail_zero_a;
    } else {
        while (b[offset_b + tail_zero_b] == 0) {
            ++tail_zero_b;
        }
    }
    offset_a += tail_zero_a;
    offset_b += tail_zero_b;
    size_type new_size     = std::bit_ceil(a.size() - offset_a + b.size() - offset_b - 1),
              new_size_ntt = new_size * ntt::NUM_PRIMES;

    Digits vec_a;
    vec_a.reserve(
        output_precision ? new_size_ntt : std::max(new_size_ntt, a.size() + b.size() + 1));
    copy_repeat(a, offset_a, ntt::NUM_PRIMES, vec_a);
    vec_a.resize(new_size_ntt);
    ntt_multithread::imm_mul_num(vec_a, ntt::mont::R_sq);
    ntt_multithread::imm_ntt(vec_a, false);

    if (a_is_b) {
        ntt_multithread::imm_mul_vec(vec_a, vec_a);
    } else {
        Digits vec_b;
        vec_b.reserve(new_size_ntt);
        copy_repeat(b, offset_b, ntt::NUM_PRIMES, vec_b);
        vec_b.resize(new_size_ntt);
        ntt_multithread::imm_mul_num(vec_b, ntt::mont::R_sq);
        ntt_multithread::imm_ntt(vec_b, false);

        ntt_multithread::imm_mul_vec(vec_a, vec_b);
    }

    ntt_multithread::imm_ntt(vec_a, true);
    ntt_multithread::imm_mul_num(vec_a, {1, 1, 1});

    ntt_multithread::crt_merge(vec_a);

    trim(vec_a);

    if (output_precision) {
        auto offset_res = get_mul_offset(vec_a, output_precision);
        while (vec_a[offset_res] == 0) {
            ++offset_res;
        }
        if (offset_res)
            vec_a.erase(vec_a.begin(), vec_a.begin() + static_cast<int64_t>(offset_res));
        if (p_result_point_pos)
            *p_result_point_pos -= static_cast<int64_t>(offset_a + offset_b + offset_res);
    } else {
        vec_a.insert(vec_a.begin(), tail_zero_a + tail_zero_b, 0);
    }

    return vec_a;
}


}  // namespace ntt


}  // namespace bigint
