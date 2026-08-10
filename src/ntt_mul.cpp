#include "bigint/ntt_mul.h"
#include "bigint/bigint_base.h"
#include "bigint/ntt_multithread.h"


namespace bigint::ntt {

static constexpr auto get_mul_offset(const Digits& v, size_type output_precision) -> size_type {
    return output_precision == 0 || v.size() <= output_precision ? 0 : v.size() - output_precision;
}

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
        if (offset_res) {
            vec_a.erase(vec_a.begin(), vec_a.begin() + static_cast<int64_t>(offset_res));
        }
        if (p_result_point_pos) {
            *p_result_point_pos -= static_cast<int64_t>(offset_a + offset_b + offset_res);
        }
    } else {
        vec_a.insert(vec_a.begin(), tail_zero_a + tail_zero_b, 0);
    }

    return vec_a;
}

}  // namespace bigint::ntt
