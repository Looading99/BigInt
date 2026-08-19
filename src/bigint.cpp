#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>


#include "bigint/bigint.h"
#include "bigint/mul.h"
#include "bigint/ntt_mul.h"


namespace bigint {

std::array<size_type, 2> BigInt::DEC_STRING_BRUTE_THRESHOLDS = DEC_STRING_BRUTE_THRESHOLDS_DEFAULT;

BigInt::BigInt(const std::string& s, bool hex)
    : BigInt() {

    if (hex) {
        constexpr uint8_t HEX_BITS = 4;
        constexpr uint8_t HEX_MAX  = (1 << HEX_BITS) - 1;
        constexpr auto    hex_map  = [] {
            std::array<uint8_t, UINT8_MAX + 1> arr{};
            arr.fill(HEX_MAX + 1);
            for (uint8_t i = 0; i < TEN; ++i) {
                arr['0' + i] = i;
            }
            for (uint8_t i = TEN; i <= HEX_MAX; ++i) {
                arr['a' + i - TEN] = arr['A' + i - TEN] = i;
            }
            return arr;
        }();
        size_type rend = 0, n = s.size();
        for (; rend < n; ++rend) {
            unsigned char c = s[rend];
            if (c == '-') {
                is_neg_ = true;
            } else if (hex_map[c] <= HEX_MAX) {
                break;
            }
        }
        uint32_t  digit = 0, offset = 0;
        size_type i = n;
        while (i > rend) {
            --i;
            uint32_t num = hex_map[static_cast<unsigned char>(s[i])];
            if (num > HEX_MAX) {
                continue;
            }
            digit |= num << offset;
            offset += HEX_BITS;
            if (offset == DIGIT_BITS) {
                offset = 0;
                data_.push_back(digit);
                digit = 0;
            }
        }
        if (digit) {
            data_.push_back(digit);
        }
    } else {  // dec
        size_type i = 0, n = s.size();
        for (; i < n; ++i) {
            unsigned char c = s[i];
            if (std::isdigit(c)) {
                break;
            } else if (c == '-') {
                is_neg_ = true;
            }
        }
        auto res = convert_from_dec_string(s, i, n);
        data_    = std::move(res.first.data_);
    }

    if (data_.size() == 0) {
        data_.push_back(0);
        is_neg_ = false;
    } else {
        remove_leading_zero();
    }
}

auto BigInt::convert_from_dec_string(const std::string& s, size_type start, size_type end)
    -> std::pair<BigInt, size_type> {
    if (end - start < std::max(2ull, DEC_STRING_BRUTE_THRESHOLDS[0])) {
        BigInt    res(0);
        size_type len = 0;
        uint64_t  cur = 0, base = 1;
        for (size_type i = start; i < end; ++i) {
            unsigned char c = s[i];
            if (!std::isdigit(c)) {
                continue;
            }
            ++len;
            base *= TEN;
            cur = cur * TEN + c - '0';
            if (base > UINT64_MAX / TEN) {
                res *= base;
                res += cur;
                base = 1;
                cur  = 0;
            }
        }
        if (base > 1) {
            res *= base;
            res += cur;
        }
        return {res, len};
    } else {
        auto half = (start + end) / 2;
        auto hi   = convert_from_dec_string(s, start, half);
        auto lo   = convert_from_dec_string(s, half, end);
        return {hi.first * get_pow_of_ten(lo.second) + lo.first, hi.second + lo.second};
    }
}

auto BigInt::to_dec_string_brute() const -> std::string {
    if (is_zero())
        return "0";

    constexpr size_type pow_max = floor_log(TEN, UINT64_MAX);
    constexpr uint64_t  divisor = fast_pow(static_cast<uint64_t>(TEN), pow_max);

    BigInt      temp = *this;
    std::string res;
    res.reserve(pow_max
                + static_cast<size_type>(
                    std::ceil(static_cast<double>(data_.size()) * DIGIT_BITS * std::log10(2))));

    while (temp) {
        uint64_t rem = temp.unsigned_inplace_divmod(divisor);
        for (size_type i = 0; i < pow_max && (rem != 0 || temp); ++i) {
            res.push_back(static_cast<char>(rem % TEN + '0'));
            rem /= TEN;
        }
    }

    if (is_neg_)
        res.push_back('-');

    std::ranges::reverse(res);
    return res;
}

auto BigInt::get_pow_of_ten(uint32_t exponent) -> BigInt {
    static std::vector<BigInt>  pows_of_ten = {TEN};
    static std::mutex           mtx;
    std::lock_guard<std::mutex> lock(mtx);

    BigInt res(1);
    if (exponent == 0) {
        return res;
    }
    uint32_t b = std::bit_width(exponent);
    while (b > pows_of_ten.size()) {
        pows_of_ten.emplace_back(pows_of_ten.back() * pows_of_ten.back());
    }
    for (uint32_t i = 0; i < b; ++i) {
        if ((exponent >> i) & 1) {
            res *= pows_of_ten[i];
        }
    }
    return res;
}

void BigInt::print(std::ostream& output, bool hex, bool direct) const {
    std::optional<std::stringstream> buffer;
    std::ostream&                    out = direct ? output : buffer.emplace();
    if (hex)
        print_hex(out);
    else
        print_dec(out);
    if (!direct) {
        output << buffer->rdbuf();
    }
}

void BigInt::print_dec(std::ostream& output, size_type len) const {
    const size_type n = data_.size();
    if (n == 0) {
        unreachable();
    }

    auto brute = [&](const BigInt& _num, size_type _len) {
        auto res     = _num.to_dec_string_brute();
        auto it      = res.begin();
        auto res_len = res.size();
        if (_num.is_neg_) {
            --res_len;
            output.put(*it);
            ++it;
        }
        if (_len > res_len) {
            for (size_type i = 0; i < _len - res_len; ++i) {
                output.put('0');
            }
        }
        for (; it != res.end(); ++it) {
            output.put(*it);
        }
    };

    if (n == 1 || n < DEC_STRING_BRUTE_THRESHOLDS[1]) {
        brute(*this, len);
        return;
    }

    std::vector<std::pair<BigInt, size_type>> stk;
    stk.reserve(std::bit_width(n));

    auto divide_push = [&](const BigInt& _num, size_type _len) {
        size_type estimate = std::ceil(
                      static_cast<double>(_num.data_.size()) * DIGIT_BITS * std::log10(2)),
                  k = estimate / 2;

        auto P      = get_pow_of_ten(k);
        auto [Q, R] = _num.divmod(P, RoundMode::Truncate);
        R.remove_sign();
        stk.emplace_back(std::move(R), k);
        stk.emplace_back(std::move(Q), _len > k ? _len - k : 0);
    };

    divide_push(*this, len);

    while (!stk.empty()) {
        auto [cur_num, cur_len] = std::move(stk.back());
        stk.pop_back();
        const size_type cn = cur_num.data_.size();
        if (cn == 0) {
            unreachable();
        }
        if (cn == 1 || cn < DEC_STRING_BRUTE_THRESHOLDS[1]) {
            brute(cur_num, cur_len);
        } else {
            divide_push(cur_num, cur_len);
        }
    }
}

void BigInt::print_hex(std::ostream& output) const {
    if (is_zero()) {
        output.put('0');
        return;
    }
    if (is_neg_) {
        output.put('-');
    };
    size_type n = data_.size();
    output << std::hex << data_.back() << std::setfill('0');
    size_type i = n - 1;
    while (i) {
        --i;
        output << std::setw(DIGIT_BITS / 4) << data_[i];
    }
    output << std::dec << std::setfill(' ');
}

auto BigInt::remove_leading_zero() -> size_type {
    size_type i = data_.size();
    if (i == 0) {
        unreachable();
    }
    while (i > 0) {
        --i;
        if (data_[i] != 0)
            break;
    }
    if (i == 0 && data_[0] == 0) {
        reset();
        return 1;
    } else {
        ++i;
        data_.resize(i);
        return i;
    }
}

auto BigInt::compare_abs(const BigInt& a, const BigInt& b) noexcept -> std::strong_ordering {
    const size_type n = a.data_.size(), m = b.data_.size();
    if (n == 0 || m == 0) {
        unreachable();
    }
    if (auto cmp = n <=> m; cmp != 0) {
        return cmp;
    }
    size_type i = n;
    while (i > 0) {
        --i;
        if (auto cmp = a.data_[i] <=> b.data_[i]; cmp != 0) {
            return cmp;
        }
    }
    return std::strong_ordering::equal;
}

void BigInt::unsigned_self_inc_or_dec(bool is_inc) {
    size_type n = data_.size(), i = 0;
    if (n == 0) {
        unreachable();
    }
    if (is_inc) {
        for (; i < n; ++i) {
            if (data_[i] != DIGIT_MASK) {
                ++data_[i];
                break;
            }
        }
        if (i == n) {
            data_.push_back(1);
        }
        if (i) {
            std::fill(data_.begin(), data_.begin() + static_cast<int64_t>(i), 0);
        }
    } else {
        for (; i < n; ++i) {
            if (data_[i] != 0) {
                --data_[i];
                break;
            }
        }
        if (i == n) {
            unreachable();
        }
        if (i) {
            std::fill(data_.begin(), data_.begin() + static_cast<int64_t>(i), DIGIT_MASK);
        }
    }
}

auto BigInt::operator++() -> BigInt& {
    unsigned_self_inc_or_dec(!is_neg_);
    return *this;
}

auto BigInt::operator++(int) -> BigInt {
    auto temp = *this;
    ++*this;
    return temp;
}

auto BigInt::operator--() -> BigInt& {
    if (is_zero()) {
        data_[0] = 1;
        is_neg_  = true;
    } else {
        unsigned_self_inc_or_dec(is_neg_);
    }
    return *this;
}

auto BigInt::operator--(int) -> BigInt {
    auto temp = *this;
    --*this;
    return temp;
}

auto BigInt::trim_all(bool is_borrow) -> uint32_t {
    if (data_.size() == 0) {
        unreachable();
    }
    uint32_t carry = 0;
    for (auto& num : data_) {
        num   = is_borrow ? num - carry : num + carry;
        carry = (num >> DIGIT_BITS) != 0;
        num &= DIGIT_MASK;
    }
    return carry;
}

void BigInt::inplace_add_or_sub(const BigInt& b, bool is_sub) {
    const size_type n = b.data_.size();
    if (data_.size() == 0 || n == 0) {
        unreachable();
    }
    if (n > data_.size()) {
        data_.resize(n);
    }
    if (is_neg_ ^ b.is_neg_ ^ is_sub) {
        for (size_type i = 0; i < n; ++i) {
            data_[i] -= b.data_[i];
        }
        remove_leading_zero();
        if (is_zero())
            return;
        if (data_.back() >> DIGIT_BITS) {
            is_neg_ = !is_neg_;
            for (auto& num : data_)
                num = -num;
        }
        trim_all(true);
        remove_leading_zero();
    } else {
        for (size_type i = 0; i < n; ++i) {
            data_[i] += b.data_[i];
        }
        uint32_t carry = trim_all(false);
        if (carry)
            data_.push_back(carry);
        else
            remove_leading_zero();
    }
}

void BigInt::inplace_add_or_sub(uint64_t b, bool is_sub) {
    if (data_.size() == 0) {
        unreachable();
    }
    if (b == 0)
        return;
    data_.resize(
        std::max(data_.size(), static_cast<size_type>(std::bit_width(b) / DIGIT_BITS) + 1));
    size_type i = 0;
    if (is_neg_ ^ is_sub) {
        while (b) {
            data_[i] -= static_cast<uint32_t>(b) & DIGIT_MASK;
            b >>= DIGIT_BITS;
            ++i;
        }
        remove_leading_zero();
        if (is_zero())
            return;
        if (data_.back() >> DIGIT_BITS) {
            is_neg_ = !is_neg_;
            for (auto& num : data_) {
                num = -num;
            }
        }
        trim_all(true);
        remove_leading_zero();
    } else {
        while (b) {
            data_[i] += static_cast<uint32_t>(b) & DIGIT_MASK;
            b >>= DIGIT_BITS;
            ++i;
        }
        uint32_t carry = trim_all(false);
        if (carry)
            data_.push_back(carry);
        else
            remove_leading_zero();
    }
}

void BigInt::unsigned_inplace_mul(uint64_t b) {
    if (is_zero() || b == 1) {
        return;
    } else if (b == 0) {
        reset();
    } else if ((b & (b - 1)) == 0) {  // b 是 2 的幂
        *this <<= std::countr_zero(b);
    } else if (b <= UINT32_MAX) {
        unsigned_inplace_mul<uint64_t, uint32_t>(b);
    } else {
        unsigned_inplace_mul<uint128_t, uint64_t>(b);
    }
}

template<typename C, typename T> void BigInt::unsigned_inplace_mul(T b) {
    if (data_.size() == 0) {
        unreachable();
    }
    C digit = 0, carry = 0;
    for (auto& num : data_) {
        digit = static_cast<C>(b) * num + carry;
        num   = static_cast<uint32_t>(digit) & DIGIT_MASK;
        carry = digit >> DIGIT_BITS;
    }
    if (carry) {
        while (carry) {
            data_.push_back(static_cast<uint32_t>(carry) & DIGIT_MASK);
            carry >>= DIGIT_BITS;
        }
    } else {
        remove_leading_zero();
    }
}

auto BigInt::unsigned_inplace_divmod(uint64_t b) -> uint64_t {
    if (data_.size() == 0) {
        unreachable();
    }
    if (b == 0) {
        throw std::domain_error("division by zero");
    }
    uint128_t rem = 0;
    size_type i   = data_.size();
    while (i) {
        --i;
        rem      = (rem << DIGIT_BITS) + data_[i];
        data_[i] = static_cast<uint32_t>(rem / b);
        rem %= b;
    }
    remove_leading_zero();
    return rem;
}

auto BigInt::brute_mul(const BigInt& a, const BigInt& b) -> BigInt {
    const size_type n = a.data_.size(), m = b.data_.size();
    if (n == 0 || m == 0) {
        unreachable();
    }
    BigInt res;
    res.data_.resize(n + m + 2);
    uint64_t carry = 0;
    for (size_type j = 0; j < m; ++j) {
        size_type i = 0;
        for (; i < n; ++i) {
            uint64_t digit =
                static_cast<uint64_t>(a.data_[i]) * b.data_[j] + carry + res.data_[i + j];
            res.data_[i + j] = static_cast<uint32_t>(digit) & DIGIT_MASK;
            carry            = digit >> DIGIT_BITS;
        }
        while (carry) {
            uint64_t digit   = res.data_[i + j] + carry;
            res.data_[i + j] = static_cast<uint32_t>(digit) & DIGIT_MASK;
            carry            = digit >> DIGIT_BITS;
            ++i;
        }
    }
    res.is_neg_ = a.is_neg_ ^ b.is_neg_;
    res.remove_leading_zero();
    return res;
}

auto BigInt::ntt_mul(const BigInt& a, const BigInt& b) -> BigInt {
    BigInt res;
    res.data_ = ntt::ntt_mul(a.data_, b.data_);
    // res.data_   = mul::mul_digits(a.data_, b.data_);
    res.is_neg_ = a.is_neg_ ^ b.is_neg_;
    res.remove_leading_zero();
    return res;
}

void BigInt::shift_left(Digits& v, uint64_t offset) {
    if (v.size() == 0) {
        unreachable();
    }
    uint64_t idx_offset = offset / DIGIT_BITS, bit_offset = offset % DIGIT_BITS;
    if (bit_offset == 0) {
        v.insert(v.begin(), idx_offset, 0);
        return;
    }
    const size_type new_size = v.size() + idx_offset + 1;
    v.resize(new_size);
    for (size_type i = new_size - 1; i > idx_offset; --i) {
        v[i] = ((v[i - idx_offset] << bit_offset)
                   | (v[i - idx_offset - 1] >> (DIGIT_BITS - bit_offset)))
               & DIGIT_MASK;
    }
    v[idx_offset] = (v[0] << bit_offset) & DIGIT_MASK;
    std::fill(v.begin(), v.begin() + static_cast<int64_t>(idx_offset), 0);
}

void BigInt::shift_right(Digits& v, uint64_t offset) {
    if (v.size() == 0) {
        unreachable();
    }
    uint64_t idx_offset = offset / DIGIT_BITS, bit_offset = offset % DIGIT_BITS;
    if (idx_offset >= v.size()) {
        v.resize(1);
        v[0] = 0;
        return;
    }
    if (bit_offset == 0) {
        v.erase(v.begin(), v.begin() + static_cast<int64_t>(idx_offset));
        return;
    }
    const size_type new_size = v.size() - idx_offset;
    for (size_type i = 0; i < new_size - 1; ++i) {
        v[i] = ((v[i + idx_offset] >> bit_offset)
                   | (v[i + idx_offset + 1] << (DIGIT_BITS - bit_offset)))
               & DIGIT_MASK;
    }
    v[new_size - 1] = v.back() >> bit_offset;
    v.resize(new_size);
}


auto BigInt::operator&=(const BigInt& b) -> BigInt& {
    if (is_zero()) {
        return *this;
    } else if (b.is_zero()) {
        reset();
        return *this;
    }
    const size_type n = data_.size(), m = b.data_.size();
    if (n == 0 || m == 0) {
        unreachable();
    }
    if (n > m) {
        data_.resize(m);
    }
    for (size_type i = 0; i < std::min(n, m); ++i) {
        data_[i] &= b.data_[i];
    }
    remove_leading_zero();
    return *this;
}

auto BigInt::operator|=(const BigInt& b) -> BigInt& {
    if (is_zero()) {
        *this = b;
        return *this;
    } else if (b.is_zero()) {
        return *this;
    }
    const size_type n = data_.size(), m = b.data_.size();
    if (n == 0 || m == 0) {
        unreachable();
    }
    if (n < m) {
        data_.insert(data_.end(), b.data_.begin() + static_cast<int64_t>(n), b.data_.end());
    }
    for (size_type i = 0; i < std::min(n, m); ++i) {
        data_[i] |= b.data_[i];
    }
    remove_leading_zero();
    return *this;
}

auto BigInt::operator^=(const BigInt& b) -> BigInt& {
    if (is_zero()) {
        *this = b;
        return *this;
    } else if (b.is_zero()) {
        return *this;
    }
    const size_type n = data_.size(), m = b.data_.size();
    if (n == 0 || m == 0) {
        unreachable();
    }
    if (n < m) {
        data_.insert(data_.end(), b.data_.begin() + static_cast<int64_t>(n), b.data_.end());
    }
    for (size_type i = 0; i < std::min(n, m); ++i) {
        data_[i] ^= b.data_[i];
    }
    remove_leading_zero();
    return *this;
}

auto BigInt::bitwise_not(size_type len) -> BigInt& {
    if (len) {
        data_.resize(len);
    }
    for (auto& num : data_) {
        num = (~num) & DIGIT_MASK;
    }
    remove_leading_zero();
    return *this;
}

auto BigInt::divmod(const BigInt& b, RoundMode mode, bool check) const
    -> std::pair<BigInt, BigInt> {
    if (b.is_zero()) {
        throw std::domain_error("division by zero");
    }
    if (is_zero()) {
        return {0, 0};
    }
    constexpr size_type PROTECT_PRECISION = 16;

    auto& a = *this;

    const size_type precision = a.data_.size() + PROTECT_PRECISION;

    BigInt Q(BigFloat::ntt_mul(BigFloat(a), BigFloat(b).reciprocal(precision), precision), mode);

    BigInt R(a - Q * b);

    if (!check || R.is_zero()) {
        return {std::move(Q), std::move(R)};
    }
    // 不同的舍入策略对 R 有不同的限制
    // 但无论限制如何，R 超出限制时都必然向 R 的反向调整
    bool adjust = false;
    switch (mode) {
    case RoundMode::Floor:  // 要求：R 与 b 同号且 |R| < |b|
        adjust = (R.is_neg_ ^ b.is_neg_) || (compare_abs(R, b) >= 0);
        break;
    case RoundMode::Ceil:  // 要求：R 与 b 异号且 |R| < |b|
        adjust = !(R.is_neg_ ^ b.is_neg_) || (compare_abs(R, b) >= 0);
        break;
    case RoundMode::RoundHalfUp:  // 要求：|R*2| <= |b| 且取到等号时与 a 异号
    {
        auto cmp = compare_abs(R * 2, b);
        adjust   = (cmp > 0) || (cmp == 0 && !(R.is_neg_ ^ a.is_neg_));
        break;
    }
    case RoundMode::Truncate: [[fallthrough]];
    default:  // 要求：R 与 a 同号且 |R| < |b|
        adjust = (R.is_neg_ ^ a.is_neg_) || (compare_abs(R, b) >= 0);
        break;
    }
    if (adjust) {
        if (R.is_neg_ ^ b.is_neg_) {
            R += b;
            --Q;
        } else {
            R -= b;
            ++Q;
        }
    }
    return {std::move(Q), std::move(R)};
}



BigFloat::BigFloat(double value)
    : point_pos_(0) {
    if (std::isinf(value)) {
        throw std::domain_error("invalid double to BigFloat conversion: inf");
    } else if (std::isnan(value)) {
        throw std::domain_error("invalid double to BigFloat conversion: nan");
    } else if (value == 0) {
        data_.push_back(0);
        is_neg_ = false;
        return;
    }

    auto value_raw_bits = std::bit_cast<uint64_t>(value);

    is_neg_ = value_raw_bits >> (DOUBLE_MANTISSA_LEN + DOUBLE_EXPONENT_LEN);

    auto exponent_bits = static_cast<int64_t>(
        (value_raw_bits >> DOUBLE_MANTISSA_LEN) & ((1 << DOUBLE_EXPONENT_LEN) - 1));

    uint64_t base   = value_raw_bits & ((1ull << DOUBLE_MANTISSA_LEN) - 1);
    int64_t  offset = -DOUBLE_MANTISSA_LEN - DOUBLE_EXPONENT_BIAS;
    if (exponent_bits == 0) {
        offset += 1;
    } else {
        base |= 1ull << DOUBLE_MANTISSA_LEN;
        offset += exponent_bits;
    }

    while (base) {
        data_.push_back(static_cast<uint32_t>(base) & DIGIT_MASK);
        base >>= DIGIT_BITS;
    }

    shift(offset);
}

auto BigFloat::to_double() const -> double {
    if (data_.size() == 0) {
        unreachable();
    }

    if (is_zero()) {
        return 0;
    }

    int64_t highest_bit_width = std::bit_width(data_.back());

    int64_t exponent =
        DIGIT_BITS * (static_cast<int64_t>(data_.size()) - 1 - point_pos_) + highest_bit_width - 1;

    int64_t exponent_bits = exponent + DOUBLE_EXPONENT_BIAS;

    if (exponent_bits >= (1 << DOUBLE_EXPONENT_LEN) - 1) {
        return is_neg_ ? -1 / 0.0 : 1 / 0.0;
    }

    int64_t  offset        = 0;
    uint64_t mantissa_bits = 0;

    auto shl = [](uint64_t x, int64_t r) { return r < 0 ? x >> (-r) : x << r; };

    if (exponent_bits <= 0) {  // 次正规数或下溢
        constexpr int64_t min_subnormal_exponent = -DOUBLE_MANTISSA_LEN - DOUBLE_EXPONENT_BIAS + 1;
        if (exponent < min_subnormal_exponent) {
            return is_neg_ ? -0.0 : 0.0;  // 下溢到 0
        }
        exponent_bits = 0;
        offset        = exponent + DOUBLE_MANTISSA_LEN + DOUBLE_EXPONENT_BIAS - highest_bit_width;
        mantissa_bits = shl(data_.back(), offset);
    } else {  // 正规数
        offset        = DOUBLE_MANTISSA_LEN - (highest_bit_width - 1);
        mantissa_bits = shl(data_.back() & ((1 << (highest_bit_width - 1)) - 1), offset);
    }

    size_type i = data_.size() - 1;

    while (offset > 0 && i > 0) {
        offset -= DIGIT_BITS;
        --i;
        mantissa_bits |= shl(data_[i], offset);
    }

    return std::bit_cast<double>(
        (static_cast<uint64_t>(is_neg_) << (DOUBLE_EXPONENT_LEN + DOUBLE_MANTISSA_LEN))
        | (exponent_bits << DOUBLE_MANTISSA_LEN) | mantissa_bits);
}


void BigFloat::shift(int64_t offset) {
    point_pos_ -= offset / DIGIT_BITS;
    offset %= DIGIT_BITS;
    if (offset == 0) {
        remove_leading_zero();
        return;
    } else if (offset > 0) {
        offset = DIGIT_BITS - offset;
        --point_pos_;
    } else {
        offset = -offset;
    }
    if (std::countr_zero(data_[0]) >= offset) {
        shift_right(data_, offset);
    } else {
        shift_left(data_, DIGIT_BITS - offset);
        ++point_pos_;
    }
    remove_leading_zero();
}


void BigFloat::shift_right(Digits& v, uint32_t offset) {
    if (v.size() == 0 || offset >= DIGIT_BITS) {
        unreachable();
    }
    const size_type n = v.size();
    for (size_type i = 0; i < n - 1; ++i) {
        v[i] = ((v[i] >> offset) | (v[i + 1] << (DIGIT_BITS - offset))) & DIGIT_MASK;
    }
    v[n - 1] >>= offset;
}

void BigFloat::shift_left(Digits& v, uint32_t offset) {
    if (v.size() == 0 || offset >= DIGIT_BITS) {
        unreachable();
    }
    v.push_back(0);
    const size_type n = v.size();
    for (size_type i = n - 1; i > 0; --i) {
        v[i] = ((v[i] << offset) | (v[i - 1] >> (DIGIT_BITS - offset))) & DIGIT_MASK;
    }
    v[0] = (v[0] << offset) & DIGIT_MASK;
}

auto BigFloat::remove_leading_zero() -> size_type {
    size_type i = data_.size();
    if (i == 0) {
        unreachable();
    }
    while (i > 0) {
        --i;
        if (data_[i] != 0)
            break;
    }
    if (i == 0 && data_[0] == 0) {
        reset();
        return 1;
    } else {
        ++i;
        data_.resize(i);
        return i;
    }
}

auto BigFloat::remove_tail_zero() -> size_type {
    if (is_zero()) {
        return 1;
    }
    auto it = data_.begin();
    while (*it == 0) {
        ++it;
    }
    if (it != data_.begin()) {
        point_pos_ -= it - data_.begin();
        data_.erase(data_.begin(), it);
    }
    return data_.size();
}

auto BigFloat::add_or_sub(const BigFloat& a, const BigFloat& b, bool is_sub) -> BigFloat {
    auto len_a = static_cast<int64_t>(a.data_.size());
    auto len_b = static_cast<int64_t>(b.data_.size());
    if (len_a == 0 || len_b == 0) {
        unreachable();
    }
    int64_t point_pos_a = a.point_pos_;
    int64_t point_pos_b = b.point_pos_;
    int64_t tail_zero_a = 0;
    int64_t tail_zero_b = 0;

    for (auto num : a.data_) {
        if (num != 0)
            break;
        ++tail_zero_a;
    }
    for (auto num : b.data_) {
        if (num != 0)
            break;
        ++tail_zero_b;
    }

    if (tail_zero_a == len_a)
        return is_sub ? -b : b;
    if (tail_zero_b == len_b)
        return a;

    int64_t point_pos_res = std::max(point_pos_a - tail_zero_a, point_pos_b - tail_zero_b);

    int64_t len_res = std::max(len_a - point_pos_a, len_b - point_pos_b) + point_pos_res;

    if (len_res < 0) {
        unreachable();
    }

    int64_t offset_a = point_pos_res - point_pos_a;
    int64_t offset_b = point_pos_res - point_pos_b;

    BigFloat res;
    res.data_.resize(len_res);
    res.point_pos_ = point_pos_res;
    res.is_neg_    = a.is_neg_;

    for (auto i = tail_zero_a; i < len_a; ++i) {
        res.data_[i + offset_a] = a.data_[i];
    }

    if (a.is_neg_ ^ b.is_neg_ ^ is_sub) {  // 绝对值相减

        for (auto i = tail_zero_b; i < len_b; ++i) {
            res.data_[i + offset_b] -= b.data_[i];
        }
        res.remove_leading_zero();
        if (res.is_zero())
            return res;
        if (res.data_.back() >> DIGIT_BITS) {
            res.is_neg_ = !res.is_neg_;
            for (auto& num : res.data_) {
                num = -num;
            }
        }
        uint32_t borrow = 0;
        for (auto& num : res.data_) {
            num -= borrow;
            borrow = (num >> DIGIT_BITS) != 0;
            num &= DIGIT_MASK;
        }
        res.remove_leading_zero();

    } else {  // 绝对值相加

        for (auto i = tail_zero_b; i < len_b; ++i) {
            res.data_[i + offset_b] += b.data_[i];
        }
        uint32_t carry = 0;
        for (auto& num : res.data_) {
            num += carry;
            carry = num >> DIGIT_BITS;
            num &= DIGIT_MASK;
        }
        if (carry)
            res.data_.push_back(carry);
        else
            res.remove_leading_zero();
    }
    res.remove_tail_zero();
    return res;
}

auto BigFloat::ntt_mul(const BigFloat& a, const BigFloat& b, size_type output_precision)
    -> BigFloat {
    BigFloat res;
    res.point_pos_ = a.point_pos_ + b.point_pos_;
    res.data_      = ntt::ntt_mul(a.data_, b.data_, output_precision, &res.point_pos_);
    // res.data_   = mul::mul_digits(a.data_, b.data_, output_precision, &res.point_pos_);
    res.is_neg_ = a.is_neg_ ^ b.is_neg_;
    res.remove_leading_zero();
    res.remove_tail_zero();
    return res;
}

void BigFloat::unsigned_inplace_mul(uint64_t b) {
    if (is_zero() || b == 1) {
        return;
    } else if (b == 0) {
        reset();
    } else if ((b & (b - 1)) == 0) {  // b 是 2 的幂
        *this <<= std::countr_zero(b);
    } else if (b <= UINT32_MAX) {
        unsigned_inplace_mul<uint64_t, uint32_t>(b);
    } else {
        unsigned_inplace_mul<uint128_t, uint64_t>(b);
    }
}

template<typename C, typename T> void BigFloat::unsigned_inplace_mul(T b) {
    if (data_.size() == 0) {
        unreachable();
    }
    C digit = 0, carry = 0;
    for (auto& num : data_) {
        digit = static_cast<C>(b) * num + carry;
        num   = static_cast<uint32_t>(digit) & DIGIT_MASK;
        carry = digit >> DIGIT_BITS;
    }
    if (carry) {
        while (carry) {
            data_.push_back(static_cast<uint32_t>(carry) & DIGIT_MASK);
            carry >>= DIGIT_BITS;
        }
    } else {
        remove_leading_zero();
    }
}

void BigFloat::print(std::ostream& output, size_type dec_digits, bool direct) const {
    std::optional<std::stringstream> buffer;
    std::ostream&                    out = direct ? output : buffer.emplace();

    BigInt int_part, dec_part;
    if (point_pos_ < static_cast<int64_t>(data_.size())) {
        int_part.data_.reserve(static_cast<int64_t>(data_.size()) - point_pos_);
        if (point_pos_ < 0) {
            int_part.data_.assign(-point_pos_, 0);
        }
        int_part.data_.insert(
            int_part.data_.end(), data_.begin() + std::max(0ll, point_pos_), data_.end());
    }
    if (int_part.data_.empty()) {
        int_part.data_.push_back(0);
    }

    if (point_pos_ > 0) {
        dec_part.data_.insert(dec_part.data_.begin(),
            data_.begin(),
            data_.begin() + std::min(point_pos_, static_cast<int64_t>(data_.size())));
    }

    if (dec_digits == 0) {
        dec_digits =
            std::floor(static_cast<double>(dec_part.data_.size() * DIGIT_BITS) * std::log10(2));
    }
    if (dec_digits) {
        if (dec_part.data_.empty()) {
            dec_part.data_.push_back(0);
        }
        dec_part.remove_leading_zero();
        auto K = fast_pow(BigInt(TEN / 2), dec_digits);
        dec_part *= K;
        int64_t  offset = DIGIT_BITS * point_pos_ - static_cast<int64_t>(dec_digits);
        BigFloat bf_dec_part(std::move(dec_part), -offset);
        dec_part = BigInt(bf_dec_part, RoundMode::RoundHalfUp);
        // 快速判断 dec_part 是否恰好等于 10^dec_digits
        auto check = [&]() -> bool {
            size_type tail_zero_cnt = 0;
            for (auto num : dec_part.data_) {
                if (num != 0) {
                    tail_zero_cnt += std::countr_zero(num);
                    break;
                }
                tail_zero_cnt += DIGIT_BITS;
            }
            if (tail_zero_cnt != dec_digits) {
                return false;
            }
            return (dec_part >> dec_digits) == K;
        };
        if (check()) {
            dec_part.reset();
            ++int_part;
        }
    }

    if (is_neg_) {
        out.put('-');
    }
    int_part.print_dec(out);
    out.put('.');
    if (dec_digits) {
        dec_part.print_dec(out, dec_digits);
    }

    if (!direct) {
        output << buffer->rdbuf();
    }
}

auto BigFloat::reciprocal(size_type precision) const -> BigFloat {
    if (data_.size() == 0) {
        unreachable();
    }
    if (is_zero()) {
        throw std::domain_error("division by zero");
    }

    if (precision == 0) {
        precision = data_.size();
    }

    int64_t exponent = DIGIT_BITS * (static_cast<int64_t>(data_.size()) - 1 - point_pos_)
                       + std::bit_width(data_.back());

    BigFloat b = *this >> exponent;
    b.remove_tail_zero();
    BigFloat  x(1 / b.to_double());
    size_type cur_precision = DOUBLE_MANTISSA_LEN / DIGIT_BITS;
    // 牛顿迭代：x = x + x * (1 - b * x)
    const BigFloat one(1);
    while (cur_precision < precision) {
        cur_precision *= 2;
        x += ntt_mul(x, one - ntt_mul(b, x, cur_precision + 1), cur_precision + 1);
    }
    x >>= exponent;
    return x;
}

// round_idx：舍去部分的最高位
static void _round(RoundMode mode, Digits& v, bool is_neg, int64_t round_idx) {
    if (round_idx < 0 || round_idx >= static_cast<int64_t>(v.size())) {
        unreachable();
    }
    bool increase_abs = !is_neg;
    switch (mode) {
    case RoundMode::Floor: increase_abs = is_neg; [[fallthrough]];
    case RoundMode::Ceil:
        if (increase_abs) {
            auto i = round_idx;
            for (; i >= 0; --i) {
                if (v[i] != 0) {
                    break;
                }
            }
            increase_abs = i != -1;
        }
        break;
    case RoundMode::RoundHalfUp: increase_abs = v[round_idx] >> (DIGIT_BITS - 1); break;
    case RoundMode::Truncate: [[fallthrough]];
    default: increase_abs = false; break;
    }
    v.erase(v.begin(), v.begin() + round_idx + 1);
    if (!increase_abs) {
        return;
    }
    size_type i = 0, n = v.size();
    for (; i < n; ++i) {
        if (v[i] != DIGIT_MASK) {
            ++v[i];
            break;
        }
    }
    if (i == n) {
        v.push_back(1);
    }
    if (i) {
        std::fill(v.begin(), v.begin() + static_cast<int64_t>(i), 0);
    }
}

void BigFloat::round(RoundMode mode, int64_t precision, RoundRelativeTo relative) {
    if (is_zero()) {
        return;
    }

    const auto n = static_cast<int64_t>(data_.size());

    switch (relative) {
    case RoundRelativeTo::Point: precision = n - point_pos_ - precision; [[fallthrough]];
    case RoundRelativeTo::Significant: [[fallthrough]];
    default: break;
    }

    const int64_t round_idx = n - precision - 1;
    if (round_idx < n) {
        if (round_idx >= 0) {
            point_pos_ -= round_idx + 1;
            _round(mode, data_, is_neg_, round_idx);
            remove_leading_zero();
        } else {
            return;
        }
    } else {
        reset();
    }
}

BigInt::BigInt(const BigFloat& x, RoundMode mode)
    : BigInt(x.is_zero() || x.point_pos_ > static_cast<int64_t>(x.data_.size())
                 ? 0u
                 : convert_from_BigFloat(mode, x.data_, x.is_neg_, x.point_pos_)) {
}

BigInt::BigInt(BigFloat&& x, RoundMode mode)
    : BigInt(x.is_zero() || x.point_pos_ > static_cast<int64_t>(x.data_.size())
                 ? 0u
                 : convert_from_BigFloat(
                       mode, std::move(x.data_), x.is_neg_, std::move(x).point_pos_)) {
    x.reset();
}

auto BigInt::convert_from_BigFloat(RoundMode mode, Digits data, bool is_neg, int64_t point_pos)
    -> BigInt {
    if (point_pos < 0) {
        data.insert(data.begin(), -point_pos, 0);
    } else if (point_pos > 0) {
        if (point_pos > static_cast<int64_t>(data.size())) {
            return 0;
        } else {
            _round(mode, data, is_neg, point_pos - 1);
        }
    }
    BigInt res;
    res.data_   = std::move(data);
    res.is_neg_ = is_neg;
    res.remove_leading_zero();
    return res;
}

}  // namespace bigint
