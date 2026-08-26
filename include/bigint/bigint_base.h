#pragma once

#ifndef __SIZEOF_INT128__
#    error "This library requires __uint128_t."
#endif

#include <array>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <tuple>
#include <type_traits>
#include <vector>


namespace bigint {

// 内部容器
using Digits = std::vector<uint32_t>;

// 基数位宽，必须保证基数小于所有模数；
// print_hex 依赖此数是 4 的倍数
constexpr uint32_t DIGIT_BITS = 28;
constexpr uint32_t DIGIT_MASK = (1u << DIGIT_BITS) - 1u;

constexpr int TEN                  = 10;
constexpr int DOUBLE_MANTISSA_LEN  = 52;
constexpr int DOUBLE_EXPONENT_LEN  = 11;
constexpr int DOUBLE_EXPONENT_BIAS = 1023;

using uint128_t = __uint128_t;
using int128_t  = __int128_t;

using size_type = std::size_t;

// 编译期常量 0
using Literal_zero = decltype(nullptr);

inline void unreachable() {
#ifdef NDEBUG
    __builtin_unreachable();
#else
    assert(false);
#endif
}

// 无符号整型。
template<typename T>
concept UnsignedIntegral =
    !std::is_same_v<std::remove_cv_t<T>, bool>
    && (std::is_unsigned_v<T> || std::is_same_v<std::remove_cv_t<T>, uint128_t>);

// 有符号整型。
template<typename T>
concept SignedIntegral = !std::is_same_v<std::remove_cv_t<T>, bool>
                         && (std::is_signed_v<T> || std::is_same_v<std::remove_cv_t<T>, int128_t>);

static_assert(UnsignedIntegral<uint128_t> && SignedIntegral<int128_t>);

template<typename T>
concept Integral = UnsignedIntegral<T> || SignedIntegral<T>;

// 小于等于 64 位的整型。
template<typename T>
concept Integer64 =
    Integral<T> && std::numeric_limits<T>::digits <= std::numeric_limits<uint64_t>::digits;

template<typename T> struct make_unsigned {
    using type = std::make_unsigned_t<T>;
};

template<> struct make_unsigned<int128_t> {
    using type = uint128_t;
};

template<> struct make_unsigned<uint128_t> {
    using type = uint128_t;
};

template<typename T> using make_unsigned_t = typename make_unsigned<std::remove_cv_t<T>>::type;

// 将有符号整数的绝对值转换为对应的无符号类型，对无符号不起作用。
template<Integral T> constexpr auto to_unsigned_abs(T x) -> make_unsigned_t<T> {
    using U = make_unsigned_t<T>;
    return x >= 0 ? static_cast<U>(x) : -static_cast<U>(x);
}

static_assert(to_unsigned_abs(int128_t(-1)) == uint128_t(1));

template<typename T> constexpr auto fast_pow(T base, uint32_t exponent) -> T {
    T res(1);
    while (exponent) {
        if (exponent & 1)
            res *= base;
        base *= base;
        exponent >>= 1;
    }
    return res;
}

constexpr auto fast_pow(uint32_t base, uint32_t exponent, uint32_t mod) -> uint32_t {
    uint64_t res = 1, _base = base;
    while (exponent) {
        if (exponent & 1)
            res = res * _base % mod;
        _base = _base * _base % mod;
        exponent >>= 1;
    }
    return res;
}

constexpr auto floor_log(uint64_t base, uint64_t x) -> size_type {
    size_type n     = 0;
    uint64_t  power = 1;
    while (power <= x / base) {
        power *= base;
        ++n;
    }
    return n;
}

// 计算 (a + b) mod p，a 和 b 必须小于 p
template<std::unsigned_integral T> constexpr inline auto add_mod(T a, T b, T p) -> T {
    if (a >= p || b >= p) {
        unreachable();
    }
    T d = p - b;
    return a < d ? a + b : a - d;
}

// 计算 (a - b) mod p，a 和 b 必须小于 p
template<std::unsigned_integral T> constexpr inline auto sub_mod(T a, T b, T p) -> T {
    if (a >= p || b >= p) {
        unreachable();
    }
    const T t = a - b;
    return t + ((T(0) - (t > a)) & p);
}

constexpr auto exgcd(int64_t a, int64_t b) -> std::array<int64_t, 3> {
    int64_t x1 = 1, x2 = 0, x3 = 0, x4 = 1;
    while (b != 0) {
        int64_t c = a / b;
        std::tie(x1, x2, x3, x4, a, b) =
            std::make_tuple(x3, x4, x1 - x3 * c, x2 - x4 * c, b, a - b * c);
    }
    return {a, x1, x2};
}

constexpr auto modinv(int64_t a, int64_t m) -> int64_t {
    return ((exgcd(a, m)[1] % m) + m) % m;
}

// 计算 ceil(a/b)
template<std::unsigned_integral T> constexpr auto ceil_div(T a, T b) -> T {
    return a / b + (a % b != 0);
}

// res = a + b，返回是否发生溢出
template<std::unsigned_integral T> constexpr auto add_overflow(T a, T b, T& res) -> bool {
    res = a + b;
    return res < b;
}

// res = a - b，返回是否发生溢出
template<std::unsigned_integral T> constexpr auto sub_overflow(T a, T b, T& res) -> bool {
    res = a - b;
    return res > a;
}

}  // namespace bigint
