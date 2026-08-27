#pragma once

#include <compare>
#include <concepts>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>


#include "bigint/bigint_base.h"
#include "bigint/mul.h"


namespace bigint {

class BigInt;
class BigFloat;

enum class RoundMode : uint8_t {
    Truncate,            // 截断（向零取整）
    Floor,               // 向下取整
    Ceil,                // 向上取整
    RoundHalfUp,         // 四舍五入
    Round = RoundHalfUp  // RoundHalfUp 的别名
};

enum class RoundRelativeTo : uint8_t {
    Significant,  // 相对于最高位，保留指定位数
    Point         // 相对于小数点，保留高位
};


// 指定工作线程数 (不调用时使用 std::thread::hardware_concurrency())，
// 传入 0 或 1 以禁用多线程，重复调用无效。
// 此函数是线程安全的，但存在竞争时不保证设置生效。
void init_thread_pool(uint32_t n);


/**
 * @brief 高精度整数。
 *
 * 可以从整型、字符串（std::string）和 BigFloat 构造。
 * 不支持 -0。
 * 支持与小于等于 64 位的整型直接进行运算。
 * 支持位运算，忽略自身的符号。
 */
class BigInt {
public:
    // 从十进制字符串转换和转换到十进制字符串的分治阈值
    static std::array<std::size_t, 2>           DEC_STRING_BRUTE_THRESHOLDS;
    static constexpr std::array<std::size_t, 2> DEC_STRING_BRUTE_THRESHOLDS_DEFAULT = {5000, 2000};

    explicit constexpr BigInt(UnsignedIntegral auto x)
        : BigInt(x, false) {}

    explicit constexpr BigInt(SignedIntegral auto x)
        : BigInt(to_unsigned_abs(x), x < 0) {}

    // 拷贝
    BigInt(const BigInt&)                    = default;
    auto operator=(const BigInt&) -> BigInt& = default;
    ~BigInt()                                = default;

    // 移动：移动后源对象被重置为 0，保持 "data_ 非空" 的不变量，
    // 因此移动后对源对象调用任何公开 API 都安全（而非未指定状态）。
    constexpr BigInt(BigInt&& other) noexcept
        : data_(std::move(other.data_))
        , is_neg_(other.is_neg_) {
        other.reset();
    }
    constexpr auto operator=(BigInt&& other) noexcept -> BigInt& {
        if (this != &other) {
            data_   = std::move(other.data_);
            is_neg_ = other.is_neg_;
            other.reset();
        }
        return *this;
    }

    // 忽略一切非法字符，字符串开头到第一个合法字符之间有 '-' 则结果为负
    explicit BigInt(const std::string& s, bool hex = false);

    explicit BigInt(const BigFloat& x, RoundMode mode = RoundMode::Truncate);

    explicit BigInt(BigFloat&& x, RoundMode mode = RoundMode::Truncate);

    [[nodiscard]] constexpr auto len() const noexcept -> std::size_t { return data_.size(); }

    [[nodiscard]] constexpr auto get_data() const noexcept -> const Digits& { return data_; }

    [[nodiscard]] constexpr auto is_zero() const noexcept -> bool {
        if (data_.size() == 0 || (data_.size() > 1 && data_.back() == 0)) {
            unreachable();
        }
        return data_.size() == 1 && data_[0] == 0;
    }

    [[nodiscard]] constexpr auto sign() const noexcept -> int {
        return is_zero() ? 0 : (is_neg_ ? -1 : 1);
    }

    [[nodiscard]] auto to_string(bool hex = false) const -> std::string {
        std::ostringstream res;
        print(res, hex, true);
        return res.str();
    };

    // hex 模式下不会输出前缀 "0x" ，会输出负号。
    // direct：控制直接输出到流或创建临时 ostringstream 再一次性输出到流中；
    // 开启 direct 时请确保没有可能影响分段输出的流格式标志。
    void print(std::ostream& output, bool hex = false, bool direct = false) const;

    // 仅可能因内存分配抛 bad_alloc；noexcept 下内存耗尽将 terminate（视为不可恢复）。
    constexpr void reset() noexcept {
        data_.resize(1);
        data_[0] = 0;
        is_neg_  = false;
    }

    constexpr void flip_sign() noexcept {
        if (!is_zero())
            is_neg_ = !is_neg_;
    }

    constexpr void remove_sign() noexcept { is_neg_ = false; }

    static auto compare_abs(const BigInt& a, const BigInt& b) noexcept -> std::strong_ordering;

    auto unsigned_inplace_divmod(uint64_t b) -> uint64_t;

    constexpr auto operator+() const -> BigInt { return *this; }

    constexpr auto operator-() const -> BigInt {
        BigInt res(*this);
        res.flip_sign();
        return res;
    }

    auto operator++() -> BigInt&;
    // 会拷贝自身并返回，如无特殊需求请使用前缀自增。
    [[nodiscard]] auto operator++(int) -> BigInt;

    auto operator--() -> BigInt&;
    // 会拷贝自身并返回，如无特殊需求请使用前缀自减。
    [[nodiscard]] auto operator--(int) -> BigInt;

    auto operator+=(const BigInt& b) -> BigInt& {
        inplace_add_or_sub(b, false);
        return *this;
    }

    auto operator+=(Integer64 auto b) -> BigInt& {
        inplace_add_or_sub(to_unsigned_abs(b), b < 0);
        return *this;
    }

    auto operator-=(const BigInt& b) -> BigInt& {
        inplace_add_or_sub(b, true);
        return *this;
    }

    auto operator-=(Integer64 auto b) -> BigInt& {
        inplace_add_or_sub(to_unsigned_abs(b), b > 0);
        return *this;
    }

    auto operator*=(const BigInt& b) -> BigInt& { return *this = *this * b; }

    auto operator*=(Integer64 auto b) -> BigInt& {
        unsigned_inplace_mul(to_unsigned_abs(b));
        if (b < 0)
            flip_sign();
        return *this;
    }

    auto operator/=(Integer64 auto b) -> BigInt& {
        if (b == 0) {
            throw std::domain_error("division by zero");
        }
        if (is_zero()) {
            return *this;
        }
        unsigned_inplace_divmod(to_unsigned_abs(b));
        if (b < 0)
            flip_sign();
        return *this;
    }

    // 低位自动补0。传入负数时调用右移。
    auto operator<<=(Integer64 auto offset) -> BigInt& {
        if (offset > 0)
            shift_left(data_, offset);
        else if (offset < 0)
            shift_right(data_, to_unsigned_abs(offset));
        remove_leading_zero();
        return *this;
    }

    // 移出最低位的数会被丢弃。传入负数时调用左移。
    // 想保留移出最低位的数可使用 BigFloat 的构造函数。
    auto operator>>=(Integer64 auto offset) -> BigInt& {
        if (offset > 0)
            shift_right(data_, offset);
        else if (offset < 0)
            shift_left(data_, to_unsigned_abs(offset));
        remove_leading_zero();
        return *this;
    }

    // 忽略符号。
    auto operator&=(const BigInt& b) -> BigInt&;

    // 忽略符号。
    auto operator|=(const BigInt& b) -> BigInt&;

    // 忽略符号。
    auto operator^=(const BigInt& b) -> BigInt&;

    // 忽略符号，就地转换，将自身长度视为传入值，不传入或传入 0 使用自身长度。
    auto bitwise_not(std::size_t len = 0) -> BigInt&;

    // 当且仅当自身为 0 时为 false。
    explicit constexpr operator bool() const noexcept { return !is_zero(); }

    // 针对编译期 0 的快速比较。
    friend auto operator<=>(const BigInt& a, Literal_zero) noexcept -> std::strong_ordering {
        return a.sign() <=> 0;
    }

    // 针对编译期 0 的快速比较。
    friend auto operator<=>(Literal_zero, const BigInt& b) noexcept -> std::strong_ordering {
        return 0 <=> b.sign();
    }

    // 针对编译期 0 的快速比较。
    friend auto operator==(const BigInt& a, Literal_zero) noexcept -> bool { return a.is_zero(); }
    // 针对编译期 0 的快速比较。
    friend auto operator==(Literal_zero, const BigInt& b) noexcept -> bool { return b.is_zero(); }

    friend auto operator<=>(const BigInt& a, const BigInt& b) noexcept -> std::strong_ordering {
        auto sign_a = a.sign(), sign_b = b.sign();
        if (auto cmp = sign_a <=> sign_b; cmp != 0) {
            return cmp;
        }
        if (sign_a == 0) {
            return std::strong_ordering::equal;
        }
        auto cmp_abs = compare_abs(a, b);
        return sign_a < 0 ? 0 <=> cmp_abs : cmp_abs;
    }

    friend auto operator==(const BigInt& a, const BigInt& b) noexcept -> bool {
        return a.sign() == b.sign() && compare_abs(a, b) == 0;
    }

    // 将自身转换成字符串并输出到流，会检测流的 hex 标志，避免拷贝请使用 print。
    friend auto operator<<(std::ostream& output, const BigInt& a) -> std::ostream& {
        a.print(output, (output.flags() & std::ios_base::hex) != 0, false);
        return output;
    }

    // 从流输入字符串并构造。
    friend auto operator>>(std::istream& input, BigInt& a) -> std::istream& {
        std::string s;
        input >> s;
        a = BigInt(s, (input.flags() & std::ios_base::hex) != 0);
        return input;
    }

    friend auto operator+(BigInt a, const BigInt& b) -> BigInt {
        a += b;
        return a;
    }
    friend auto operator+(BigInt a, Integer64 auto b) -> BigInt {
        a += b;
        return a;
    }
    friend auto operator+(Integer64 auto a, BigInt b) -> BigInt {
        b += a;
        return b;
    }

    friend auto operator-(BigInt a, const BigInt& b) -> BigInt {
        a -= b;
        return a;
    }
    friend auto operator-(BigInt a, Integer64 auto b) -> BigInt {
        a -= b;
        return a;
    }
    friend auto operator-(Integer64 auto a, BigInt b) -> BigInt {
        b -= a;
        b.flip_sign();
        return b;
    }

    friend auto operator*(const BigInt& a, const BigInt& b) -> BigInt {
        if (a.is_zero() || b.is_zero()) {
            return BigInt(0);
        } else {
            BigInt res;
            res.data_   = mul::mul_digits(a.data_, b.data_);
            res.is_neg_ = a.is_neg_ ^ b.is_neg_;
            res.remove_leading_zero();
            return res;
        }
    }
    friend auto operator*(BigInt a, Integer64 auto b) -> BigInt {
        a *= b;
        return a;
    }
    friend auto operator*(Integer64 auto a, BigInt b) -> BigInt {
        b *= a;
        return b;
    }

    friend auto operator/(BigInt a, Integer64 auto b) -> BigInt {
        a /= b;
        return a;
    }

    friend auto operator<<(BigInt a, Integer64 auto offset) -> BigInt {
        a <<= offset;
        return a;
    }
    friend auto operator>>(BigInt a, Integer64 auto offset) -> BigInt {
        a >>= offset;
        return a;
    }

    // 忽略符号。
    friend auto operator&(BigInt a, const BigInt& b) -> BigInt {
        a &= b;
        return a;
    }
    // 忽略符号。
    friend auto operator|(BigInt a, const BigInt& b) -> BigInt {
        a |= b;
        return a;
    }
    // 忽略符号。
    friend auto operator^(BigInt a, const BigInt& b) -> BigInt {
        a ^= b;
        return a;
    }

    // 返回商和余数。可以通过传入 mode 控制商的舍入方向。
    // 不传入或传入枚举范围之外的值视为截断。默认进行结果检查。
    [[nodiscard]] auto divmod(const BigInt& b, RoundMode mode = RoundMode::Truncate,
        bool check = true) const -> std::pair<BigInt, BigInt>;

    static auto get_pow_of_ten(uint32_t exponent) -> BigInt;

    friend class BigFloat;

private:
    Digits data_;

    bool is_neg_;

    explicit constexpr BigInt()
        : is_neg_(false) {}

    explicit constexpr BigInt(UnsignedIntegral auto value, bool is_neg)
        : is_neg_(is_neg) {
        if (value == 0) {
            data_.push_back(0);
        } else if constexpr (sizeof(value) <= sizeof(uint64_t)) {
            // 64 位及以下直接存放，无需循环
            data_.push_back(static_cast<uint64_t>(value));
        } else {
            uint128_t v = value;  // uint128_t 累加器，避免 uint64_t 移位 64 的 UB
            while (v) {
                data_.push_back(static_cast<uint64_t>(v));
                v >>= 64;
            }
        }
    }

    static auto convert_from_dec_string(const std::string& s, std::size_t start, std::size_t end)
        -> std::pair<BigInt, std::size_t>;

    [[nodiscard]] auto to_dec_string_brute() const -> std::string;
    // 假定流标志干净，传入 len 填充前导 0
    void print_dec(std::ostream& output, std::size_t len = 0) const;
    // 假定流标志干净
    void print_hex(std::ostream& output) const;

    auto remove_leading_zero() -> std::size_t;

    void unsigned_self_inc_or_dec(bool is_dec);

    void inplace_add_or_sub(const BigInt& b, bool is_sub);

    void inplace_add_or_sub(uint64_t b, bool is_sub);

    void unsigned_inplace_mul(uint64_t b);

    static void shift_left(Digits& v, uint64_t offset);

    static void shift_right(Digits& v, uint64_t offset);

    static auto convert_from_BigFloat(RoundMode mode, Digits data, bool is_neg, int64_t point_pos)
        -> BigInt;
};

/**
 * @brief 高精度小数。
 *
 * 可以从整型、字符串、double 和 BigInt 构造。
 * 可以转换成 double 。
 * 不支持 -0.0 。
 * 不支持比较，考虑作差并判断符号。
 * 支持与 64 位整数进行乘法和移位。
 * 可以按照给定精度舍入，@see round 。
 */
class BigFloat {
public:
    explicit BigFloat(const BigInt& x, int64_t offset = 0)
        : data_(x.data_)
        , point_pos_(0)
        , is_neg_(x.is_neg_) {
        *this <<= offset;
        remove_tail_zero();
    }

    explicit BigFloat(BigInt&& x, int64_t offset = 0)
        : data_(std::move(x.data_))
        , point_pos_(0)
        , is_neg_(std::move(x).is_neg_) {
        x.reset();
        *this <<= offset;
        remove_tail_zero();
    }

    // 委托给 BigInt 构造
    explicit BigFloat(std::integral auto value, int64_t offset = 0)
        : BigFloat(BigInt(value), offset) {}

    // 委托给 BigInt 构造，不支持处理字符串中的小数点！
    explicit BigFloat(const std::string& s, bool hex = false, int64_t offset = 0)
        : BigFloat(BigInt(s, hex), offset) {}

    explicit BigFloat(double value);

    // 拷贝
    BigFloat(const BigFloat&)                    = default;
    auto operator=(const BigFloat&) -> BigFloat& = default;
    ~BigFloat()                                  = default;

    // 移动：移动后源对象被重置为 0，保持 "data_ 非空" 的不变量，
    // 因此移动后对源对象调用任何公开 API 都安全（而非未指定状态）。
    constexpr BigFloat(BigFloat&& other) noexcept
        : data_(std::move(other.data_))
        , point_pos_(other.point_pos_)
        , is_neg_(other.is_neg_) {
        other.reset();
    }
    constexpr auto operator=(BigFloat&& other) noexcept -> BigFloat& {
        if (this != &other) {
            data_      = std::move(other.data_);
            point_pos_ = other.point_pos_;
            is_neg_    = other.is_neg_;
            other.reset();
        }
        return *this;
    }

    [[nodiscard]] constexpr auto len() const noexcept -> std::size_t { return data_.size(); }

    [[nodiscard]] constexpr auto get_point_pos() const noexcept -> int64_t { return point_pos_; }

    [[nodiscard]] constexpr auto get_data() const noexcept -> const Digits& { return data_; }

    [[nodiscard]] constexpr auto is_zero() const noexcept -> bool {
        if (data_.size() == 0 || (data_.size() > 1 && data_.back() == 0)) {
            unreachable();
        }
        return data_.size() == 1 && data_[0] == 0;
    }

    [[nodiscard]] constexpr auto sign() const noexcept -> int {
        return is_zero() ? 0 : (is_neg_ ? -1 : 1);
    }

    [[nodiscard]] auto to_double() const -> double;

    // 仅可能因内存分配抛 bad_alloc；noexcept 下内存耗尽将 terminate（视为不可恢复）。
    constexpr void reset() noexcept {
        data_.resize(1);
        data_[0]   = 0;
        point_pos_ = 0;
        is_neg_    = false;
    }

    constexpr void flip_sign() noexcept {
        if (!is_zero())
            is_neg_ = !is_neg_;
    }

    constexpr void remove_sign() noexcept { is_neg_ = false; }

    // 输出人类可读的十进制小数，
    // 小数位数不传或传 0 使用二进制小数位数 * log10(2) ，想只输出整数请使用 round 。
    // **不要**将输出的字符串直接传递给 BigFloat 的构造函数，会导致精度损失和丢失小数点信息！
    // direct 参数同 BigInt::print 。
    void print(std::ostream& output, std::size_t dec_digits = 0, bool direct = false) const;

    // 使用默认位数将自身转换为字符串输出到流，避免复制请直接使用 print 。
    friend auto operator<<(std::ostream& output, const BigFloat& a) -> std::ostream& {
        a.print(output, 0, false);
        return output;
    }

    // @see print
    [[nodiscard]] auto to_string(std::size_t dec_digits = 0) const -> std::string {
        std::ostringstream res;
        print(res, dec_digits, true);
        return res.str();
    }

    friend auto operator>>(std::istream& input, BigFloat& a) -> std::istream& {
        std::string s;
        input >> s;
        a = BigFloat(s, (input.flags() & std::ios_base::hex) != 0);
        return input;
    }

    // 求倒数，precision 为 0 时使用输入精度
    [[nodiscard]] auto reciprocal(std::size_t precision = 0) const -> BigFloat;

    // reciprocal 的别名
    [[nodiscard]] auto inv(std::size_t precision = 0) const -> BigFloat {
        return reciprocal(precision);
    }

    [[nodiscard]] constexpr auto operator+() const -> BigFloat { return *this; }

    [[nodiscard]] constexpr auto operator-() const -> BigFloat {
        BigFloat res(*this);
        res.flip_sign();
        return res;
    }

    /**
     * @brief 将自身舍入到给定精度
     * @param mode 舍入模式，@see RoundMode 。超出枚举范围的值相当于 Truncate 。
     * @param precision 舍入精度。
     * @param relative 设置舍入精度相对于最高位还是小数点，@see RoundRelativeTo 。
     *                 超出枚举范围的值相当于 Significant 。
     */
    void round(RoundMode mode, int64_t precision, RoundRelativeTo relative);

    auto operator<<=(int64_t offset) -> BigFloat& {
        shift(offset);
        return *this;
    }

    auto operator>>=(int64_t offset) -> BigFloat& {
        shift(-offset);
        return *this;
    }

    auto operator+=(const BigFloat& b) -> BigFloat& {
        *this = *this + b;
        return *this;
    }

    auto operator-=(const BigFloat& b) -> BigFloat& {
        *this = *this - b;
        return *this;
    }

    auto operator*=(Integer64 auto b) -> BigFloat& {
        unsigned_inplace_mul(to_unsigned_abs(b));
        if (b < 0)
            flip_sign();
        return *this;
    };

    // 建议使用 mul 指定精度
    auto operator*=(const BigFloat& b) -> BigFloat& {
        *this = *this * b;
        return *this;
    }

    friend auto operator<<(BigFloat x, int64_t offset) -> BigFloat {
        x <<= offset;
        return x;
    }
    friend auto operator>>(BigFloat x, int64_t offset) -> BigFloat {
        x >>= offset;
        return x;
    }

    friend auto operator+(const BigFloat& a, const BigFloat& b) -> BigFloat {
        return add_or_sub(a, b, false);
    }

    friend auto operator-(const BigFloat& a, const BigFloat& b) -> BigFloat {
        return add_or_sub(a, b, true);
    }

    // 建议使用 mul 指定精度
    friend auto operator*(const BigFloat& a, const BigFloat& b) -> BigFloat {
        if (a.is_zero() || b.is_zero()) {
            return BigFloat(0);
        } else {
            return mul(a, b, 0);
        }
    }

    friend auto operator*(BigFloat a, Integer64 auto b) -> BigFloat {
        a *= b;
        return a;
    }
    friend auto operator*(Integer64 auto a, BigFloat b) -> BigFloat {
        b *= a;
        return b;
    }

    static auto mul(const BigFloat& a, const BigFloat& b, std::size_t precision) -> BigFloat {
        if (a.is_zero() || b.is_zero()) {
            return BigFloat(0);
        } else {
            BigFloat res;
            res.point_pos_ = a.point_pos_ + b.point_pos_;
            res.data_      = mul::mul_digits(a.data_, b.data_, precision, &res.point_pos_);
            res.is_neg_    = a.is_neg_ ^ b.is_neg_;
            res.remove_leading_zero();
            res.remove_tail_zero();
            return res;
        }
    }

    friend class BigInt;

private:
    Digits  data_;
    int64_t point_pos_;
    bool    is_neg_;

    explicit constexpr BigFloat()
        : point_pos_(0)
        , is_neg_(false) {}

    void shift(int64_t offset);

    static void shift_right(Digits& v, uint32_t offset);

    static void shift_left(Digits& v, uint32_t offset);

    auto remove_leading_zero() -> std::size_t;

    auto remove_tail_zero() -> std::size_t;

    static auto add_or_sub(const BigFloat& a, const BigFloat& b, bool is_sub) -> BigFloat;

    void unsigned_inplace_mul(uint64_t b);
};

template<class T>
concept can_remove_sign = requires(T X) {
    { X.remove_sign() };
};

template<can_remove_sign T> auto abs(const T& X) -> T {
    auto res = X;
    res.remove_sign();
    return res;
}

struct BigIntPrintHelper {
    const BigInt* ptr;
    bool          hex;
    bool          direct;

    friend auto operator<<(std::ostream& output, const BigIntPrintHelper& helper) -> std::ostream& {
        helper.ptr->print(output, helper.hex, helper.direct);
        return output;
    }
};

struct BigFloatPrintHelper {
    const BigFloat* ptr;
    std::size_t     dec_digits;
    bool            direct;

    friend auto operator<<(std::ostream& output, const BigFloatPrintHelper& helper)
        -> std::ostream& {
        helper.ptr->print(output, helper.dec_digits, helper.direct);
        return output;
    }
};

// 流式代理输出，参数 @see BigInt::print，与 operator<< 不同，不会检测流标志
inline auto print(const BigInt& BigInt_to_print, bool hex = false, bool direct = false)
    -> BigIntPrintHelper {
    return {.ptr = &BigInt_to_print, .hex = hex, .direct = direct};
}

// 流式代理输出，参数 @see BigFloat::print
inline auto print(const BigFloat& BigFloat_to_print, std::size_t dec_digits = 0,
    bool direct = false) -> BigFloatPrintHelper {
    return {.ptr = &BigFloat_to_print, .dec_digits = dec_digits, .direct = direct};
}

}  // namespace bigint
