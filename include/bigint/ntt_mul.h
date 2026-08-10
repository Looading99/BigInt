#pragma once

#include "bigint/bigint_base.h"


namespace bigint::ntt {

// NTT 乘法接口，将两个数组（按照 output_precision ）相乘。必须保证输入数组非全0。
auto ntt_mul(const Digits& a, const Digits& b, size_type output_precision = 0,
    int64_t* p_result_point_pos = nullptr) -> Digits;

}  // namespace bigint::ntt
