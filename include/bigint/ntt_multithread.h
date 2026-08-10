#pragma once

#include <array>
#include <cstdint>


#include "bigint/bigint_base.h"
#include "bigint/ntt.h"


namespace bigint::ntt_multithread {

// Montgomery 域的交织式多模数 NTT (interleaved multi-modulus NTT)，
// 对下标模 NUM_PRIMES 同余类分别应用模数为 P[i] 的 NTT，
// 务必保证传入的数组长度是 2 的幂 * NUM_PRIMES 。
void imm_ntt(Digits& v, bool rev);

void imm_mul_num(Digits& v, std::array<uint32_t, ntt::NUM_PRIMES> nums);

void imm_mul_vec(Digits& v, const Digits& v2);

// 利用中国剩余定理从 imm_ntt 逆变换结果中复原出真实数据，
// 拆成 3 个 DIGIT_BITS 位二进制数写回原位，
// 调用者应当保证这样的拆分是安全的。
void crt_merge(Digits& v);

}  // namespace bigint::ntt_multithread
