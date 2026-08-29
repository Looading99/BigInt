# BigInt

基于 **FFT / 多模数 NTT / SSA 多算法自动分发** 的高性能高精度整数 / 高精度小数 C++20 库，使用多线程 NTT 与 AVX2+FMA 向量化加速。

## 目录

- [特性](#特性)
- [目录结构](#目录结构)
- [构建](#构建)
- [测试](#测试)
- [安装与集成](#安装与集成)
  - [方式一：find_package](#方式一find_package)
  - [方式二：add_subdirectory](#方式二add_subdirectory)
  - [方式三：直接包含头文件](#方式三直接包含头文件)
- [使用示例](#使用示例)
- [API 概览](#api-概览)
- [实现说明](#实现说明)
- [已知限制](#已知限制)

## 特性

- **高精度整数 `BigInt`**：支持任意位长的整数四则运算、位运算、比较、字符串互转、快速幂、`get_pow_of_ten`、`divmod` 及多种舍入模式。
- **高精度小数 `BigFloat`**：支持从 `BigInt` / 整型 / 字符串（不支持解析字符串中的小数点）构造、与 `double` 互转、转换成字符串、按精度舍入、求倒数（`reciprocal`，别名`inv`）。
- **多算法快速乘法**：按输入规模自动分发 `brute → fft → ntt → ssa`；FFT 采用 DIF/DIT 基-2 混合 radix 变换与动态 digit_bits（AVX2+FMA），NTT 为三模数 + Montgomery 模乘，SSA 处理超大规模。
- **多线程**：`init_thread_pool` 可指定工作线程数，NTT 大数乘法自动并行；本库线程安全，可被多个线程并发使用。
- **易于集成**：CMake 静态库目标 `bigint::bigint`，支持 `find_package` 安装导出。

## 目录结构

```
include/bigint/
    bigint_base.h        # 基础类型与工具（Digits、128 位类型支持、Integral/make_unsigned、模运算）
    bigint.h             # BigInt / BigFloat 公开 API
    mul.h                # 统一乘法接口（brute/fft/ntt/ssa 分发、mul_digits）
    ntt_base.h           # NTT 基础（常量定义、Montgomery 算法、SIMD 实现）
    aligned_allocator.h  # SIMD 对齐分配器（FFT使用）
src/
    bigint.cpp           # BigInt / BigFloat 实现
    mul/
        fft.cpp              # FFT 乘法（DIF/DIT 基-2 混合 radix、动态 digit_bits、AVX2+FMA）
        ntt_multithread.cpp  # 多线程 NTT 乘法
        ssa.cpp              # SSA 乘法
tests/
    test_common.h            # 公共测试工具（固定种子随机源 + 基数 2^64 朴素参考实现）
    test_*.cpp               # 按功能域拆分的 9 个测试程序（见“测试”章节）
    measure_digit_bits.cpp   # FFT digit_bits 精度测量与随机回归（目标 test_fft_digit_bits）
examples/
    benchmark.cpp        # 基准测试示例程序（目标 bigint_benchmark）
    binary_split.cpp     # 二进制分裂算法模板（ main() 中为计算e的示例）（目标 bst）
    pi.cpp               # 使用二进制分裂算法计算任意位数的 pi （目标 pi）
cmake/
    bigintConfig.cmake.in  # find_package 配置模板
```


## 构建

环境要求：

- CMake ≥ 3.15
- 支持 C++20 的编译器：GCC ≥ 10 或 Clang ≥ 12
- 需要 `__uint128_t`/`__int128_t` 与 AVX2 + FMA（`-mavx2 -mfma`）支持
- ⚠️ **MSVC 编译器不支持**（缺 `__uint128_t`）；**clang 的 Windows MSVC target**（如 LLVM 官方二进制）已支持——CMake 会自动链接 compiler-rt builtins 提供 128 位除法/取模运行时函数（CI 中以 clang++ GNU 驱动 + MSVC target 验证，Debug 严格警告 `-Werror` 全绿）
- CMake 生成器无强制要求，可按平台选用，如：

```bash
# Linux / macOS（gcc/clang，默认 Unix Makefiles）
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build

# Windows + MinGW-w64（gcc/clang，需 mingw32-make 在 PATH 中）
cmake -S . -B build -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release
cmake --build build

# 各平台通用：也可使用 Ninja
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

构建产物：

- `libbigint.a` — 静态库
- examples 中的示例（`bigint_benchmark` / `bst` / `pi`，`-DBUILD_EXAMPLES=OFF` 可关闭）
- 9 个按功能域拆分的测试程序（`test_constructors` / `test_arithmetic` / `test_divmod` / `test_shift_bitwise` / `test_compare` / `test_object` / `test_string` / `test_large` / `test_bigfloat`，`-DBUILD_TESTS=OFF` 可关闭）
- `test_fft_digit_bits` — FFT 精度测试与乘法回归（随 BUILD_TESTS 开关）

## 测试

```bash
# 一键运行全部测试（CTest，每个功能域一个测试目标）
ctest --test-dir build --output-on-failure

# 单独运行
build/test_constructors     # 构造：整型（含 128 位）/ limb 数组 / 字符串 / 流输入
build/test_arithmetic       # 加减乘 / 自增自减 / 64 位混合 / 进位借位长链
build/test_divmod           # divmod 四种舍入模式 / 除零 / 单 limb 除法
build/test_shift_bitwise    # 移位（含负偏移）/ 位运算 / bitwise_not
build/test_compare          # <=> 全序 / 与 0 的快速比较 / compare_abs
build/test_object           # 拷贝 / 移动语义 / reset / 符号操作
build/test_string           # dec/hex 输出 / print / 流 / 分治解析
build/test_large            # 大数：乘法分发全深度 / 大数除法 / get_pow_of_ten
build/test_bigfloat         # BigFloat：double 往返 / 加减乘对拍 / round / reciprocal / 输出
build/test_fft_digit_bits regress    # 乘法回归：FFT 各 digit_bits vs NTT 参考
build/test_fft_digit_bits measure    # 重测 FFT 精度表（输出可直接回填 mul.h）
```

- 测试按功能域拆分，共享 `tests/test_common.h`（固定种子随机源 + 基数 2^64 朴素参考实现，确定性可复现）。验证策略：
  1. 中小规模与独立参考**随机对拍**：加减乘、divmod 四种舍入、BigFloat 加减乘的 limb 级比对、字符串/hex 解析与输出；
  2. 大数用数学可推导的**结构模式验证**（如 `(2^64k-1)²` 的 limb 布局），覆盖 brute → FFT 全深度直至 FFT 容量上限（~84M bit），免 O(n²) 参考；
  3. 大十进制串的**分治/暴力解析路径交叉验证**（`DEC_STRING_BRUTE_THRESHOLDS` 三配置一致性）。
- 可选 NTT 路径回归（规模超出 FFT 上限，耗时较长，默认跳过）：`BIGINT_TEST_NTT=1 build/test_large`。
- CI（GitHub Actions）：ubuntu（gcc / clang++ × Debug/Release）+ windows msys2（g++ × Debug/Release）+ windows-msvc-clang（clang++ + MSVC STL × Debug/Release），Debug 构建零警告（`-Werror`）并运行 `ctest`。

## 安装与集成

安装到指定前缀（默认系统目录）：

```powershell
cmake --install build --prefix <你的安装前缀>
```

### 方式一：find_package

安装后，消费方项目中：

```cmake
find_package(bigint CONFIG REQUIRED)

add_executable(app main.cpp)
target_link_libraries(app PRIVATE bigint::bigint)
```

配置时通过 `-DCMAKE_PREFIX_PATH=<安装前缀>` 或 `-Dbigint_DIR=<前缀>/lib/cmake/bigint` 指定位置。消费方应与库使用**同一编译器工具链**。

### 方式二：add_subdirectory

将本项目作为子目录引入：

```cmake
add_subdirectory(path/to/BigInt)

add_executable(app main.cpp)
target_link_libraries(app PRIVATE bigint::bigint)
```

### 方式三：直接包含头文件

将 `include/bigint` 加入头文件搜索路径，并把 `libbigint.a` 加入链接即可。

## 使用示例

```cpp
#include <iostream>

#include "bigint/bigint.h"

using bigint::BigInt;
using bigint::BigFloat;

int main() {
    bigint::init_thread_pool(8);         // 指定 8 个工作线程；不调用则默认使用硬件并发线程数

    // 从字符串 / 整型构造
    BigInt a("123456789012345678901234567890");
    BigInt b(114514);

    // 大整数快速幂
    BigInt c = bigint::fast_pow(b, 1919810);
    std::cout << "114514^1919810 len=" << c.len() << "\n";

    // 四则运算、比较（C++20 三路比较）
    BigInt sum = a + c * 3 - b;
    if (sum > c) { /* ... */ }

    // 转十进制字符串 / 输出到流
    std::string s = c.to_string();  // 需要字符串时使用
    c.print(std::cout, false, true); std::cout << '\n';     // 直接输出到流，避免临时字符串复制（推荐）
    // std::cout << bigint::print(c, false, true) << '\n';  // 或流式输出代理，其余参数与成员函数版相同（推荐）
    // std::cout << std::hex << c << '\n';                  // 或用 operator<< ，会自动检测流的 hex 标志，但只能构建临时流后输出

    // 高精度小数
    BigFloat x(3.141592653589793);
    BigFloat y = x * (x + x);           // 加法、乘法
    BigFloat y_inv = y.reciprocal(10);  // （或使用y.inv）求倒数，指定 10 个 64 位 limb 的内部精度
    y.round(bigint::RoundMode::RoundHalfUp, 50, bigint::RoundRelativeTo::Point); // 按精度舍入
    double d = y.to_double();           // 转回 double
    std::cout << bigint::print(y, 10, true) << '\n'  // 以十进制小数格式输出 10 位小数
    // 也可使用 y.to_string()，y.print()，std::cout << y 等

    return 0;
}
```
> [!WARNING]
> 1.性能提示：`operator<<` 内部为隔离流的格式标志会构造临时流再拷贝，建议直接调用成员函数 `print()` 一次性写入目标流，避免额外的内存分配与复制。也可用流式代理 `std::cout << bigint::print(x)`（其余参数与成员函数版相同）<br>
> 2.语义差异：`BigInt` 的 `operator<<` 会检测流的 `hex` 标志（`std::cout << std::hex << x` 输出十六进制）；而 `print(x)` 默认十进制、不受流标志影响，需要十六进制时请显式 `print(x, true)`

## API 概览

### `bigint::BigInt` — 高精度整数

| 类别 | API |
| --- | --- |
| 构造 | 任意无/有符号整型<sup>[1]</sup>、`std::string`（`hex=false` 十进制 / `true` 十六进制）、64 位 limb 数组（`std::span<const uint64_t>` 复制 / `Digits&&` 移动，可指定符号）、`BigFloat`（移动/拷贝；可指定舍入模式）<sup>[2]</sup> |
| 查询 | `len()`、`get_data()`、`is_zero()`、`sign()` |
| 转换 | `to_string(hex=false)`、`print(ostream, hex=false, direct=false)` |
| 算数运算 | `+ - * / += -= *= /=`<sup>[3]</sup>、`++ --`<sup>[4]</sup>、一元 `+ -`、`unsigned_inplace_divmod(uint64_t)`、`divmod(BigInt, RoundMode)` |
| 位运算<sup>[5]</sup> |  `& \| ^ << >>` 及对应复合赋值<sup>[6]</sup>、`bitwise_not(len)`原地按位取反（无 `~` ） |
| 比较 |`compare_abs(a, b)`无符号比较、`<=>`、`==`（含与编译期常量 `0`<sup>[7]</sup> 的快速比较） |
| 其他 | `reset()`、`flip_sign()`、`remove_sign()`、流输入输出 `<< >>`、`get_pow_of_ten()` |

>注：<br>
>[1] :包括`__uint128_t`和`__int128_t` <br>
>[2] :除拷贝/移动外，构造函数均为 `explicit`，需直接初始化（不支持隐式转换）<br>
>[3] :`+ - *` 及对应的复合赋值支持 64 位整数和`BigInt`，`/`和`/=`仅支持64位整数（与`BigInt`间求商余请使用`divmod`，支持四种商的舍入模式）；另外，`+= -=`是原地的， `*= /=` 与 64 位整型之间是原地的，可优先考虑使用它们 <br>
>[4] :包括前置和后置，后置会返回值改变前的副本，如无特殊需求请使用前置 <br>
>[5] :位运算忽略`BigInt`的符号，且运算后移除前导零 <br>
>[6] :复合赋值是原地的；移位可传入负数，相当于反向移位 <br>
>[7] :隐式转换为 `nullptr`

### `bigint::BigFloat` — 高精度小数

- 构造：`BigInt`（移动/拷贝）、整型、字符串（可带 `offset`）、64 位 limb 数组（`std::span<const uint64_t>` 复制 / `Digits&&` 移动，可带符号与 `offset`）、`double`（除拷贝/移动外均为 `explicit`）
- 方法：`sign()`、`to_double()`、`to_string(dec_digits)`、`print(ostream, dec_digits, direct)`、`reciprocal(precision)`（别名 `inv(precision)`）、`round(mode, precision, relative)`、`get_data()`、`get_point_pos()`等
- 运算：`+ -`、乘法`*`和`mul(a, b, precision)`（`*`无精度限制，`mul`需传入目标精度，精度单位为 64 位 limb）、`<< >> <<= >>=`（二进制移位，等价于乘/除以 2 的幂，可传入负数）、一元 `+ -`
- 舍入模式 `RoundMode`：`Truncate` / `Floor` / `Ceil` / `RoundHalfUp`（别名 `Round`）
- 精度参照 `RoundRelativeTo`：`Significant`（相对最高位，可理解为: `114.514, 4 -> 114.5`）/ `Point`（相对小数点，可理解为：`114.514, 1 -> 110`/`114.514, -1 -> 114.5`）
- 说明：不支持比较，建议作差后判断符号；不支持 `-0.0`

### 顶层函数

- `init_thread_pool(uint32_t n)` — 设置工作线程数（`0`/`1` 禁用多线程；不调用时默认 `hardware_concurrency()`；重复调用无效）
- `fast_pow(T base, uint32_t exponent)` — 快速幂模板
- `abs(const T&)` — 绝对值（适用于 `BigInt` / `BigFloat`）
- `print(const BigInt&, bool hex=false, bool direct=false)` — 流式输出代理，配合 `operator<<`：`std::cout << bigint::print(x, true)` 输出十六进制
- `print(const BigFloat&, std::size_t dec_digits=0, bool direct=false)` — 同理，可指定小数位数

## 实现说明

- **表示方式**：以 `2^64` 为基数（`DIGIT_BITS = 64`）、小端序 `uint64_t` 数组存储，无前导零；乘法等内部实现与存储表示统一为 64 位块，无任何基数转换。
- **乘法**：按输入规模自动分发 `brute → fft → ntt → ssa`（统一入口 `bigint::mul::mul_digits`，输入输出直接为 64 位 limb）：
  - **朴素**：小规模（≤ ~16K bit）
  - **FFT**：DIF/DIT 基-2 混合 radix 变换（长度可为任意 2 的幂）+ 动态 digit_bits（默认下限 5，覆盖约 84M bit），AVX2+FMA 向量化（单线程）
  - **NTT**：三模数 + Montgomery 模乘 + SIMD，多线程，覆盖至约 33M bit
  - **SSA**：超过 NTT 容量的超大规模
- **理论规模上限**：FFT 与 NTT 各有容量上限；SSA 无固定上限（仅受内存与时间约束）。
- **NTT线程池**：有超时忙等（默认`5ms`）+ 动态块大小 + 工作窃取 的任务模型，8 线程时相对单线程约 3.6~4.3× 加速。

## 已知限制

- NTT 线程池的线程数超过 CPU 并行数（排除 LPE 核心）时性能下降严重，建议根据具体 CPU 情况手动指定线程数。
- FFT 目前为单线程（仅 NTT 支持多线程）。
- MSVC 编译器不支持（缺 `__uint128_t`）；建议使用 GCC / Clang，或 clang 的 Windows MSVC target。
- `BigFloat` 不支持比较运算，建议作差后判断符号。
- 部分功能依赖 AVX2 与 FMA，需在支持的 CPU 上编译运行。
