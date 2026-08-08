# BigInt

基于 **多模数NTT** 的高性能高精度整数 / 高精度小数 C++20 库，支持多线程与 AVX2 向量化加速。

## 特性

- **高精度整数 `BigInt`**：支持任意位长的整数四则运算、位运算、比较、字符串互转、快速幂、`get_pow_of_ten`、`divmod` 及多种舍入模式。
- **高精度小数 `BigFloat`**：支持从 `BigInt` / 整型 / 字符串（不支持解析字符串中的小数点）构造、与 `double` 互转、按精度舍入、求倒数（`reciprocal`）。
- **NTT 快速乘法**：三模数 NTT + Montgomery 模乘 + AVX2 SIMD 向量化。
- **多线程**：`init_thread_pool` 可指定工作线程数，大数乘法自动并行；本库线程安全，可被多个线程并发使用。
- **易于集成**：CMake 静态库目标 `bigint::bigint`，支持 `find_package` 安装导出。

## 目录结构

```
include/bigint/
    bigint_base.h        # 基础类型与工具（Digits、模运算、快速幂等）
    bigint.h             # BigInt / BigFloat 公开 API
    ntt.h                # NTT 基础（常量定义、位逆序置换、Montgomery 算法、SIMD 实现）
    ntt_multithread.h    # 多线程 NTT 接口
src/
    bigint.cpp           # BigInt / BigFloat 实现
    ntt_multithread.cpp  # 多线程 NTT 实现
examples/
    main.cpp             # 基准测试示例程序（目标 bigint_benchmark）
cmake/
    bigintConfig.cmake.in  # find_package 配置模板
```

## 环境要求

- CMake ≥ 3.15
- 支持 C++20 的编译器：**GCC ≥ 10** 或 **Clang ≥ 12**（MinGW 环境，如 msys2 ucrt64 / clang64）
- 需要 `__uint128_t` 与 AVX2（`-mavx2`）支持
- ⚠️ 不支持 MSVC（缺少 `__uint128_t`）

## 构建

本项目使用 **MinGW Makefiles** 生成器。

```powershell
cmake -S . -B build -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

构建产物：

- `libbigint.a` — 静态库
- `bigint_benchmark.exe` — 基准测试示例（`-DBUILD_EXAMPLES=OFF` 可关闭）

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
    std::cout << "114514^1919810 has " << c.len() * bigint::DIGIT_BIS << " bits\n";

    // 四则运算、比较（C++20 三路比较）
    BigInt sum = a + c * 3 - b;
    if (sum > c) { /* ... */ }

    // 转十进制字符串 / 输出到流
    std::string s = c.to_string();            // 需要字符串时使用
    c.print(std::cout, false, true); std::cout << '\n';  // 直接输出到流，避免临时字符串复制（推荐）
    // std::cout << c << '\n';                // 或用 operator<<（内部经临时流）

    // 高精度小数
    BigFloat x(3.141592653589793);
    BigFloat y = x * (x + x);              // 加法、乘法
    BigFloat inv = y.reciprocal(10);       // 求倒数，指定 10 位内部数字的精度
    y.round(bigint::RoundMode::RoundHalfUp, 50, bigint::RoundRelativeTo::Point); // 按精度舍入
    double d = y.to_double();              // 转回 double

    return 0;
}
```

> 💡 **性能提示**：`to_string()` 与 `operator<<` 内部都会构造临时字符串/临时流再拷贝。如果只是把大数输出到流（`std::cout`、文件等），建议直接调用 `print(output, hex, direct)` 一次性写入目标流，避免额外的内存分配与复制。`direct` 参数：`true` 直接写入目标流（要求流格式标志干净）；`false`（默认）先写入临时流再输出，以隔离调用方流的格式标志。

## API 概览

### `bigint::BigInt` — 高精度整数

| 类别 | API |
| --- | --- |
| 构造 | 无/有符号整型（模板）、`std::string`（`hex=false` 十进制 / `true` 十六进制）、`BigFloat`（移动/拷贝；可指定舍入模式） |
| 查询 | `len()`、`get_data()`、`is_zero()`、`sign()` |
| 转换 | `to_string(hex=false)`、`print(ostream, hex=false, direct=false)` |
| 算数运算 | `+ - * /`、`+= -= *= /=`、`++ --`、一元 `+ -`、`unsigned_inplace_divmod(uint64_t b)`就地除法 |
| 位运算\* |  `& \| ^ << >>` 及对应 `=` 版本、`bitwise_not(len)`原地按位取反 |
| 比较 |`compare_abs(a, b)`无符号比较、`<=>`、`==`（含与编译期常量 `0` 的快速比较） |
| 其他 | `divmod(b, mode)` 带舍入的商余、`get_pow_of_ten()`、`reset()`、`flip_sign()`、`remove_sign()`、流输入输出 `<< >>` |

> \* 位运算忽略`BigInt`的符号，且自动移除前导零。

### `bigint::BigFloat` — 高精度小数

- 构造：`BigInt`（移动/拷贝）、整型、字符串（可带 `offset`）、`double`
- 方法：`sign()`、`to_double()`、`reciprocal(precision)`、`round(mode, precision, relative)`、`get_data()`、`get_point_pos()`等
- 运算：`+ -`、乘法`*`和`mul(a, b, precision)`（`*`无精度限制，`mul`需传入目标精度）、`<<= >>=`（二进制移位）、一元 `+ -`
- 舍入模式 `RoundMode`：`Truncate` / `Floor` / `Ceil` / `RoundHalfUp`（别名 `Round`）
- 精度参照 `RoundRelativeTo`：`Significant`（相对最高位）/ `Point`（相对小数点）
- 说明：不支持比较，建议作差后判断符号；不支持 `-0.0`

### 顶层函数

- `init_thread_pool(uint32_t n)` — 设置工作线程数（`0`/`1` 禁用多线程；不调用时默认 `hardware_concurrency()`；重复调用无效）
- `fast_pow(T base, uint32_t exponent)` — 快速幂
- `abs(const T&)` — 绝对值（适用于 `BigInt` / `BigFloat`）

## 实现说明

- **表示方式**：以 `2^28` 为基数、小端序 `uint32_t` 数组存储（`DIGIT_BITS = 28`），无前导零。
- **乘法**：使用三模数 NTT，结合 Montgomery 模乘、AVX2 向量化，多线程下分块并行。
- **理论最大值**：受 NTT 最大变换长度限制，可表示整数的理论最大值约为 2^(28·2^26)（每个数 `DIGIT_BITS = 28` 二进制位 × 最大变换长度 2^26，后者由三个 NTT 素数中 2 的幂因子最小值决定）。
- **线程池**：有超时忙等（默认100ms）+ 动态块分配的任务模型，8 线程时 NTT 乘法相对单线程约 3.6~4.3× 加速。

## 已知限制

- 不支持 MSVC 编译。
- `BigInt` 的 `operator/` 仅支持除以 64 位整型；`BigInt` 之间求商余可使用 `divmod`（可指定舍入模式）。
- `BigFloat` 不支持比较运算，建议作差后判断符号。
- 部分功能依赖 AVX2，需在支持的 CPU 上编译运行。
