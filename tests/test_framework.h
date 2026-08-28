// 极简测试框架（零依赖）：断言宏 + 用例/失败计数 + 汇总。
//
// 用法：
//   TEST_CHECK(cond, "描述")     —— 断言，失败时打印定位与描述
//   bigint_test::summary("名字") —— 汇总输出 cases/fails，返回 0/1（供 main 返回）
//
// 测试风格为"确定性随机对拍 + 已知值 + 边界用例"，无需 xUnit 级功能。
#pragma once

#include <cstddef>
#include <iostream>


namespace bigint_test {

inline auto cases() -> std::size_t& {
    static std::size_t n = 0;
    return n;
}

inline auto fails() -> std::size_t& {
    static std::size_t n = 0;
    return n;
}

inline void check(bool cond, const char* expr, const char* file, int line, const char* msg) {
    ++cases();
    if (!cond) {
        ++fails();
        std::cout << "FAIL: " << msg << " (" << file << ':' << line << "): " << expr << '\n';
    }
}

// 汇总输出，返回 0 表示全部通过（供 main 返回）
inline auto summary(const char* name) -> int {
    std::cout << name << ": cases=" << cases() << " fails=" << fails() << '\n';
    return fails() == 0 ? 0 : 1;
}

}  // namespace bigint_test

#define TEST_CHECK(cond, msg) \
    bigint_test::check(static_cast<bool>(cond), #cond, __FILE__, __LINE__, msg)
