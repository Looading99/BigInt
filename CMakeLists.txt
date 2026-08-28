cmake_minimum_required(VERSION 3.15)

project(BigInt VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(BUILD_EXAMPLES "Build the example/benchmark programs" ON)
option(BUILD_TESTS "Build test programs" ON)

# ---- 库目标 ----
add_library(bigint STATIC
    src/bigint.cpp
    src/mul/fft.cpp
    src/mul/ntt_multithread.cpp
    src/mul/ssa.cpp
)
add_library(bigint::bigint ALIAS bigint)

# clang + MSVC target（如 LLVM 官方 Windows 二进制默认 x86_64-pc-windows-msvc）：
# NMake 生成器以 -nostdlib 链接、不自动链接 compiler-rt builtins，导致 __uint128_t
# 除法/取模的运行时函数（__udivti3/__umodti3 等）未定义。这里显式链接 builtins 库。
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND CMAKE_CXX_SIMULATE_ID STREQUAL "MSVC")
    get_filename_component(_llvm_bin "${CMAKE_CXX_COMPILER}" DIRECTORY)
    get_filename_component(_llvm_root "${_llvm_bin}" DIRECTORY)
    file(GLOB _clang_rt_builtins
        "${_llvm_root}/lib/clang/*/lib/windows/clang_rt.builtins-*.lib")
    if(_clang_rt_builtins)
        target_link_libraries(bigint PUBLIC "${_clang_rt_builtins}")
        message(STATUS "Linking clang compiler-rt builtins: ${_clang_rt_builtins}")
    else()
        message(WARNING
            "clang compiler-rt builtins not found (needed for __udivti3 etc.)")
    endif()
endif()

# 构建时使用源码树中的 include，安装后使用安装前缀下的 include。
target_include_directories(bigint PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_compile_options(
    bigint
    PRIVATE
        -mavx -mavx2 -mfma
        $<$<CONFIG:Debug>:-Wall -Wextra -Wpedantic
            -Wshadow -Wnon-virtual-dtor -Wold-style-cast -Wcast-align
            -Woverloaded-virtual -Wnull-dereference -Wformat=2 
            -Wmisleading-indentation>
        # -Werror 仅对 GNU 风格前端（g++/clang++）生效。clang-cl（MSVC 兼容模式）
        # 默认开启的警告集与 clang++ 差异很大（-Wshorten-64-to-32、-Wsign-conversion、
        # -Wc++98-compat*、-Wcomma、-Wunsafe-buffer-usage 等），-Werror 下无法编译，
        # 暂不启用，待后续逐步修掉这些警告后再恢复。
        $<$<AND:$<CONFIG:Debug>,$<NOT:$<CXX_COMPILER_FRONTEND_VARIANT:MSVC>>>:-Werror>
)

# ---- 示例/基准程序 ----
if(BUILD_EXAMPLES)
    add_executable(bigint_benchmark examples/benchmark.cpp)
    target_link_libraries(bigint_benchmark PRIVATE bigint::bigint)
    add_executable(bst examples/binary_split.cpp)
    target_link_libraries(bst PRIVATE bigint::bigint)
    add_executable(pi examples/pi.cpp)
    target_link_libraries(pi PRIVATE bigint::bigint)
endif()

# ---- 测试程序 ----
if(BUILD_TESTS)
    enable_testing()
    add_executable(test_bigint tests/test_bigint.cpp)
    target_link_libraries(test_bigint PRIVATE bigint::bigint)
    add_test(NAME bigint_core COMMAND $<TARGET_FILE:test_bigint>)

    add_executable(test_fft_digit_bits tests/measure_digit_bits.cpp)
    target_link_libraries(test_fft_digit_bits PRIVATE bigint::bigint)
    add_test(NAME bigint_mul_regress COMMAND $<TARGET_FILE:test_fft_digit_bits> regress)
endif()

# ---- 安装与导出（find_package(bigint CONFIG) 支持） ----
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

install(TARGETS bigint
    EXPORT bigintTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(DIRECTORY include/bigint
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(EXPORT bigintTargets
    FILE bigintTargets.cmake
    NAMESPACE bigint::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/bigint
)

configure_package_config_file(
    cmake/bigintConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/bigintConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/bigint
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/bigintConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/bigintConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/bigintConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/bigint
)
