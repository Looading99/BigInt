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

# 构建时使用源码树中的 include，安装后使用安装前缀下的 include。
target_include_directories(bigint PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_compile_options(
    bigint
    PRIVATE
        $<$<CONFIG:Debug>:-mavx -mavx2 -mfma -Wall -Wextra -Wpedantic -Werror
            -Wshadow -Wnon-virtual-dtor -Wold-style-cast -Wcast-align
            -Woverloaded-virtual -Wnull-dereference -Wformat=2 
            -Wmisleading-indentation>
        $<$<CONFIG:Release>:-mavx -mavx2 -mfma -flto>
        $<$<CONFIG:RelWithDebInfo>:-mavx -mavx2 -mfma -flto>
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
    add_executable(test_fft_digit_bits tests/measure_digit_bits.cpp)
    target_link_libraries(test_fft_digit_bits PRIVATE bigint::bigint)
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
