#
# Created by Ivan Maieli on 13/08/2026.
#

cmake_minimum_required(VERSION 3.25)

project(
        kern
        VERSION 0.1.0
        DESCRIPTION "Lightweight C++ inference engine for small language models"
        LANGUAGES CXX
)

option(KERN_BUILD_TESTS "Build Kern tests" ON)
add_library(kern
        include/kern/version.hpp
        src/version.cpp
        include/kern/shape.hpp
        src/shape.cpp
        include/kern/dtype.hpp
        include/kern/tensor.hpp
        include/kern/config.hpp
        src/tensor.cpp
        include/kern/memory_pool.hpp
        src/memory_pool.cpp
        include/kern/buffer.hpp
        src/buffer.cpp
        include/kern/ops.hpp
        src/ops.cpp
        include/kern/thread_pool.hpp
        src/thread_pool.cpp
        include/kern/graph.hpp
        src/graph.cpp
)

add_library(kern::kern ALIAS kern)

# FEAT_FP16FML dispatch: clang predefines __ARM_FEATURE_FP16_FML for every
# apple arm64 -mcpu (apple-m1 included), but M1/M2 do not implement it. The
# FMLAL kernels live in the only TU compiled with the feature on
# (kern_fmlal, -mcpu=generic+fp16fml); every other target compiles with it
# off, so fmlal instructions cannot leak past the runtime check in ops.cpp.
# Note: an explicit -target-feature wins over a later -mcpu, so the two flag
# sets must never meet in the same command line. The SHELL: prefix keeps the
# repeated -Xclang token from being de-duplicated.
set(KERN_NO_FP16FML "SHELL:-Xclang -target-feature -Xclang -fp16fml")

add_library(kern_fmlal
        src/ops_fmlal.cpp
        src/matmul_tiling.hpp
)

target_include_directories(
        kern_fmlal
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_compile_features(
        kern_fmlal
        PRIVATE
        cxx_std_20
)

target_compile_options(
        kern_fmlal
        PRIVATE
        -Wall
        -Wextra
        -Wpedantic
        -Wconversion
        -Wshadow
        -mcpu=generic+fp16fml
)

target_link_libraries(
        kern
        PRIVATE
        kern_fmlal
)

target_include_directories(
        kern
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_features(
        kern
        PUBLIC
        cxx_std_20
)

target_compile_options(
        kern
        PRIVATE
        -Wall
        -Wextra
        -Wpedantic
        -Wconversion
        -Wshadow
        ${KERN_NO_FP16FML}
)

add_executable(kern-run
        app/kern_run.cpp
)

target_link_libraries(
        kern-run
        PRIVATE
        kern::kern
)

if(KERN_BUILD_TESTS)
    enable_testing()

    add_executable(kern-tests
            tests/main.cpp
            tests/tests.hpp
            tests/test_macros.hpp
            tests/version_test.cpp
            tests/shape_test.cpp
            tests/dtype_test.cpp
            tests/tensor_test.cpp
            tests/memory_pool_test.cpp
            tests/ops_test.cpp
            tests/fp16_test.cpp
            tests/graph_test.cpp
            tests/bench_test.cpp
    )
    target_link_libraries(
            kern-tests
            PRIVATE
            kern::kern
    )
    target_compile_options(
            kern-tests
            PRIVATE
            ${KERN_NO_FP16FML}
    )
    add_test(
            NAME kern.tests
            COMMAND kern-tests
    )
endif()
