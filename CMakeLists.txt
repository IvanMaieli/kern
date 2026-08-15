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
        src/tensor.cpp
        include/kern/memory_pool.hpp
        src/memory_pool.cpp
)

add_library(kern::kern ALIAS kern)

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
    )
    target_link_libraries(
            kern-tests
            PRIVATE
            kern::kern
    )
    add_test(
            NAME kern.tests
            COMMAND kern-tests
    )
endif()
