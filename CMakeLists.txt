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

    add_executable(kern-version-test
            tests/version_test.cpp
    )

    target_link_libraries(
            kern-version-test
            PRIVATE
            kern::kern
    )

    add_test(
            NAME kern.version
            COMMAND kern-version-test
    )

endif()
