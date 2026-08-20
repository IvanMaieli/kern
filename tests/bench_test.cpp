#include <kern/ops.hpp>
#include <kern/tensor.hpp>
#include <kern/dtype.hpp>
#include <kern/shape.hpp>
#include "test_macros.hpp"
#include "tests.hpp"
#include <chrono>
#include <cstdio>

void test_bench_matmul() {
    constexpr size_t N = 512;
    const kern::Shape shape{N, N};

    const kern::Tensor t_a(shape, kern::DataType::float32);
    const kern::Tensor t_b(shape, kern::DataType::float32);
    kern::Tensor t_out(shape, kern::DataType::float32);

    // Warm-up (not strictly necessary for naive, but good practice)
    kern::ops::MatMul(t_a, t_b, t_out);

    const auto start = std::chrono::high_resolution_clock::now();
    
    kern::ops::MatMul(t_a, t_b, t_out);

    const auto end = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double, std::milli> duration = end - start;
    
    std::printf("MatMul %zux%zu took: %.3f ms\n", N, N, duration.count());
}

void test_bench_add_broadcast() {
    constexpr size_t R = 512;
    constexpr size_t C = 512;

    kern::Tensor t_a(kern::Shape{R, C}, kern::DataType::float32);
    kern::Tensor t_b(kern::Shape{C}, kern::DataType::float32);
    kern::Tensor t_out(kern::Shape{R, C}, kern::DataType::float32);

    auto* a_ptr = static_cast<float*>(t_a.data());
    auto* b_ptr = static_cast<float*>(t_b.data());
    for (size_t i = 0; i < R * C; ++i) a_ptr[i] = 1.0f;
    for (size_t j = 0; j < C; ++j) b_ptr[j] = 2.0f;

    kern::ops::Add(t_a, t_b, t_out); // warm-up

    const auto start = std::chrono::high_resolution_clock::now();
    kern::ops::Add(t_a, t_b, t_out);
    const auto end = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double, std::milli> duration = end - start;

    std::printf("Broadcast Add %zux%zu + %zu took: %.3f ms\n", R, C, C, duration.count());
}

void test_bench_matmul_transposed() {
    constexpr size_t N = 512;

    kern::Tensor t_a(kern::Shape{N, N}, kern::DataType::float32);
    kern::Tensor t_b_t(kern::Shape{N, N}, kern::DataType::float32);
    kern::Tensor t_out(kern::Shape{N, N}, kern::DataType::float32);

    kern::ops::MatMulTransposed(t_a, t_b_t, t_out); // warm-up

    const auto start = std::chrono::high_resolution_clock::now();
    kern::ops::MatMulTransposed(t_a, t_b_t, t_out);
    const auto end = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double, std::milli> duration = end - start;

    std::printf("MatMulTransposed %zux%zu took: %.3f ms\n", N, N, duration.count());
}

void test_bench_gelu() {
    constexpr size_t R = 512;
    constexpr size_t C = 2048;

    kern::Tensor t_in(kern::Shape{R, C}, kern::DataType::float32);
    kern::Tensor t_out(kern::Shape{R, C}, kern::DataType::float32);

    auto* in_ptr = static_cast<float*>(t_in.data());
    for (size_t i = 0; i < R * C; ++i) in_ptr[i] = 0.001f * static_cast<float>(i % 2000) - 1.0f;

    kern::ops::GELU(t_in, t_out); // warm-up

    const auto start = std::chrono::high_resolution_clock::now();
    constexpr int iters = 20;
    for (int it = 0; it < iters; ++it) kern::ops::GELU(t_in, t_out);
    const auto end = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double, std::milli> duration = end - start;

    std::printf("GELU %zux%zu took: %.3f ms/iter\n", R, C, duration.count() / iters);
}
