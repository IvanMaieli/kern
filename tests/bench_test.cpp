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
