#include <kern/ops.hpp>
#include <kern/tensor.hpp>
#include <kern/dtype.hpp>
#include <kern/shape.hpp>
#include "tests.hpp"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <vector>

// Distribution instead of a plain mean: multicore kernels jitter with
// scheduler and frequency state, so the median/p99/min split tells whether
// the code is slow or the machine was noisy.
template <typename F>
static void bench_dist(const char* label, int iters, F&& fn) {
    for (int w = 0; w < 5; ++w) fn(); // warm-up: clocks, caches, QoS
    std::vector<double> ms(static_cast<size_t>(iters));
    for (int i = 0; i < iters; ++i) {
        const auto start = std::chrono::high_resolution_clock::now();
        fn();
        const auto end = std::chrono::high_resolution_clock::now();
        ms[static_cast<size_t>(i)] =
            std::chrono::duration<double, std::milli>(end - start).count();
    }
    std::sort(ms.begin(), ms.end());
    // Nearest-rank p99: floor(0.99 * n). With the sample counts used here
    // (20-30) that is the maximum, which is honest — a true tail estimate
    // needs more samples than the distribution has.
    const size_t p99i = ms.size() * 99 / 100;
    std::printf("%s: median %.3f ms, p99 %.3f ms, min %.3f ms\n",
                label, ms[ms.size() / 2], ms[p99i], ms.front());
}

void test_bench_matmul() {
    constexpr size_t N = 512;
    const kern::Shape shape{N, N};

    const kern::Tensor t_a(shape, kern::DataType::float32);
    const kern::Tensor t_b(shape, kern::DataType::float32);
    kern::Tensor t_out(shape, kern::DataType::float32);

    bench_dist("MatMul 512x512", 30, [&] { kern::ops::MatMul(t_a, t_b, t_out); });
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

    bench_dist("MatMulTransposed 512x512", 30,
               [&] { kern::ops::MatMulTransposed(t_a, t_b_t, t_out); });
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

void test_bench_matvec() {
    // Decode-path GEMV: many output features, one (or few) token rows.
    constexpr size_t M = 4096;
    constexpr size_t K = 4096;

    kern::Tensor t_a(kern::Shape{M, K}, kern::DataType::float32);
    kern::Tensor t_v(kern::Shape{K}, kern::DataType::float32);
    kern::Tensor t_out(kern::Shape{M}, kern::DataType::float32);

    bench_dist("MatVec [4096x4096].[4096]", 20, [&] { kern::ops::MatVec(t_a, t_v, t_out); });
}

void test_bench_matvec_f16() {
    constexpr size_t M = 4096;
    constexpr size_t K = 4096;

    kern::Tensor t_a(kern::Shape{M, K}, kern::DataType::float16);
    kern::Tensor t_v(kern::Shape{K}, kern::DataType::float16);
    kern::Tensor t_out(kern::Shape{M}, kern::DataType::float16);

    auto* a_ptr = static_cast<__fp16*>(t_a.data());
    auto* v_ptr = static_cast<__fp16*>(t_v.data());
    for (size_t i = 0; i < M * K; ++i) a_ptr[i] = static_cast<__fp16>(0.001f * static_cast<float>(i % 512) - 0.25f);
    for (size_t i = 0; i < K; ++i) v_ptr[i] = static_cast<__fp16>(0.001f * static_cast<float>(i % 512) - 0.25f);

    bench_dist("MatVec f16 [4096x4096].[4096]", 20, [&] { kern::ops::MatVec(t_a, t_v, t_out); });
}

void test_bench_matmultransposed_f16() {
    constexpr size_t N = 512;

    kern::Tensor t_a(kern::Shape{N, N}, kern::DataType::float16);
    kern::Tensor t_b_t(kern::Shape{N, N}, kern::DataType::float16);
    kern::Tensor t_out(kern::Shape{N, N}, kern::DataType::float16);

    auto* a_ptr = static_cast<__fp16*>(t_a.data());
    auto* b_ptr = static_cast<__fp16*>(t_b_t.data());
    for (size_t i = 0; i < N * N; ++i) {
        a_ptr[i] = static_cast<__fp16>(0.001f * static_cast<float>(i % 512) - 0.25f);
        b_ptr[i] = static_cast<__fp16>(0.001f * static_cast<float>((i * 7) % 512) - 0.25f);
    }

    bench_dist("MatMulTransposed f16 512x512", 30,
               [&] { kern::ops::MatMulTransposed(t_a, t_b_t, t_out); });
}
