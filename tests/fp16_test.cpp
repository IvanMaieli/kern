#include <kern/ops.hpp>
#include <kern/tensor.hpp>
#include <kern/dtype.hpp>
#include <kern/shape.hpp>
#include "test_macros.hpp"
#include "tests.hpp"
#include <cmath>
#include <stdexcept>

// float16 kernels convert to float32 lanes for the arithmetic and convert
// back on store, so the result of a kernel equals the float32 kernel applied
// to rounded inputs. Integer-valued fixtures stay exact in fp16; float
// fixtures use loose tolerances matching fp16 storage granularity (~1e-3).

void test_ops_matvec() {
    // K = 5 covers the NEON body + scalar tail; M = 3 rows.
    kern::Tensor t_a(kern::Shape{3, 5}, kern::DataType::float32);
    kern::Tensor t_v(kern::Shape{5}, kern::DataType::float32);
    kern::Tensor t_out(kern::Shape{3}, kern::DataType::float32);

    auto* a_ptr = static_cast<float*>(t_a.data());
    auto* v_ptr = static_cast<float*>(t_v.data());
    for (std::size_t i = 0; i < 15; ++i) a_ptr[i] = static_cast<float>(i) + 1.0f;
    for (std::size_t i = 0; i < 5; ++i) v_ptr[i] = static_cast<float>(i) * 0.5f;

    kern::ops::MatVec(t_a, t_v, t_out);

    const auto* out_ptr = static_cast<const float*>(t_out.data());
    for (std::size_t m = 0; m < 3; ++m) {
        float expected = 0.0f;
        for (std::size_t k = 0; k < 5; ++k)
            expected += (static_cast<float>(m * 5 + k) + 1.0f) * (static_cast<float>(k) * 0.5f);
        CHECK(std::abs(out_ptr[m] - expected) < 1e-5f);
    }
}

void test_ops_matvec_decode_shape() {
    // M = 1: exercises the K-split path on a large K (memory-bound decode).
    constexpr std::size_t K = 4096;
    kern::Tensor t_a(kern::Shape{1, K}, kern::DataType::float32);
    kern::Tensor t_v(kern::Shape{K}, kern::DataType::float32);
    kern::Tensor t_out(kern::Shape{1}, kern::DataType::float32);

    auto* a_ptr = static_cast<float*>(t_a.data());
    auto* v_ptr = static_cast<float*>(t_v.data());
    for (std::size_t i = 0; i < K; ++i) {
        a_ptr[i] = 0.001f * static_cast<float>(i % 512) - 0.25f;
        v_ptr[i] = 0.001f * static_cast<float>((i * 7) % 512) - 0.25f;
    }

    kern::ops::MatVec(t_a, t_v, t_out);

    double expected = 0.0;
    for (std::size_t i = 0; i < K; ++i) expected += static_cast<double>(a_ptr[i]) * v_ptr[i];
    // Accumulation order differs between the split chunks and the reference;
    // allow a small relative tolerance.
    CHECK(std::abs(static_cast<double>(static_cast<const float*>(t_out.data())[0]) - expected)
          < 1e-3 * std::abs(expected) + 1e-3);
}

void test_fp16_add() {
    const kern::Shape shape{2, 2};
    kern::Tensor t1(shape, kern::DataType::float16);
    kern::Tensor t2(shape, kern::DataType::float16);
    kern::Tensor out(shape, kern::DataType::float16);

    auto* p1 = static_cast<__fp16*>(t1.data());
    auto* p2 = static_cast<__fp16*>(t2.data());
    p1[0] = __fp16(1.0); p1[1] = __fp16(2.0); p1[2] = __fp16(3.0); p1[3] = __fp16(4.0);
    p2[0] = __fp16(5.0); p2[1] = __fp16(6.0); p2[2] = __fp16(7.0); p2[3] = __fp16(8.0);

    kern::ops::Add(t1, t2, out);

    const auto* o = static_cast<const __fp16*>(out.data());
    CHECK(static_cast<float>(o[0]) == 6.0f);
    CHECK(static_cast<float>(o[1]) == 8.0f);
    CHECK(static_cast<float>(o[2]) == 10.0f);
    CHECK(static_cast<float>(o[3]) == 12.0f);
}

void test_fp16_relu() {
    // 10 elements: NEON body (8) + scalar tail (2).
    kern::Tensor t_in(kern::Shape{10}, kern::DataType::float16);
    kern::Tensor t_out(kern::Shape{10}, kern::DataType::float16);

    auto* in_ptr = static_cast<__fp16*>(t_in.data());
    const float values[10] = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f, -4.0f, 4.0f, -5.0f, 5.0f};
    for (std::size_t i = 0; i < 10; ++i) in_ptr[i] = static_cast<__fp16>(values[i]);

    kern::ops::ReLU(t_in, t_out);

    const auto* out_ptr = static_cast<const __fp16*>(t_out.data());
    const float expected[10] = {0.0f, 0.0f, 0.0f, 1.0f, 2.0f, 3.0f, 0.0f, 4.0f, 0.0f, 5.0f};
    for (std::size_t i = 0; i < 10; ++i)
        CHECK(static_cast<float>(out_ptr[i]) == expected[i]);
}

void test_fp16_gelu() {
    const std::size_t n = 33;
    kern::Tensor t_in(kern::Shape{n}, kern::DataType::float16);
    kern::Tensor t_out(kern::Shape{n}, kern::DataType::float16);

    auto* in_ptr = static_cast<__fp16*>(t_in.data());
    for (std::size_t i = 0; i < n; ++i)
        in_ptr[i] = static_cast<__fp16>(-4.0f + 0.25f * static_cast<float>(i));

    kern::ops::GELU(t_in, t_out);

    const auto* out_ptr = static_cast<const __fp16*>(t_out.data());
    const float k1 = 0.7978845608f;
    const float k2 = 0.044715f;
    for (std::size_t i = 0; i < n; ++i) {
        const float x = static_cast<float>(in_ptr[i]);
        const float expected = 0.5f * x * (1.0f + std::tanh(k1 * (x + k2 * x * x * x)));
        // fp16 storage granularity: ulp at |y|~4 is about 2e-3.
        CHECK(std::abs(static_cast<float>(out_ptr[i]) - expected) < 1e-2f);
    }
}

void test_fp16_matmul() {
    // Integer fixture: exactly representable in fp16 (all values < 2048).
    kern::Tensor t_a(kern::Shape{2, 3}, kern::DataType::float16);
    kern::Tensor t_b(kern::Shape{3, 2}, kern::DataType::float16);
    kern::Tensor t_out(kern::Shape{2, 2}, kern::DataType::float16);

    auto* a_ptr = static_cast<__fp16*>(t_a.data());
    auto* b_ptr = static_cast<__fp16*>(t_b.data());
    const float a_vals[6] = {1, 2, 3, 4, 5, 6};
    const float b_vals[6] = {7, 8, 9, 10, 11, 12};
    for (std::size_t i = 0; i < 6; ++i) {
        a_ptr[i] = static_cast<__fp16>(a_vals[i]);
        b_ptr[i] = static_cast<__fp16>(b_vals[i]);
    }

    kern::ops::MatMul(t_a, t_b, t_out);

    const auto* out_ptr = static_cast<const __fp16*>(t_out.data());
    CHECK(static_cast<float>(out_ptr[0]) == 58.0f);
    CHECK(static_cast<float>(out_ptr[1]) == 64.0f);
    CHECK(static_cast<float>(out_ptr[2]) == 139.0f);
    CHECK(static_cast<float>(out_ptr[3]) == 154.0f);
}

void test_fp16_matmul_zero_k() {
    // Same contract as the float32 kernel: K == 0 writes zeros (empty sum)
    // instead of leaving the stale buffer behind.
    kern::Tensor t_a(kern::Shape{2, 0}, kern::DataType::float16);
    kern::Tensor t_b(kern::Shape{0, 3}, kern::DataType::float16);
    kern::Tensor t_out(kern::Shape{2, 3}, kern::DataType::float16);

    auto* out_ptr = static_cast<__fp16*>(t_out.data());
    for (std::size_t i = 0; i < 6; ++i) out_ptr[i] = static_cast<__fp16>(42.0f);

    kern::ops::MatMul(t_a, t_b, t_out);

    for (std::size_t i = 0; i < 6; ++i) CHECK(static_cast<float>(out_ptr[i]) == 0.0f);
}

void test_fp16_matmul_transposed() {
    kern::Tensor t_a(kern::Shape{2, 3}, kern::DataType::float16);
    kern::Tensor t_b_t(kern::Shape{2, 3}, kern::DataType::float16);
    kern::Tensor t_out(kern::Shape{2, 2}, kern::DataType::float16);

    auto* a_ptr = static_cast<__fp16*>(t_a.data());
    auto* bt_ptr = static_cast<__fp16*>(t_b_t.data());
    const float a_vals[6] = {1, 2, 3, 4, 5, 6};
    const float bt_vals[6] = {7, 9, 11, 8, 10, 12};
    for (std::size_t i = 0; i < 6; ++i) {
        a_ptr[i] = static_cast<__fp16>(a_vals[i]);
        bt_ptr[i] = static_cast<__fp16>(bt_vals[i]);
    }

    kern::ops::MatMulTransposed(t_a, t_b_t, t_out);

    const auto* out_ptr = static_cast<const __fp16*>(t_out.data());
    CHECK(static_cast<float>(out_ptr[0]) == 58.0f);
    CHECK(static_cast<float>(out_ptr[1]) == 64.0f);
    CHECK(static_cast<float>(out_ptr[2]) == 139.0f);
    CHECK(static_cast<float>(out_ptr[3]) == 154.0f);
}

void test_fp16_matvec() {
    kern::Tensor t_a(kern::Shape{3, 5}, kern::DataType::float16);
    kern::Tensor t_v(kern::Shape{5}, kern::DataType::float16);
    kern::Tensor t_out(kern::Shape{3}, kern::DataType::float16);

    auto* a_ptr = static_cast<__fp16*>(t_a.data());
    auto* v_ptr = static_cast<__fp16*>(t_v.data());
    for (std::size_t i = 0; i < 15; ++i) a_ptr[i] = static_cast<__fp16>(static_cast<float>(i) + 1.0f);
    for (std::size_t i = 0; i < 5; ++i) v_ptr[i] = static_cast<__fp16>(static_cast<float>(i) * 0.5f);

    kern::ops::MatVec(t_a, t_v, t_out);

    const auto* out_ptr = static_cast<const __fp16*>(t_out.data());
    for (std::size_t m = 0; m < 3; ++m) {
        float expected = 0.0f;
        for (std::size_t k = 0; k < 5; ++k)
            expected += (static_cast<float>(m * 5 + k) + 1.0f) * (static_cast<float>(k) * 0.5f);
        CHECK(std::abs(static_cast<float>(out_ptr[m]) - expected) < 1e-2f);
    }
}

void test_fp16_softmax() {
    kern::Tensor t_in(kern::Shape{1, 3}, kern::DataType::float16);
    kern::Tensor t_out(kern::Shape{1, 3}, kern::DataType::float16);

    auto* in_ptr = static_cast<__fp16*>(t_in.data());
    in_ptr[0] = __fp16(1.0); in_ptr[1] = __fp16(2.0); in_ptr[2] = __fp16(3.0);

    kern::ops::Softmax(t_in, t_out);

    const auto* out_ptr = static_cast<const __fp16*>(t_out.data());
    CHECK(static_cast<float>(out_ptr[0]) > 0.0f && static_cast<float>(out_ptr[0]) < 0.1f);
    CHECK(static_cast<float>(out_ptr[1]) > 0.2f && static_cast<float>(out_ptr[1]) < 0.3f);
    CHECK(static_cast<float>(out_ptr[2]) > 0.6f && static_cast<float>(out_ptr[2]) < 0.7f);
}

void test_fp16_layernorm() {
    kern::Tensor t_in(kern::Shape{1, 4}, kern::DataType::float16);
    kern::Tensor t_out(kern::Shape{1, 4}, kern::DataType::float16);

    auto* in_ptr = static_cast<__fp16*>(t_in.data());
    in_ptr[0] = __fp16(1.0); in_ptr[1] = __fp16(2.0);
    in_ptr[2] = __fp16(3.0); in_ptr[3] = __fp16(4.0);

    kern::ops::LayerNorm(t_in, t_out);

    const auto* out_ptr = static_cast<const __fp16*>(t_out.data());
    CHECK(std::abs(static_cast<float>(out_ptr[0]) - (-1.34164f)) < 5e-3f);
}

void test_fp16_int8_unsupported() {
    const kern::Shape shape{2, 2};
    kern::Tensor t1(shape, kern::DataType::int8);
    kern::Tensor t2(shape, kern::DataType::int8);
    kern::Tensor out(shape, kern::DataType::int8);

    bool caught = false;
    try { kern::ops::Add(t1, t2, out); } catch (const std::runtime_error&) { caught = true; }
    CHECK(caught);

    caught = false;
    kern::Tensor m_out(kern::Shape{2, 2}, kern::DataType::int8);
    try { kern::ops::MatMul(t1, t2, m_out); } catch (const std::runtime_error&) { caught = true; }
    CHECK(caught);
}
