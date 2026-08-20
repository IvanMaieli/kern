#include <kern/ops.hpp>
#include <kern/tensor.hpp>
#include <kern/dtype.hpp>
#include <kern/shape.hpp>
#include "test_macros.hpp"
#include "tests.hpp"
#include <vector>
#include <stdexcept>
#include <limits>
#include <cmath>

void test_ops_add() {
    const kern::Shape shape{2, 2};
    kern::Tensor t1(shape, kern::DataType::float32);
    kern::Tensor t2(shape, kern::DataType::float32);
    kern::Tensor out(shape, kern::DataType::float32);

    const auto t1_data = static_cast<float*>(t1.data());
    const auto t2_data = static_cast<float*>(t2.data());
    
    t1_data[0] = 1.0f; t1_data[1] = 2.0f; t1_data[2] = 3.0f; t1_data[3] = 4.0f;
    t2_data[0] = 5.0f; t2_data[1] = 6.0f; t2_data[2] = 7.0f; t2_data[3] = 8.0f;

    kern::ops::Add(t1, t2, out);

    const auto out_data = static_cast<float*>(out.data());
    CHECK(out_data[0] == 6.0f);
    CHECK(out_data[1] == 8.0f);
    CHECK(out_data[2] == 10.0f);
    CHECK(out_data[3] == 12.0f);
}

void test_ops_add_mismatched_shape() {
    const kern::Tensor t1(kern::Shape{2, 2}, kern::DataType::float32);
    const kern::Tensor t2(kern::Shape{1, 4}, kern::DataType::float32);
    kern::Tensor out(kern::Shape{2, 2}, kern::DataType::float32);

    bool caught = false;
    try { kern::ops::Add(t1, t2, out); } catch (const std::invalid_argument&) { caught = true; }
    CHECK(caught);
}

void test_ops_reshape() {
    const kern::Shape shape_in{4};
    const kern::Shape shape_out{2, 2};
    kern::Tensor t_in(shape_in, kern::DataType::float32);
    kern::Tensor t_out(shape_out, kern::DataType::float32);

    auto* in_data = static_cast<float*>(t_in.data());
    in_data[0] = 1.0f; in_data[1] = 2.0f; in_data[2] = 3.0f; in_data[3] = 4.0f;

    kern::ops::Reshape(t_in, shape_out, t_out);

    const auto* out_data = static_cast<float*>(t_out.data());
    CHECK(out_data[0] == 1.0f);
    CHECK(out_data[1] == 2.0f);
    CHECK(out_data[2] == 3.0f);
    CHECK(out_data[3] == 4.0f);
}

void test_ops_permute() {
    const kern::Shape shape_in{2, 3, 4};
    const kern::Shape shape_out{4, 2, 3};
    const kern::Tensor t_in(shape_in, kern::DataType::float32);
    kern::Tensor t_out(shape_out, kern::DataType::float32);

    kern::ops::Permute(t_in, {2, 0, 1}, t_out);

    CHECK(t_out.shape() == shape_out);
    CHECK(t_out.buffer() == t_in.buffer());
}

void test_ops_permute_data() {
    const kern::Shape shape_in{2, 3, 4};
    kern::Tensor t_in(shape_in, kern::DataType::float32);
    kern::Tensor t_out(kern::Shape{4, 2, 3}, kern::DataType::float32);

    auto* in_ptr = static_cast<float*>(t_in.data());
    for (std::size_t i = 0; i < 24; ++i) in_ptr[i] = static_cast<float>(i);

    kern::ops::Permute(t_in, {2, 0, 1}, t_out);

    // {2,3,4} has strides {12,4,1}; order {2,0,1} must produce strides
    // {1,12,4}, so the view maps (c0,c1,c2) to source index c0 + 12*c1 + 4*c2.
    // (Recomputing contiguous strides {6,3,1} reads the wrong elements.)
    const auto* out_ptr = static_cast<const float*>(t_out.data());
    for (std::size_t c0 = 0; c0 < 4; ++c0)
        for (std::size_t c1 = 0; c1 < 2; ++c1)
            for (std::size_t c2 = 0; c2 < 3; ++c2)
                CHECK(out_ptr[t_out.shape().linear_index({c0, c1, c2})] ==
                      in_ptr[c0 + 12 * c1 + 4 * c2]);
}

void test_ops_transpose_data() {
    kern::Tensor t_in(kern::Shape{2, 3}, kern::DataType::float32);
    kern::Tensor t_out(kern::Shape{3, 2}, kern::DataType::float32);

    auto* in_ptr = static_cast<float*>(t_in.data());
    for (std::size_t i = 0; i < 6; ++i) in_ptr[i] = static_cast<float>(i);

    kern::ops::Transpose(t_in, 0, 1, t_out);

    // out[i][j] must read in[j][i]: linear index i + 3*j, not 2*i + j.
    const auto* out_ptr = static_cast<const float*>(t_out.data());
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 2; ++j)
            CHECK(out_ptr[t_out.shape().linear_index({i, j})] == in_ptr[j * 3 + i]);
}

void test_ops_broadcast_add() {
    const kern::Shape shape_a{2, 3};
    const kern::Shape shape_b{1, 3};
    const kern::Shape shape_out{2, 3};
    
    kern::Tensor t_a(shape_a, kern::DataType::float32);
    kern::Tensor t_b(shape_b, kern::DataType::float32);
    kern::Tensor t_out(shape_out, kern::DataType::float32);

    auto* a_ptr = static_cast<float*>(t_a.data());
    auto* b_ptr = static_cast<float*>(t_b.data());
    
    a_ptr[0] = 1.0f; a_ptr[1] = 2.0f; a_ptr[2] = 3.0f;
    a_ptr[3] = 4.0f; a_ptr[4] = 5.0f; a_ptr[5] = 6.0f;
    b_ptr[0] = 10.0f; b_ptr[1] = 20.0f; b_ptr[2] = 30.0f;

    kern::ops::Add(t_a, t_b, t_out);

    const auto* out_ptr = static_cast<float*>(t_out.data());
    CHECK(out_ptr[0] == 11.0f);
    CHECK(out_ptr[1] == 22.0f);
    CHECK(out_ptr[2] == 33.0f);
    CHECK(out_ptr[3] == 14.0f);
    CHECK(out_ptr[4] == 25.0f);
    CHECK(out_ptr[5] == 36.0f);
}

void test_ops_broadcast_add_rank_mismatch() {
    // [2,3] + [3]: the rank-1 operand right-aligns against the last axis.
    kern::Tensor t_a(kern::Shape{2, 3}, kern::DataType::float32);
    kern::Tensor t_b(kern::Shape{3}, kern::DataType::float32);
    kern::Tensor t_out(kern::Shape{2, 3}, kern::DataType::float32);

    auto* a_ptr = static_cast<float*>(t_a.data());
    auto* b_ptr = static_cast<float*>(t_b.data());
    a_ptr[0] = 1.0f; a_ptr[1] = 2.0f; a_ptr[2] = 3.0f;
    a_ptr[3] = 4.0f; a_ptr[4] = 5.0f; a_ptr[5] = 6.0f;
    b_ptr[0] = 10.0f; b_ptr[1] = 20.0f; b_ptr[2] = 30.0f;

    kern::ops::Add(t_a, t_b, t_out);

    const auto* out_ptr = static_cast<const float*>(t_out.data());
    CHECK(out_ptr[0] == 11.0f);
    CHECK(out_ptr[1] == 22.0f);
    CHECK(out_ptr[2] == 33.0f);
    CHECK(out_ptr[3] == 14.0f);
    CHECK(out_ptr[4] == 25.0f);
    CHECK(out_ptr[5] == 36.0f);
}

void test_ops_broadcast_add_scalar() {
    // A rank-0 operand broadcasts to every element through a zero stride.
    kern::Tensor t_a(kern::Shape{2, 2}, kern::DataType::float32);
    kern::Tensor t_b(kern::Shape{}, kern::DataType::float32);
    kern::Tensor t_out(kern::Shape{2, 2}, kern::DataType::float32);

    auto* a_ptr = static_cast<float*>(t_a.data());
    static_cast<float*>(t_b.data())[0] = 10.0f;
    a_ptr[0] = 1.0f; a_ptr[1] = 2.0f; a_ptr[2] = 3.0f; a_ptr[3] = 4.0f;

    kern::ops::Add(t_a, t_b, t_out);

    const auto* out_ptr = static_cast<const float*>(t_out.data());
    CHECK(out_ptr[0] == 11.0f);
    CHECK(out_ptr[1] == 12.0f);
    CHECK(out_ptr[2] == 13.0f);
    CHECK(out_ptr[3] == 14.0f);
}

void test_ops_add_inplace() {
    // out shares a's buffer: same shape, contiguous, element-wise safe.
    kern::Tensor t_a(kern::Shape{2, 2}, kern::DataType::float32);
    kern::Tensor t_b(kern::Shape{2, 2}, kern::DataType::float32);
    kern::Tensor t_out = t_a;

    auto* a_ptr = static_cast<float*>(t_a.data());
    auto* b_ptr = static_cast<float*>(t_b.data());
    a_ptr[0] = 1.0f; a_ptr[1] = 2.0f; a_ptr[2] = 3.0f; a_ptr[3] = 4.0f;
    b_ptr[0] = 10.0f; b_ptr[1] = 20.0f; b_ptr[2] = 30.0f; b_ptr[3] = 40.0f;

    kern::ops::Add(t_a, t_b, t_out);

    const auto* out_ptr = static_cast<const float*>(t_out.data());
    CHECK(out_ptr[0] == 11.0f);
    CHECK(out_ptr[1] == 22.0f);
    CHECK(out_ptr[2] == 33.0f);
    CHECK(out_ptr[3] == 44.0f);
}

void test_ops_add_inplace_broadcast_rejected() {
    // out aliases a broadcast operand: writing output rows would clobber a
    // before it is read, so Add must refuse.
    kern::Tensor t_a(kern::Shape{1, 3}, kern::DataType::float32);
    kern::Tensor t_b(kern::Shape{2, 3}, kern::DataType::float32);
    kern::Tensor t_out(kern::Shape{2, 3}, kern::DataType::float32, t_a.buffer());

    bool caught = false;
    try { kern::ops::Add(t_a, t_b, t_out); } catch (const std::invalid_argument&) { caught = true; }
    CHECK(caught);
}

void test_ops_matmul() {
    const kern::Shape shape_a{2, 3};
    const kern::Shape shape_b{3, 2};
    const kern::Shape shape_out{2, 2};
    
    kern::Tensor t_a(shape_a, kern::DataType::float32);
    kern::Tensor t_b(shape_b, kern::DataType::float32);
    kern::Tensor t_out(shape_out, kern::DataType::float32);

    auto* a_ptr = static_cast<float*>(t_a.data());
    auto* b_ptr = static_cast<float*>(t_b.data());
    
    // Mat A [2x3]
    a_ptr[0] = 1.0f; a_ptr[1] = 2.0f; a_ptr[2] = 3.0f;
    a_ptr[3] = 4.0f; a_ptr[4] = 5.0f; a_ptr[5] = 6.0f;
    
    // Mat B [3x2]
    b_ptr[0] = 7.0f;  b_ptr[1] = 8.0f;
    b_ptr[2] = 9.0f;  b_ptr[3] = 10.0f;
    b_ptr[4] = 11.0f; b_ptr[5] = 12.0f;

    kern::ops::MatMul(t_a, t_b, t_out);

    const auto* out_ptr = static_cast<float*>(t_out.data());
    // Row 0: 1*7+2*9+3*11  = 7+18+33  = 58
    // Row 0: 1*8+2*10+3*12 = 8+20+36  = 64
    // Row 1: 4*7+5*9+6*11  = 28+45+66 = 139
    // Row 1: 4*8+5*10+6*12 = 32+50+72 = 154

    CHECK(out_ptr[0] == 58.0f);
    CHECK(out_ptr[1] == 64.0f);
    CHECK(out_ptr[2] == 139.0f);
    CHECK(out_ptr[3] == 154.0f);
}

void test_ops_matmul_output_reuse() {
    kern::Tensor t_a(kern::Shape{2, 3}, kern::DataType::float32);
    kern::Tensor t_b(kern::Shape{3, 2}, kern::DataType::float32);
    kern::Tensor t_out(kern::Shape{2, 2}, kern::DataType::float32);

    auto* a_ptr = static_cast<float*>(t_a.data());
    auto* b_ptr = static_cast<float*>(t_b.data());
    a_ptr[0] = 1.0f; a_ptr[1] = 2.0f; a_ptr[2] = 3.0f;
    a_ptr[3] = 4.0f; a_ptr[4] = 5.0f; a_ptr[5] = 6.0f;
    b_ptr[0] = 7.0f;  b_ptr[1] = 8.0f;
    b_ptr[2] = 9.0f;  b_ptr[3] = 10.0f;
    b_ptr[4] = 11.0f; b_ptr[5] = 12.0f;

    auto* out_ptr = static_cast<float*>(t_out.data());
    // Poison the output: a kernel that accumulates into out without zeroing
    // first folds the NaNs (or any stale result) into the product.
    for (std::size_t i = 0; i < 4; ++i) out_ptr[i] = std::numeric_limits<float>::quiet_NaN();

    kern::ops::MatMul(t_a, t_b, t_out);
    CHECK(out_ptr[0] == 58.0f);
    CHECK(out_ptr[1] == 64.0f);
    CHECK(out_ptr[2] == 139.0f);
    CHECK(out_ptr[3] == 154.0f);

    // A second run into the same buffer must not accumulate on top of the first.
    kern::ops::MatMul(t_a, t_b, t_out);
    CHECK(out_ptr[0] == 58.0f);
    CHECK(out_ptr[1] == 64.0f);
    CHECK(out_ptr[2] == 139.0f);
    CHECK(out_ptr[3] == 154.0f);
}

void test_ops_matmul_non_contiguous_rejected() {
    const kern::Tensor t_a(kern::Shape{3, 2}, kern::DataType::float32);
    kern::Tensor t_view(kern::Shape{2, 3}, kern::DataType::float32);
    kern::ops::Permute(t_a, {1, 0}, t_view); // zero-copy view, strides {1,3}

    const kern::Tensor t_b(kern::Shape{3, 2}, kern::DataType::float32);
    kern::Tensor t_out(kern::Shape{2, 2}, kern::DataType::float32);

    // Raw-pointer kernels must refuse non-contiguous views instead of
    // silently reading the wrong elements.
    bool caught = false;
    try { kern::ops::MatMul(t_view, t_b, t_out); } catch (const std::invalid_argument&) { caught = true; }
    CHECK(caught);
}

void test_ops_matmul_transposed() {
    // Same product as test_ops_matmul, with B supplied pre-transposed [N, K].
    kern::Tensor t_a(kern::Shape{2, 3}, kern::DataType::float32);
    kern::Tensor t_b_t(kern::Shape{2, 3}, kern::DataType::float32);
    kern::Tensor t_out(kern::Shape{2, 2}, kern::DataType::float32);

    auto* a_ptr = static_cast<float*>(t_a.data());
    auto* bt_ptr = static_cast<float*>(t_b_t.data());
    a_ptr[0] = 1.0f; a_ptr[1] = 2.0f; a_ptr[2] = 3.0f;
    a_ptr[3] = 4.0f; a_ptr[4] = 5.0f; a_ptr[5] = 6.0f;
    // Columns of B, stored row-wise.
    bt_ptr[0] = 7.0f;  bt_ptr[1] = 9.0f;  bt_ptr[2] = 11.0f;
    bt_ptr[3] = 8.0f;  bt_ptr[4] = 10.0f; bt_ptr[5] = 12.0f;

    kern::ops::MatMulTransposed(t_a, t_b_t, t_out);

    const auto* out_ptr = static_cast<const float*>(t_out.data());
    CHECK(out_ptr[0] == 58.0f);
    CHECK(out_ptr[1] == 64.0f);
    CHECK(out_ptr[2] == 139.0f);
    CHECK(out_ptr[3] == 154.0f);
}

void test_ops_relu() {
    const kern::Shape shape{2, 2};
    kern::Tensor t_in(shape, kern::DataType::float32);
    kern::Tensor t_out(shape, kern::DataType::float32);

    auto* in_ptr = static_cast<float*>(t_in.data());
    in_ptr[0] = -1.0f; in_ptr[1] = 0.0f; in_ptr[2] = 1.0f; in_ptr[3] = 2.0f;

    kern::ops::ReLU(t_in, t_out);

    const auto* out_ptr = static_cast<float*>(t_out.data());
    CHECK(out_ptr[0] == 0.0f);
    CHECK(out_ptr[1] == 0.0f);
    CHECK(out_ptr[2] == 1.0f);
    CHECK(out_ptr[3] == 2.0f);
}

void test_ops_relu_vectorized() {
    // 10 elements: exercises both the NEON body (8) and the scalar tail (2).
    kern::Tensor t_in(kern::Shape{10}, kern::DataType::float32);
    kern::Tensor t_out(kern::Shape{10}, kern::DataType::float32);

    auto* in_ptr = static_cast<float*>(t_in.data());
    const float values[10] = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f, -4.0f, 4.0f, -5.0f, 5.0f};
    for (std::size_t i = 0; i < 10; ++i) in_ptr[i] = values[i];

    kern::ops::ReLU(t_in, t_out);

    const auto* out_ptr2 = static_cast<const float*>(t_out.data());
    const float expected[10] = {0.0f, 0.0f, 0.0f, 1.0f, 2.0f, 3.0f, 0.0f, 4.0f, 0.0f, 5.0f};
    for (std::size_t i = 0; i < 10; ++i) CHECK(out_ptr2[i] == expected[i]);
}

void test_ops_gelu() {
    const kern::Shape shape{1};
    kern::Tensor t_in(shape, kern::DataType::float32);
    kern::Tensor t_out(shape, kern::DataType::float32);

    auto* in_ptr = static_cast<float*>(t_in.data());
    in_ptr[0] = 0.0f; 

    kern::ops::GELU(t_in, t_out);

    const auto* out_ptr = static_cast<float*>(t_out.data());
    // GELU(0) should be 0
    CHECK(std::abs(out_ptr[0]) < 1e-5f);
}

void test_ops_gelu_accuracy() {
    // 33 elements: NEON body + scalar tail, values across the tanh knee.
    const std::size_t n = 33;
    kern::Tensor t_in(kern::Shape{n}, kern::DataType::float32);
    kern::Tensor t_out(kern::Shape{n}, kern::DataType::float32);

    auto* in_ptr = static_cast<float*>(t_in.data());
    for (std::size_t i = 0; i < n; ++i)
        in_ptr[i] = -4.0f + 0.25f * static_cast<float>(i);

    kern::ops::GELU(t_in, t_out);

    const auto* out_ptr = static_cast<const float*>(t_out.data());
    const float k1 = 0.7978845608f;
    const float k2 = 0.044715f;
    for (std::size_t i = 0; i < n; ++i) {
        const float x = in_ptr[i];
        const float expected = 0.5f * x * (1.0f + std::tanh(k1 * (x + k2 * x * x * x)));
        CHECK(std::abs(out_ptr[i] - expected) < 1e-5f);
    }
}

void test_ops_softmax() {
    const kern::Shape shape{1, 3};
    kern::Tensor t_in(shape, kern::DataType::float32);
    kern::Tensor t_out(shape, kern::DataType::float32);

    auto* in_ptr = static_cast<float*>(t_in.data());
    in_ptr[0] = 1.0f; in_ptr[1] = 2.0f; in_ptr[2] = 3.0f;

    kern::ops::Softmax(t_in, t_out);

    const auto* out_ptr = static_cast<float*>(t_out.data());
    // Softmax([1, 2, 3]) = [e^1, e^2, e^3] / sum(e^1, e^2, e^3)
    // approx: [0.09, 0.24, 0.67]
    CHECK(out_ptr[0] > 0.0f && out_ptr[0] < 0.1f);
    CHECK(out_ptr[1] > 0.2f && out_ptr[1] < 0.3f);
    CHECK(out_ptr[2] > 0.6f && out_ptr[2] < 0.7f);
}

void test_ops_layernorm() {
    const kern::Shape shape{1, 4};
    kern::Tensor t_in(shape, kern::DataType::float32);
    kern::Tensor t_out(shape, kern::DataType::float32);

    auto* in_ptr = static_cast<float*>(t_in.data());
    in_ptr[0] = 1.0f; in_ptr[1] = 2.0f; in_ptr[2] = 3.0f; in_ptr[3] = 4.0f;

    kern::ops::LayerNorm(t_in, t_out);

    const auto* out_ptr = static_cast<float*>(t_out.data());
    // Mean = 2.5
    // Var = [(1-2.5)^2 + (2-2.5)^2 + (3-2.5)^2 + (4-2.5)^2] / 4
    //     = [2.25 + 0.25 + 0.25 + 2.25] / 4 = 1.25
    // Std = sqrt(1.25) approx 1.118
    // Result = (x - 2.5) / 1.118
    CHECK(std::abs(out_ptr[0] - (-1.34164f)) < 1e-4f);
}
