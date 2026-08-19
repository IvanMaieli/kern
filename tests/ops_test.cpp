#include <kern/ops.hpp>
#include <kern/tensor.hpp>
#include <kern/dtype.hpp>
#include <kern/shape.hpp>
#include "test_macros.hpp"
#include "tests.hpp"
#include <vector>
#include <stdexcept>

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

void test_ops_relu() {
    const kern::Shape shape{2, 2};
    kern::Tensor t_in(shape, kern::DataType::float32);
    kern::Tensor t_out(shape, kern::DataType::float32);

    float* in_ptr = static_cast<float*>(t_in.data());
    in_ptr[0] = -1.0f; in_ptr[1] = 0.0f; in_ptr[2] = 1.0f; in_ptr[3] = 2.0f;

    kern::ops::ReLU(t_in, t_out);

    const auto* out_ptr = static_cast<float*>(t_out.data());
    CHECK(out_ptr[0] == 0.0f);
    CHECK(out_ptr[1] == 0.0f);
    CHECK(out_ptr[2] == 1.0f);
    CHECK(out_ptr[3] == 2.0f);
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
