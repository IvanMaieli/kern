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
    kern::Shape shape_in{2, 3, 4};
    kern::Shape shape_out{4, 2, 3};
    kern::Tensor t_in(shape_in, kern::DataType::float32);
    kern::Tensor t_out(shape_out, kern::DataType::float32);

    kern::ops::Permute(t_in, {2, 0, 1}, t_out);

    CHECK(t_out.shape() == shape_out);
    CHECK(t_out.buffer() == t_in.buffer());
}
