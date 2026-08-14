#include <kern/tensor.hpp>
#include <kern/dtype.hpp>
#include <kern/shape.hpp>
#include "test_macros.hpp"
#include "tests.hpp"

void test_tensor_basics() {
    const kern::Shape shape{2, 3};
    kern::Tensor t(shape, kern::DataType::float32);

    CHECK(t.shape().element_count() == 6);
    CHECK(t.dtype() == kern::DataType::float32);
    CHECK(t.data() != nullptr);
    CHECK(t.size_bytes() >= 6 * sizeof(float));
}

void test_tensor_alignment() {
    const kern::Shape shape{10};
    kern::Tensor t(shape, kern::DataType::float32);

    const uintptr_t address = reinterpret_cast<uintptr_t>(t.data());
    CHECK((address % ALIGNMENT) == 0);
}

void test_tensor_move() {
    const kern::Shape shape{100};
    kern::Tensor t1(shape, kern::DataType::float32);
    const void* original_ptr = t1.data();

    // Move
    kern::Tensor t2(std::move(t1));

    CHECK(t2.data() == original_ptr);
    CHECK(t1.data() == nullptr);
    CHECK(t2.shape().element_count() == 100);
}
