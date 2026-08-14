//
// Created by Ivan Maieli on 14/08/2026.
//

#pragma once

#include <kern/dtype.hpp>
#include <kern/shape.hpp>

namespace kern {
    class Tensor {
    public:
        Tensor() = default;
        Tensor(Shape shape, DType dtype);

        // Reducing copy probability
        Tensor(const Tensor&) = delete;
        Tensor& operator=(const Tensor&) = delete;

        // Move semantics with '=' symbol
        Tensor(Tensor&& other) noexcept;
        Tensor& operator=(Tensor&& other) noexcept;

        const Shape& shape() const { return _shape; }
        DType dtype() const { return _dtype; }
        void* data() { return _data; }                  // For non-const purposes
        const void* data() const { return _data; }      // For const purposes
        void release();                                 // Free helper

    private:
        Shape shape_;
        DataType dtype_;
        void* data_;
        size_t capacity_;
    };
}
