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
        Tensor(const Shape& shape, const DataType dtype);

        ~Tensor();

        // Reducing copy probability
        Tensor(const Tensor&) = delete;
        Tensor& operator=(const Tensor&) = delete;

        // Move semantics with '=' symbol
        explicit Tensor(Tensor&& other) noexcept;
        Tensor& operator=(Tensor&& other) noexcept;

        const Shape& shape() const { return shape_; }
        DataType dtype() const { return dtype_; }
        size_t size_bytes() const noexcept { return capacity_; };
        void* data() { return data_; }                  // For non-const purposes
        const void* data() const { return data_; }      // For const purposes
    private:
        Shape shape_;
        DataType dtype_;
        void* data_;
        size_t capacity_;
        void release();                                 // Free helper

    };
}
