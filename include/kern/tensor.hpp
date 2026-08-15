//
// Created by Ivan Maieli on 14/08/2026.
//

#pragma once

#include <kern/dtype.hpp>
#include <kern/shape.hpp>

namespace kern {
    class MemoryPool;

    class Tensor {
    public:
        Tensor() = default;
        Tensor(const Shape& shape, const DataType dtype);
        Tensor(const Shape& shape, const DataType dtype, MemoryPool& pool);

        ~Tensor();

        // Reducing copy probability
        Tensor(const Tensor&) = delete;
        Tensor& operator=(const Tensor&) = delete;

        // Move semantics with '=' symbol
        explicit Tensor(Tensor&& other) noexcept;
        Tensor& operator=(Tensor&& other) noexcept;

        [[nodiscard]] const Shape& shape() const { return shape_; }
        [[nodiscard]] DataType dtype() const { return dtype_; }
        [[nodiscard]] size_t size_bytes() const noexcept { return capacity_; }
        [[nodiscard]] void* data() { return data_; }                                // For non-const purposes
        [[nodiscard]] const void* data() const { return data_; }                    // For const purposes
    private:
        Shape shape_;
        DataType dtype_;
        void* data_;
        size_t capacity_;
        MemoryPool* pool_ = nullptr;
        void release();                                                             // Free helper
    };
}
