#pragma once

#include <kern/dtype.hpp>
#include <kern/shape.hpp>
#include <kern/buffer.hpp>
#include <kern/config.hpp>
#include <memory>

namespace kern {
    class MemoryPool;

    class Tensor {
    public:
        Tensor() = default;
        Tensor(const Shape& shape, const DataType dtype);
        Tensor(const Shape& shape, const DataType dtype, MemoryPool& pool);
        
        // Constructor for View
        Tensor(const Shape& shape, const DataType dtype, std::shared_ptr<Buffer> buffer);

        ~Tensor() = default;

        Tensor(const Tensor&) = default;
        Tensor& operator=(const Tensor&) = default;

        Tensor(Tensor&& other) noexcept = default;
        Tensor& operator=(Tensor&& other) noexcept = default;

        [[nodiscard]] const Shape& shape() const { return shape_; }
        [[nodiscard]] DataType dtype() const { return dtype_; }
        [[nodiscard]] size_t size_bytes() const noexcept { return buffer_ ? buffer_->size() : 0; }
        [[nodiscard]] void* data() { return buffer_ ? buffer_->data() : nullptr; }
        [[nodiscard]] const void* data() const { return buffer_ ? buffer_->data() : nullptr; }
        [[nodiscard]] std::shared_ptr<Buffer> buffer() const { return buffer_; }
        
        void set_shape(const Shape& shape) { shape_ = shape; }
        void set_buffer(const std::shared_ptr<Buffer> &buffer) { buffer_ = buffer; }

    private:
        Shape shape_;
        DataType dtype_;
        std::shared_ptr<Buffer> buffer_;
    };
}
