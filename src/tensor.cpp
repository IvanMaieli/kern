//
// Created by Ivan Maieli on 14/08/2026.
//

#include <kern/tensor.hpp>
#include <kern/dtype.hpp>
#include <kern/shape.hpp>

namespace kern {
    namespace  {
        size_t calc_bytes(const Shape& shape, const DataType dtype ) {
            return shape.element_count() * size_bytes(dtype);
        }
        size_t align_capacity(const size_t capacity) {
                 return (capacity + ALIGNMENT - 1) & ~(ALIGNMENT - 1);
        }
    }

    Tensor::Tensor(const Shape& shape, const DataType dtype)
        : shape_(shape), dtype_(dtype), data_(nullptr), capacity_(calc_bytes(shape_, dtype_)) {
        if (capacity_ > 0) {
            const size_t padded_capacity = align_capacity(capacity_);
            data_ = std::aligned_alloc(ALIGNMENT, padded_capacity);
            if (!data_) throw std::bad_alloc();
        }

    }

    Tensor::Tensor(Tensor&& other) noexcept :
        shape_(other.shape_), dtype_(other.dtype_),
        data_(other.data_), capacity_(other.capacity_) {
        other.data_ = nullptr;
        other.capacity_ = 0;
    }

    Tensor& Tensor::operator=(Tensor&& other) noexcept {
        if (this == &other) return *this;
        release();
        shape_ = std::move(other.shape_);
        dtype_ = other.dtype_;
        data_ = other.data_;
        capacity_ = other.capacity_;
        other.data_ = nullptr;
        other.capacity_ = 0;
        return *this;
    }

    void Tensor::release()  {
        if (data_) {
            std::free(data_);
            data_ = nullptr;
        }
        capacity_ = 0;
    }

    Tensor::~Tensor() { release(); }
}