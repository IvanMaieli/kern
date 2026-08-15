#include <kern/tensor.hpp>
#include <kern/memory_pool.hpp>
#include <cstdlib>
#include <new>

namespace kern {

    namespace {
        size_t calculate_bytes(const Shape& shape, const DataType dtype) {
            return shape.element_count() * size_bytes(dtype);
        }

        size_t align_capacity(const size_t capacity) {
            return capacity + ALIGNMENT - 1 & ~(ALIGNMENT - 1);
        }
    } // namespace

    Tensor::Tensor(const Shape& shape, const DataType dtype)
        : shape_(shape),
          dtype_(dtype),
          data_(nullptr),
          capacity_(calculate_bytes(shape, dtype)) {

        if (capacity_ > 0) {
            const size_t padded_capacity = align_capacity(capacity_);
            data_ = std::aligned_alloc(ALIGNMENT, padded_capacity);
            if (!data_) throw std::bad_alloc();
        }
    }

    Tensor::Tensor(const Shape& shape, const DataType dtype, MemoryPool& pool)
        : shape_(shape),
          dtype_(dtype),
          data_(nullptr),
          capacity_(calculate_bytes(shape, dtype)),
          pool_(&pool) {

        if (capacity_ > 0) {
            const size_t padded_capacity = align_capacity(capacity_);
            data_ = pool.allocate(padded_capacity);
        }
    }

    Tensor::~Tensor() {
        release();
    }

    Tensor::Tensor(Tensor&& other) noexcept
        : shape_(other.shape_),
          dtype_(other.dtype_),
          data_(other.data_),
          capacity_(other.capacity_),
          pool_(other.pool_) {

        other.data_ = nullptr;
        other.capacity_ = 0;
        other.pool_ = nullptr;
    }

    Tensor& Tensor::operator=(Tensor&& other) noexcept {
        if (this != &other) {
            release();

            shape_ = other.shape_;
            dtype_ = other.dtype_;
            data_ = other.data_;
            capacity_ = other.capacity_;
            pool_ = other.pool_;

            other.data_ = nullptr;
            other.capacity_ = 0;
            other.pool_ = nullptr;
        }
        return *this;
    }

    void Tensor::release() {
        // Only free if we own the memory (i.e., not using a MemoryPool)
        if (pool_ == nullptr && data_) std::free(data_);
        data_ = nullptr;
        capacity_ = 0;
        pool_ = nullptr;
    }

} // namespace kern
