#include <kern/tensor.hpp>
#include <kern/memory_pool.hpp>
#include <new>

namespace kern {

    namespace {
        size_t calculate_bytes(const Shape& shape, DataType dtype) {
            return shape.element_count() * size_bytes(dtype);
        }

        size_t align_capacity(size_t capacity) {
            return (capacity + ALIGNMENT - 1) & ~(ALIGNMENT - 1);
        }
    } // namespace

    Tensor::Tensor(const Shape& shape, DataType dtype)
        : shape_(shape), dtype_(dtype) {
        const size_t capacity = calculate_bytes(shape, dtype);
        if (capacity > 0) {
            buffer_ = std::make_shared<Buffer>(align_capacity(capacity));
        }
    }

    Tensor::Tensor(const Shape& shape, DataType dtype, MemoryPool& pool)
        : shape_(shape), dtype_(dtype) {
        const size_t capacity = calculate_bytes(shape, dtype);
        if (capacity > 0) {
            size_t padded_capacity = align_capacity(capacity);
            void* data = pool.allocate(padded_capacity);
            buffer_ = std::make_shared<Buffer>(data, padded_capacity);
        }
    }

    Tensor::Tensor(const Shape& shape, DataType dtype, std::shared_ptr<Buffer> buffer)
        : shape_(shape), dtype_(dtype), buffer_(std::move(buffer)) {}
}
