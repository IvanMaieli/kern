#include <kern/buffer.hpp>
#include <kern/tensor.hpp>
#include <cstdlib>
#include <new>

namespace kern {
    Buffer::Buffer(const size_t size) : size_(size), owns_(true) {
        data_ = std::aligned_alloc(ALIGNMENT, size);
        if (!data_) throw std::bad_alloc();
    }

    Buffer::Buffer(void* data, const size_t size) : data_(data), size_(size), owns_(false) {}

    Buffer::~Buffer() {
        if (owns_ && data_) std::free(data_);
    }
}
