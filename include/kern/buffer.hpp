#pragma once
#include <cstddef>
#include <kern/config.hpp>

namespace kern {
    class Buffer {
    public:
        explicit Buffer(size_t size);
        Buffer(void* data, size_t size);
        ~Buffer();

        Buffer(const Buffer&) = delete;
        Buffer& operator=(const Buffer&) = delete;

        [[nodiscard]] void* data() const noexcept { return data_; }
        [[nodiscard]] size_t size() const noexcept { return size_; }

    private:
        void* data_;
        size_t size_;
        bool owns_;
    };
}
