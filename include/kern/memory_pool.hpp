//
// Created by Ivan Maieli on 15/08/2026.
//

#pragma once

#include <cstddef>

static constexpr size_t ALIGNMENT = 64UL;       // SIMD Alignment optimization

namespace kern {
    class MemoryPool {
    public:
        explicit MemoryPool(size_t total_size);
        ~MemoryPool();

        // Disallow copying
        MemoryPool(const MemoryPool&) = delete;
        MemoryPool& operator=(const MemoryPool&) = delete;

        // Allocate a 64-byte aligned block
        void* allocate(size_t size);

        // Reset the pool
        void reset();

    private:
        void* pool_start_;
        size_t total_size_;
        size_t current_offset_;
    };

} // namespace kern