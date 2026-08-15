//
// Created by Ivan Maieli on 15/08/2026.
//

#include <kern/memory_pool.hpp>
#include <new>
#include "kern/tensor.hpp"

namespace kern {

    MemoryPool::MemoryPool(const size_t total_size)
        : total_size_((total_size + ALIGNMENT - 1) & ~(ALIGNMENT - 1)), current_offset_(0) {
        // Allocate the large block aligned to ALIGNMENT bytes
        pool_start_ = std::aligned_alloc(ALIGNMENT, total_size_);
        if (!pool_start_) throw std::bad_alloc();
    }

    MemoryPool::~MemoryPool() {
        if (pool_start_) std::free(pool_start_);
    }

    void* MemoryPool::allocate(const size_t size) {
        const uintptr_t current_addr = reinterpret_cast<uintptr_t>(pool_start_) + current_offset_;

        // Calculate padding for ALIGNMENT-byte alignment using bitwise ops
        const uintptr_t misalignment = current_addr & (ALIGNMENT - 1);
        const uintptr_t padding = (ALIGNMENT - misalignment) & (ALIGNMENT - 1);

        if (current_offset_ + padding + size > total_size_)
            throw std::bad_alloc();

        current_offset_ += padding;
        void* allocated_ptr = static_cast<char*>(pool_start_) + current_offset_;
        current_offset_ += size;

        return allocated_ptr;
    }

    void MemoryPool::reset() {
        current_offset_ = 0;
    }

} // namespace kern

