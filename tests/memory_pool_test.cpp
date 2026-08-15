#include <kern/memory_pool.hpp>
#include <kern/tensor.hpp>
#include "test_macros.hpp"
#include "tests.hpp"
#include <new>

void test_memory_pool_alignment() {
    kern::MemoryPool pool(1024);
    
    // First allocation should be aligned
    void* p1 = pool.allocate(10);
    const auto addr1 = reinterpret_cast<uintptr_t>(p1);
    CHECK(addr1 % ALIGNMENT == 0);
    
    // Second allocation should also be aligned
    void* p2 = pool.allocate(20);
    const auto addr2 = reinterpret_cast<uintptr_t>(p2);
    CHECK(addr2 % ALIGNMENT == 0);
    
    // Ensure they don't overlap
    CHECK(static_cast<char*>(p2) >= static_cast<char*>(p1) + 10);
}

void test_memory_pool_reset() {
    kern::MemoryPool pool(256);
    const void* p1 = pool.allocate(100);
    
    pool.reset();
    
    // After reset, we should be able to allocate from the start again
    const void* p2 = pool.allocate(100);
    CHECK(p2 == p1);
}

void test_memory_pool_oom() {
    kern::MemoryPool pool(100);
    
    // Should throw bad_alloc if we exceed capacity
    bool caught = false;
    try {
        auto _t = pool.allocate(200);
    } catch (const std::bad_alloc&) {
        caught = true;
    }
    CHECK(caught);
}
