# Kern Inference Engine

Kern is a lightweight, high-performance C++ inference engine meticulously engineered for Apple Silicon (M-series) architectures. The engine is designed to minimize runtime overhead by prioritizing deterministic memory management, cache-aligned data structures, and copy-free tensor operations.

## Architectural Highlights

Kern differs from generic tensor libraries by emphasizing low-latency execution suitable for small language model (SLM) inference.

### 1. Deterministic Memory Management
Kern eliminates unpredictable memory fragmentation and runtime `free()` latency through two primary mechanisms:
*   **Linear Memory Pool:** A pre-allocated memory arena for batch allocations. This design ensures that all memory requests during the inference cycle are satisfied via simple pointer arithmetic, bypassing the OS memory allocator entirely.
*   **RAII & Reference Counting:** The `Buffer` class acts as the single source of truth for raw memory, managed via `std::shared_ptr`. This allows the engine to track memory ownership safely and automatically, ensuring that buffers are only deallocated when the final `Tensor` view using them is destroyed.

### 2. Copy-Free Structural Operations
To achieve maximum efficiency, structural changes to tensors do not replicate data. When a `Reshape` operation is performed, the engine updates the tensor's metadata (`Shape`) while maintaining a reference to the original `Buffer`. This approach enables complex graph transformations (transposes, reshapes) at virtually zero cost.

### 3. Hardware-Aware Alignment
All memory allocations are strictly aligned to 64-byte boundaries, conforming to the cache line architecture and SIMD alignment requirements of Apple Silicon. This ensures that vectorized operations (SIMD) can execute without unaligned access penalties or additional data shuffling.

---

## Project Structure

*   `app/`: Reference implementations and application-level logic.
*   `include/kern/`: Public APIs defining the core abstractions:
    *   `Tensor`: The high-level data container.
    *   `Buffer`: The reference-counted raw memory manager.
    *   `MemoryPool`: The linear allocator.
    *   `Shape`: Multidimensional metadata translator.
*   `src/`: Core implementation logic.
*   `tests/`: Comprehensive unit test suite, including alignment validation, ownership semantics, and operator correctness.

---

## Getting Started

### Prerequisites
*   Compiler supporting C++20 (Clang recommended).
*   CMake 3.25+.
*   Ninja (highly recommended for rapid iteration).

### Build Instructions

1.  **Configure:**
    ```bash
    mkdir build && cd build
    cmake .. -G Ninja
    ```

2.  **Compile:**
    ```bash
    ninja
    ```

3.  **Validate:**
    Execute the test suite to verify implementation integrity:
    ```bash
    ./kern-tests
    ```

---

## Usage Example

The following demonstrates the engine's capability to allocate memory from a pool, perform a zero-copy reshape, and execute an operator.

```cpp
#include <kern/tensor.hpp>
#include <kern/memory_pool.hpp>
#include <kern/ops.hpp>

// 1. Initialize Memory Pool
kern::MemoryPool pool(1024);

// 2. Define Shape and create Tensors using the pool
kern::Shape shape_linear{4};
kern::Shape shape_matrix{2, 2};

kern::Tensor t_in(shape_linear, kern::DataType::float32, pool);
kern::Tensor t_out(shape_matrix, kern::DataType::float32, pool);

// 3. Perform a zero-copy Reshape
// t_out now points to the same underlying buffer as t_in
kern::ops::Reshape(t_in, shape_matrix, t_out);

// 4. Execute Operator
kern::ops::Add(t_in, t_in, t_out);
```

## Developer Notes

*   **Memory Safety:** The engine uses `[[nodiscard]]` extensively on accessors, allocators, and metadata getters. Please honor these warnings during development; they are present to catch ownership or logic errors at compile time.
*   **Ownership Semantics:** Avoid manually freeing memory in `src/` modules. Ownership is explicitly handled by the `Buffer` and `MemoryPool` RAII wrappers.
*   **Adding Operators:** All new operators must be defined within the `kern::ops` namespace. Favor functional-style operators that accept an output tensor reference, allowing the user to dictate memory allocation policies.
