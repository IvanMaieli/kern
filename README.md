# Kern Inference Engine

Kern is a lightweight, high-performance C++20 inference engine meticulously engineered for Apple Silicon (M-series) architectures. The engine is designed to minimize runtime overhead by prioritizing deterministic memory management, cache-aligned data structures, and copy-free tensor operations.

## Design Philosophy

Kern adheres to a strict set of architectural principles designed for performance and reliability in constrained environments:

1.  **Deterministic Memory Management (RAII):** We eliminate unpredictable memory fragmentation and runtime `free()` latency. By using a custom linear `MemoryPool` and a centralized `Buffer` management system, we ensure that memory lifecycles are explicitly defined and managed.
2.  **Copy-Free View Semantics:** Transformations such as `Reshape`, `Transpose`, and `Permute` are implemented as zero-copy "view" operations. By decoupling the `Tensor` (metadata: `Shape` and `Strides`) from the `Buffer` (data), we manipulate views without duplicating underlying data.
3.  **Hardware-Aware Performance:** All memory allocations are strictly aligned to 64-byte boundaries, conforming to the SIMD alignment requirements and cache line architecture of Apple Silicon.

---

## Architectural Implementation Details

### Memory Ownership and Sharing
To facilitate copy-free views, Kern implements a robust memory ownership model:
*   **Buffer Class:** Acting as the single source of truth, `Buffer` manages the raw memory block. It handles allocation via `std::aligned_alloc` and deallocation via `std::free`.
*   **Reference Counting:** We utilize `std::shared_ptr<Buffer>` within each `Tensor`. This allows multiple tensor "views" to coexist and share the same memory buffer safely. The `Buffer` object only releases its underlying memory when the final `shared_ptr` referencing it is destroyed.
*   **Ownership Flag:** The `owns_` boolean within `Buffer` distinguishes between memory managed by the `Buffer` object itself and externally managed memory (e.g., from a `MemoryPool`), preventing double-free scenarios.

### Multidimensional Metadata (Shape & Strides)
Kern handles N-dimensional tensors through a stride-based mapping:
*   **Stride-based Indexing:** Instead of relying on contiguity, every `Shape` tracks a `strides_` array. This allows the `linear_index` calculation to map N-dimensional coordinates `(c0, c1, ..., cN)` to a flat 1D memory index.
*   **Permutation/Transpose:** By updating the `Shape` metadata and swapping stride/dimension values, operations like `Transpose` or `Permute` reconfigure the mapping without touching the physical buffer.

### Broadcasting Logic
The engine supports automatic broadcasting during binary operations (e.g., `Add`).
*   **Stride-Zero Broadcasting:** To broadcast a dimension of size 1 to size N, we set the stride of that dimension to 0. During index calculation, this effectively "freezes" the pointer for that dimension, allowing the engine to read the same memory location repeatedly across the broadcasted axis without extra data movement.

---

## Project Structure

*   `app/`: Reference implementations and application-level logic.
*   `include/kern/`: Public APIs defining core abstractions:
    *   `Tensor`: Lightweight container for metadata and buffer references.
    *   `Buffer`: Reference-counted memory container.
    *   `MemoryPool`: Linear allocator for zero-overhead batch allocations.
    *   `Shape`: Metadata translator handling strides and N-dimensional mapping.
    *   `ops/`: Mathematical operators.
*   `src/`: Core implementation logic.
*   `tests/`: Comprehensive unit test suite, including alignment validation, ownership semantics, and operator correctness.

---

## Build and Usage

### Requirements
*   Compiler supporting C++20 (Clang recommended).
*   CMake 3.25+.
*   Ninja (recommended).

### Build Instructions
```bash
mkdir build && cd build
cmake .. -G Ninja
ninja
./kern-tests
```

### Usage Example
```cpp
#include <kern/tensor.hpp>
#include <kern/memory_pool.hpp>
#include <kern/ops.hpp>

kern::MemoryPool pool(1024);
kern::Shape shape_a{2, 3};
kern::Shape shape_b{1, 3}; // Compatible for broadcasting
kern::Tensor t_a(shape_a, kern::DataType::float32, pool);
kern::Tensor t_b(shape_b, kern::DataType::float32, pool);
kern::Tensor t_out({2, 3}, kern::DataType::float32, pool);

// Broadcast Add: t_b is effectively stretched to [2, 3]
kern::ops::Add(t_a, t_b, t_out);
```

## Developer Notes

*   **Memory Safety:** The engine utilizes `[[nodiscard]]` to prevent accidental resource leaks or ignored results.
*   **Adding Operators:** All new operators must be defined within the `kern::ops` namespace. Favor functional-style operators that accept an output tensor reference, allowing the user to dictate memory allocation policies.
