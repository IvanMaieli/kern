# Kern: High-Performance Inference Engine for Apple Silicon

Kern is a high-performance C++20 inference engine meticulously engineered from the ground up for Apple Silicon (M-series). It is designed to bridge the gap between high-level machine learning frameworks and raw hardware performance.

## Design Philosophy

Kern is built on the belief that peak inference performance requires full control over the memory layout and the CPU execution pipeline. Our core tenets are:

*   **Deterministic Memory Lifecycle:** In inference, memory fragmentation is the enemy of latency. We avoid unpredictable runtime allocations through a custom `MemoryPool` and an RAII-based `Buffer` management system. Memory is allocated once (or managed explicitly) and reused.
*   **Zero-Copy View Semantics:** Most inference operations (Reshape, Transpose, Permute) shouldn't touch the data; they should only change how the data is *interpreted*. Kern achieves this by decoupling metadata (`Shape` and `Strides`) from the data buffer, allowing complex structural transformations to happen at zero cost.
*   **Cache Locality:** Modern CPUs are faster than RAM. Kern optimizes performance not just by reducing operations, but by optimizing memory access patterns (Loop Reordering and Tiling) to ensure data stays within the CPU's L1/L2 caches for as long as possible.

---

## Architectural Deep Dive

### 1. Metadata and Stride-based Indexing
Kern handles N-dimensional tensors through a sophisticated stride-based mapping. Instead of enforcing memory contiguity, every `Tensor` maintains a `Shape` object containing `dimensions_` and `strides_`.
The linear index for any coordinate $(c_0, c_1, ..., c_N)$ is calculated as:
$$ Index = \sum_{i=0}^{N} (c_i \times stride_i) $$
This allows the engine to represent non-contiguous views (e.g., a transposed matrix) as a standard tensor, delegating the complexity to the index calculation rather than moving physical bytes.

### 2. Broadcasting via Stride-Zero
To support flexible arithmetic, we implemented a robust broadcasting mechanism. When a dimension needs to be "broadcasted" (e.g., adding a vector `[1, 1024]` to a matrix `[32, 1024]`), we set the `stride` of the broadcasted dimension to **0**.
During the index calculation, the coordinate for that axis is multiplied by 0, effectively "freezing" the pointer at the same memory location for that dimension. This allows the engine to broadcast shapes without any data duplication or extra memory allocation.

### 3. Memory Safety and Aliasing
We implement a functional-style interface where the user defines the destination tensor. To maximize efficiency, we added an `is_aliased` check. This allows our element-wise kernels (Add, ReLU, GELU) to safely detect when input and output buffers overlap (in-place operations), preventing expensive allocations in memory-constrained scenarios while maintaining mathematical safety for structural operators (MatMul, Permute).

---

## Operators Implementation

Our operator library is designed to be highly modular, with all kernels placed in the `kern::ops` namespace.

*   **MatMul:** The engine's powerhouse. We utilize an optimized `m-k-n` loop ordering. By reordering the inner loop from the traditional `n` (column) to `n` (sequential access to B's rows), we ensure sequential memory access, drastically improving the CPU cache hit rate.
*   **Activation Functions:** Foundational units like `ReLU` and `GELU` (using a fast `tanh` approximation) are implemented as highly parallelizable element-wise kernels.
*   **Reduction Operators:** `Softmax` and `LayerNorm` introduce the concept of "reduction" (aggregating multiple values). `Softmax` includes a "Max Trick" for numerical stability to prevent overflows in exponentiation.

---

## Performance & Optimization Strategy

1.  **Restrict Pointers:** All raw data pointers are decorated with `__restrict`. This tells the compiler that the memory regions do not overlap, enabling more aggressive vectorization and SIMD instruction generation.
2.  **64-Byte Alignment:** All memory buffers are aligned to 64-byte boundaries, matching the cache line size of Apple Silicon and the requirement for optimal NEON SIMD loading.
3.  **Benchmarking Infrastructure:** We utilize `std::chrono` for micro-benchmarking core operators to track performance regressions and validate the impact of optimization techniques like SIMD and Tiling.

---

## Roadmap for Future Development

*   **SIMD Vectorization:** Explicit implementation of ARM NEON intrinsics (`vld1q_f32`, `vaddq_f32`, `vfmaq_f32`) for all kernels to achieve theoretical peak hardware utilization.
*   **Advanced MatMul:** Implementation of Cache Tiling to further minimize RAM roundtrips for large matrices.
*   **Graph Execution Engine:** Moving from manually called operators to a `Session` object that compiles and executes an entire model graph with optimal scheduling.
