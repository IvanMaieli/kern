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
To support flexible arithmetic, we implement broadcasting through precomputed *effective strides*: each operand is right-aligned against the output rank, and any axis it broadcasts on gets a stride of **0**. A zero stride "freezes" the pointer at the same memory location for that axis, so shapes broadcast without any data duplication or extra memory allocation — including rank-0 scalars, which broadcast to every element.
The output is walked with an odometer over the leading axes (no per-element div/mod), and the last axis is vectorized with NEON whenever both operands have unit stride there. In-place operation (`out` aliasing an input) is permitted when the aliased operand matches `out` exactly (same shape, contiguous) and rejected otherwise.

### 3. Memory Safety, Aliasing and Views
We implement a functional-style interface where the user defines the destination tensor. An `is_aliased` check lets element-wise kernels (Add, ReLU, GELU) detect overlapping input/output buffers and run safely in place — same-shape in-place `Add` takes the vectorized path directly. Kernels that cannot tolerate overlap (MatMul, MatMulTransposed) reject an aliased `out` outright.
Zero-copy view operations (`Transpose`, `Permute`) permute the `Shape`'s strides together with its dimensions, so the metadata always describes the real buffer layout. Raw-pointer kernels verify `Shape::is_contiguous()` and reject views they cannot read correctly instead of silently misindexing.

---

## Performance Benchmarking

We rigorously benchmark core operators to track improvements. Below is the historical performance for a **MatMul (512x512, float32)** operation on Apple Silicon.

| Implementation Stage | Description | Execution Time (ms) | Speedup |
| :--- | :--- | :--- | :--- |
| **Baseline (Scalar)** | Naive triple-nested loop | ~3500+ ms | 1.0x |
| **Loop Reordered (M-K-N)** | Optimized for cache-line access | ~1880 ms | ~1.8x |
| **SIMD (NEON + FMA)** | Vectorized inner loop (4 elements) | ~1070 ms | ~3.3x |
| **Multicore (std::jthread)** | Parallelized over M rows | ~170 ms | ~20.5x |
| **Tiled + ThreadPool** | M-N-K 64x64 tiles, persistent pool, 2x-unrolled FMA | ~8 ms | ~440x |

Current operator benchmarks (Release build, M-series):

| Operator | Shape | Execution Time |
| :--- | :--- | :--- |
| MatMul | 512x512 | ~8 ms |
| MatMulTransposed | 512x512 (B supplied as B^T) | ~4 ms |
| Broadcast Add | 512x512 + 512 | ~0.04 ms |
| GELU | 512x2048 | ~0.17 ms/iter |

*Note: Benchmarks are averaged over multiple iterations using `std::chrono` on an M-series Apple Silicon chip (`kern-tests` bench suite, Release build).*

---

## Operators Implementation

Our operator library is designed to be highly modular, with all kernels placed in the `kern::ops` namespace.

*   **MatMul / MatMulTransposed:** The engine's powerhouse. `MatMul` uses an optimized tiled `m-n-k` loop ordering with SIMD acceleration across a persistent thread pool, and zeroes its output tiles so destination buffers can be safely reused. `MatMulTransposed` takes `B` pre-transposed (`[N, K]`): both operands are then contiguous along the reduction axis, making it ~2x faster than `MatMul` on the same shapes — the preferred layout for stored weights.
*   **Activation Functions:** `ReLU` (NEON `vmax`) and `GELU` (a vectorized `tanh` approximation built from a Pade approximant and the double-angle identity, accurate to ~1e-7) run as parallel element-wise kernels.
*   **Reduction Operators:** `Softmax` and `LayerNorm` introduce the concept of "reduction" (aggregating multiple values). `Softmax` includes a "Max Trick" for numerical stability to prevent overflows in exponentiation.

---

## Performance & Optimization Strategy

1.  **Restrict Pointers:** All raw data pointers are decorated with `__restrict`. This tells the compiler that the memory regions do not overlap, enabling more aggressive vectorization and SIMD instruction generation.
2.  **64-Byte Alignment:** All memory buffers are aligned to 64-byte boundaries, matching the cache line size of Apple Silicon and the requirement for optimal NEON SIMD loading.
3.  **Benchmarking Infrastructure:** We utilize `std::chrono` for micro-benchmarking core operators to track performance regressions and validate the impact of optimization techniques like SIMD.

---

## Roadmap for Future Development

*   ~~Multicore Parallelization~~ and ~~Tiled MatMul~~: shipped — a persistent `ThreadPool` drives all compute-heavy kernels, and MatMul tiles the `M-N-K` loops in 64x64 blocks.
*   **Decode-Path GEMV:** With `M = 1` (LLM decoding), parallelizing only over M leaves the pool idle; partition the workload over N/K as well.
*   **float16 / int8 Kernels:** The dtype system exists; the kernels do not yet.
*   **Graph Execution Engine:** Moving from manually called operators to a `Session` object that compiles and executes an entire model graph with optimal scheduling.
