# Kern: High-Performance Inference Engine for Apple Silicon

Kern is a high-performance C++20 inference engine meticulously engineered from the ground up for Apple Silicon (M-series). It is designed to bridge the gap between high-level machine learning frameworks and raw hardware performance.

## Design Philosophy

Kern is built on the belief that peak inference performance requires full control over the memory layout and the CPU execution pipeline. Our core tenets are:

*   **Deterministic Memory Lifecycle:** In inference, memory fragmentation is the enemy of latency. We avoid unpredictable runtime allocations through a custom `MemoryPool`, an RAII-based `Buffer` management system, and a graph-level activation arena planned by liveness analysis. Memory is allocated once (or managed explicitly) and reused.
*   **Zero-Copy View Semantics:** Most inference operations (Reshape, Transpose, Permute) shouldn't touch the data; they should only change how the data is *interpreted*. Kern achieves this by decoupling metadata (`Shape` and `Strides`) from the data buffer, allowing complex structural transformations to happen at zero cost.
*   **Cache Locality:** Modern CPUs are faster than RAM. Kern optimizes performance not just by reducing operations, but by optimizing memory access patterns (Loop Reordering and Tiling) to ensure data stays within the CPU's L1/L2 caches for as long as possible.
*   **Two Storage Formats, One Kernel Body:** float32 and float16 share identical kernel code through a dtype-traits layer. float16 halves the memory footprint (the bandwidth-bound win) while all arithmetic stays in float32 lanes, keeping accumulation precision identical to the float32 kernels.

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
We implement a functional-style interface where the user defines the destination tensor. An `is_aliased` check lets element-wise kernels (Add, ReLU, GELU) detect overlapping input/output buffers and run safely in place — same-shape in-place `Add` takes the vectorized path directly. Kernels that cannot tolerate overlap (MatMul, MatMulTransposed, MatVec) reject an aliased `out` outright.
Zero-copy view operations (`Transpose`, `Permute`) permute the `Shape`'s strides together with its dimensions, so the metadata always describes the real buffer layout. Raw-pointer kernels verify `Shape::is_contiguous()` and reject views they cannot read correctly instead of silently misindexing.

### 4. Parallel Execution: `parallel_for`
All compute-heavy kernels share one dispatch primitive: `parallel_for(total, min_work, body)` splits a range across the persistent `ThreadPool` (hardware-concurrency `jthread`s) and joins via a `std::latch`. A `min_work` threshold keeps small tensors single-threaded — dispatching a 10-element ReLU to eight cores would cost 100x more in synchronization than in arithmetic.

### 5. Dtype Traits: float16 Without a Second Kernel Library
`F32Traits` and `F16Traits` provide `load4/store4/load1/store1`; every templated kernel body (Add, ReLU, GELU, Softmax, LayerNorm, MatMul, MatMulTransposed, MatVec) is written once against the traits. For float16, loading converts `__fp16` vectors to float32 lanes (`vcvt_f32_f16`) and storing converts back — so the fp16 result equals the fp32 kernel applied to rounded inputs, with half the bytes streamed from memory. Zero is an all-zero byte pattern in both formats, so the tiled-MatMul zero-fill stays a plain `memset`.

Where conversion throughput would cap performance (dot products, the MatMul micro-kernel), the traits path is bypassed for **FMLAL** kernels (`vfmlalq_low/high_f16`, guarded by `__ARM_FEATURE_FP16_FML`): fp16 loads feed a widening multiply-accumulate into fp32 accumulators directly — no conversion instructions, 8 elements per vector pair, and fp32 accumulation precision. Toolchains or targets without `fp16fml` fall back to the conversion kernels automatically.

### 5b. Register-Blocked Micro-Kernels
Two throughput techniques applied throughout the compute kernels:
*   **Independent FMA chains:** a single accumulator vector serializes on FMA latency (~4 cycles); every dot product and the MatMul micro-kernel keep 4+ independent chains so both FMA pipes stay busy.
*   **4x16 register micro-tiles in MatMul:** inside a 64x64 tile, 16 accumulators cover 4 rows x 16 output columns, so each 4-vector B load feeds 16 FMAs — the inner loop is FMA-bound instead of load-bound (a 1-row block pays one B load per 4 FMAs and wastes ~2x). `out` is read/written once per k-tile instead of once per k step (64x less output traffic), and the first k-tile defines the tile from zero so stale destination buffers never leak into results. The f16 variant runs the same shape with FMLAL widening multiply-accumulates.

### 6. MatVec: the Decode-Path GEMV
`out[M] = A[M,K] · v[K]` is the shape LLM decoding hits every token (M = 1). With so few rows, row-parallelism cannot fill the pool, so `MatVec` chooses between two strategies:
*   **M ≥ core count:** parallel over rows, one NEON dot product per row.
*   **M small:** the K axis is split into chunks; each thread computes partial dot products into a *disjoint scratch column* (no atomics), and the caller reduces them in deterministic order.

---

## Compute Graph

`kern::Graph` (in `include/kern/graph.hpp`) lifts kern from a kernel library to an execution engine:

```cpp
kern::Graph g;
auto x  = g.add_input(kern::Shape{2, 16}, kern::DataType::float32);
auto wq = g.add_param(kern::Shape{16, 16}, kern::DataType::float32);
auto h  = g.add_matmul_transposed(x, wq);
auto n  = g.add_layernorm(h);
auto a  = g.add_gelu(g.add_matmul_transposed(n, w2));
auto r  = g.add_add(a, x);        // residual
g.mark_output(r);
g.build();
g.run();                           // read g.value(r)
```

Design points:

*   **SSA-style API:** every node produces a fresh `TensorId` and may only consume previously defined ids, so submission order is automatically a valid topological order — no scheduling pass.
*   **Arena planning by liveness (`build()`):** each activation's live range is `[defining node, last use]`; a linear-scan allocator (the same idea used for register allocation in JITs) packs non-overlapping ranges into slots of one 64-byte-aligned `Buffer` arena. A slot is recycled only when the new value is defined strictly after the previous tenant's last use, so a node never reads an input through memory already rewritten as its own output. `mark_output()` extends a value's live range so its slot survives `run()`.
*   **Shape/dtype validation at add time:** dimension mismatches throw when the graph is built, not mid-inference.
*   Supported nodes: `MatMul`, `MatMulTransposed`, `MatVec`, `Add` (broadcasting included), `ReLU`, `GELU`, `Softmax`, `LayerNorm`; float32 and float16.

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

Current operator benchmarks (Release build, M-series; median of 20–30 iterations after warm-up):

| Operator | Shape | Median | p99 | Throughput |
| :--- | :--- | :--- | :--- | :--- |
| MatMul (f32) | 512x512 | ~0.72 ms | ~0.82 ms | ~370 GFLOPS (~63% of 4-P-core peak) |
| MatMulTransposed (f32) | 512x512 (B as B^T) | ~1.39 ms | ~1.48 ms | ~195 GFLOPS |
| MatMulTransposed (f16) | 512x512 | ~0.84 ms | ~0.93 ms | ~320 GFLOPS |
| MatVec (f32) | [4096x4096]·[4096] | ~0.47 ms | ~0.51 ms | ~135 GB/s |
| MatVec (f16) | [4096x4096]·[4096] | ~0.23 ms | ~0.24 ms | ~145 GB/s (≈ peak) |
| Broadcast Add | 512x512 + 512 | ~0.02 ms | | |
| GELU | 512x2048 | ~0.13 ms/iter | | |

Two scaling notes worth knowing:
*   With the 4x16 register-blocked micro-kernel, plain `MatMul` on row-major B now **outperforms `MatMulTransposed`** (~370 vs ~195 GFLOPS): the register block amortizes each B load across 4 rows, which the dot-product layout cannot. `MatMulTransposed` remains the right shape for `M = 1` decode work (see `MatVec`), but batched matmuls should use `MatMul`.
*   Thread-pool workers set `QOS_CLASS_USER_INITIATED` on Apple platforms so the scheduler biases them onto P-cores; combined with median/p99 benchmark reporting, per-iteration jitter dropped from ~65% to ~10%.

*Note: Benchmarks are averaged over multiple iterations using `std::chrono` on an M-series Apple Silicon chip (`kern-tests` bench suite, Release build).*

---

## Operators Implementation

Our operator library is designed to be highly modular, with all kernels placed in the `kern::ops` namespace. Every operator supports **float32 and float16**; `int8` is rejected with a clear error until quantized kernels land.

*   **MatMul / MatMulTransposed:** The engine's powerhouse. `MatMul` uses an optimized tiled `m-n-k` loop ordering with SIMD acceleration across a persistent thread pool, and zeroes its output tiles so destination buffers can be safely reused. `MatMulTransposed` takes `B` pre-transposed (`[N, K]`): both operands are then contiguous along the reduction axis, making it ~2x faster than `MatMul` on the same shapes — the preferred layout for stored weights.
*   **MatVec:** the decode-path GEMV (see above), with row-parallel and K-split strategies.
*   **Activation Functions:** `ReLU` (NEON `vmax`) and `GELU` (a vectorized `tanh` approximation built from a Pade approximant and the double-angle identity, accurate to ~1e-7) run as parallel element-wise kernels.
*   **Reduction Operators:** `Softmax` and `LayerNorm` are NEON-vectorized in all passes (max reduction, mean/variance via FMA, normalization by reciprocal product) and parallelized over rows. `Softmax` includes a "Max Trick" for numerical stability to prevent overflows in exponentiation.

---

## Performance & Optimization Strategy

1.  **Restrict Pointers:** All raw data pointers are decorated with `__restrict`. This tells the compiler that the memory regions do not overlap, enabling more aggressive vectorization and SIMD instruction generation.
2.  **64-Byte Alignment:** All memory buffers are aligned to 64-byte boundaries, matching the cache line size of Apple Silicon and the requirement for optimal NEON SIMD loading.
3.  **Work-Thresholded Threading:** element-wise kernels dispatch to the pool only when the per-task chunk exceeds ~1024 elements; below that they run inline, avoiding synchronization overhead that would dwarf the arithmetic.
4.  **Benchmarking Infrastructure:** We utilize `std::chrono` for micro-benchmarking core operators to track performance regressions and validate the impact of optimization techniques like SIMD.

---

## Building and Testing

```bash
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release -G Ninja
cmake --build build-release -j
./build-release/kern-tests       # runs the full suite incl. benchmarks (read numbers in Release only)
```

The test suite covers: shape/stride semantics, zero-copy views, broadcasting (including rank-0 scalars and rejected in-place broadcasting), MatMul buffer reuse with NaN-poisoned outputs, non-contiguous view rejection, NEON tail handling, GELU accuracy sweeps, fp16 kernels, MatVec (small-K tail and 4096-wide decode shape), and the graph engine (correctness vs. manually-chained ops, arena reuse, rerun stability).

---

## Roadmap for Future Development

*   ~~Multicore Parallelization~~, ~~Tiled MatMul~~, ~~Decode-Path GEMV~~, ~~float16 kernels~~, ~~Graph Execution Engine~~ and ~~Native fp16 FMA (FMLAL)~~: shipped.
*   **Attention + KV cache:** the next inference-graph operator; the arena planner already provides the memory discipline a bounded KV cache needs.
*   **int8 / Quantized Kernels:** the dtype slot exists; kernels and dequantization paths do not.
*   **Weight Loading:** import pretransposed fp16 weights from file (safetensors-style) to feed `Graph::add_param`.
*   **Kernel Fusion in the Graph:** fuse LayerNorm→GEMV and bias-add into GELU within a node to skip round-trips to memory between producers and consumers.
