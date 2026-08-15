# Kern Inference Engine

Kern is a lightweight C++ inference engine, designed to deliver high performance on Apple Silicon architectures. The architecture is built around rigorous memory management and optimized core mathematical operations essential for small language models.

## Design Philosophy

The project follows three fundamental pillars:

1.  **Deterministic Memory Management:** We use the RAII (Resource Acquisition Is Initialization) paradigm to eliminate memory leaks and fragmentation. Through a custom linear Memory Pool, we avoid costly dynamic allocations during the inference cycle.
2.  **Hardware-Aware Performance:** The engine is optimized to align data to 64-byte boundaries, adhering to the requirements of cache lines and SIMD units on Apple Silicon.
3.  **Modular Architecture:** The codebase strictly separates data structures (Tensors) from operational behavior (Operators), ensuring maintainability and ease of extension for new mathematical operations.

## Project Structure

*   `app/`: Contains sample applications and the main executable.
*   `include/kern/`: Public headers defining the interfaces (`Tensor`, `Shape`, `MemoryPool`).
*   `src/`: Implementation of system logic.
*   `tests/`: Unit test suite ensuring the correctness of each module.
*   `cmake/`: Build support scripts.

## Requirements

*   Compiler supporting C++20 (e.g., Clang on macOS).
*   CMake 3.25 or higher.
*   Ninja (recommended for build speed).

## Build Instructions

The project uses CMake to manage the build process.

1.  **Configuration:**
    From the project root directory, create a build directory and configure the project:

    ```bash
    mkdir build && cd build
    cmake .. -G Ninja
    ```

2.  **Compilation:**
    To compile the library and the test executable:

    ```bash
    ninja
    ```

3.  **Running Tests:**
    To verify that every module functions correctly:

    ```bash
    ./kern-tests
    ```

## Usage

Below is an example of how to use Kern to create tensors and apply an addition operation via the Memory Pool:

```cpp
#include <kern/tensor.hpp>
#include <kern/memory_pool.hpp>
#include <kern/ops.hpp>

// ... 

kern::MemoryPool pool(1024);
kern::Shape shape{2, 2};

kern::Tensor t1(shape, kern::DataType::float32, pool);
kern::Tensor t2(shape, kern::DataType::float32, pool);
kern::Tensor out(shape, kern::DataType::float32, pool);

// Computation: out = t1 + t2
kern::ops::Add(t1, t2, out);
```

## Developer Notes

The allocation system utilizes `[[nodiscard]]` to ensure that results from memory operations and data accessors are not accidentally ignored, preventing common bugs related to data ownership. Each new operator should be implemented as a function within the `kern::ops` namespace, accepting the output tensor by reference to allow end-users full control over memory resources.
