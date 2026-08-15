#include "tests.hpp"
#include <cstdio>
#include <cstdlib>

int main() {
    std::printf("Running version...\n"); test_version();
    std::printf("Running shape...\n"); test_shape();
    std::printf("Running dtype...\n"); test_dtype();
    std::printf("Running tensor basics...\n"); test_tensor_basics();
    std::printf("Running tensor alignment...\n"); test_tensor_alignment();
    std::printf("Running tensor move...\n"); test_tensor_move();
    std::printf("Running pool alignment...\n"); test_memory_pool_alignment();
    std::printf("Running pool reset...\n"); test_memory_pool_reset();
    std::printf("Running pool oom...\n"); test_memory_pool_oom();

    if (failures != 0) {
        std::fprintf(stderr, "%d check(s) failed.\n", failures);
        return EXIT_FAILURE;
    }

    std::printf("All checks passed.\n");
    return EXIT_SUCCESS;
}


