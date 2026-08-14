#include "tests.hpp"
#include <cstdio>
#include <cstdlib>

int main() {
    test_version();
    test_shape();
    test_dtype();
    test_tensor_basics();
    test_tensor_alignment();
    test_tensor_move();

    if (failures != 0) {
        std::fprintf(stderr, "%d check(s) failed.\n", failures);
        return EXIT_FAILURE;
    }

    std::printf("All checks passed.\n");
    return EXIT_SUCCESS;
}


