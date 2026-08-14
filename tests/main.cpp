#include "tests.hpp"
#include <cstdio>
#include <cstdlib>

int main() {
    test_version();
    test_shape();
    test_dtype();

    if (failures != 0) {
        std::fprintf(stderr, "%d check(s) failed.\n", failures);
        return EXIT_FAILURE;
    }

    std::printf("All checks passed.\n");
    return EXIT_SUCCESS;
}

