#include <kern/dtype.hpp>
#include <kern/shape.hpp>
#include <kern/version.hpp>

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string_view>

namespace {

int failures = 0;

#define CHECK(condition)                                                        \
    do {                                                                        \
        if (!(condition)) {                                                     \
            std::fprintf(                                                       \
                stderr,                                                         \
                "CHECK failed: %s:%d: %s\n",                                    \
                __FILE__,                                                       \
                __LINE__,                                                       \
                #condition);                                                    \
            ++failures;                                                         \
        }                                                                       \
    } while (false)

void test_version() {
    const kern::Version version = kern::version();

    CHECK(version.major == 0);
    CHECK(version.minor == 1);
    CHECK(version.patch == 0);
    CHECK(kern::version_string() == std::string_view{"0.1.0"});
    CHECK(!kern::version_string().empty());
}

void test_shape() {
    const kern::Shape matrix{2, 3};

    CHECK(matrix.rank() == 2);
    CHECK(!matrix.empty());
    CHECK(matrix.dimension(0) == 2);
    CHECK(matrix.dimension(1) == 3);
    CHECK(matrix.element_count() == 6);

    constexpr kern::Shape scalar;

    CHECK(scalar.rank() == 0);
    CHECK(scalar.empty());
    CHECK(scalar.element_count() == 1);

    const kern::Shape zeroed{2, 0, 4};

    CHECK(zeroed.rank() == 3);
    CHECK(zeroed.element_count() == 0);

    bool threw = false;
    try {
        static_cast<void>(kern::Shape{1, 2, 3, 4, 5});
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);

    threw = false;
    try {
        static_cast<void>(matrix.dimension(2));
    } catch (const std::out_of_range&) {
        threw = true;
    }
    CHECK(threw);
}

static_assert(kern::size_bytes(kern::DataType::float32) == 4);
static_assert(kern::size_bytes(kern::DataType::float16) == 2);
static_assert(kern::size_bytes(kern::DataType::int8) == 1);

void test_dtype() {
    CHECK(kern::size_bytes(kern::DataType::float32) == sizeof(float));
    CHECK(kern::size_bytes(static_cast<kern::DataType>(255)) == 0);
    CHECK(kern::name(kern::DataType::float32) == std::string_view{"float32"});
    CHECK(kern::name(static_cast<kern::DataType>(255)) == std::string_view{"unknown"});
}

} // namespace

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