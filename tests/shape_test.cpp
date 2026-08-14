#include <kern/shape.hpp>
#include "test_macros.hpp"
#include "tests.hpp"
#include <stdexcept>

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
