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
    CHECK(matrix.is_contiguous());

    // Views: swapping axes must permute the existing strides, not recompute
    // contiguous ones, or the shape no longer describes the buffer layout.
    kern::Shape view{2, 3};
    view.swap_dimensions(0, 1);
    CHECK(view.dimension(0) == 3);
    CHECK(view.dimension(1) == 2);
    CHECK(view.stride(0) == 1); // was stride(1) of {2,3}
    CHECK(view.stride(1) == 3); // was stride(0) of {2,3}
    CHECK(!view.is_contiguous());

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
