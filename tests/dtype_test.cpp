#include <kern/dtype.hpp>
#include "test_macros.hpp"
#include "tests.hpp"
#include <string_view>

static_assert(kern::size_bytes(kern::DataType::float32) == 4);
static_assert(kern::size_bytes(kern::DataType::float16) == 2);
static_assert(kern::size_bytes(kern::DataType::int8) == 1);

void test_dtype() {
    CHECK(kern::size_bytes(kern::DataType::float32) == sizeof(float));
    CHECK(kern::size_bytes(static_cast<kern::DataType>(255)) == 0);
    CHECK(kern::name(kern::DataType::float32) == std::string_view{"float32"});
    CHECK(kern::name(static_cast<kern::DataType>(255)) == std::string_view{"unknown"});
}
