#include <kern/version.hpp>
#include "test_macros.hpp"
#include "tests.hpp"
#include <string_view>

void test_version() {
    const auto [major, minor, patch] = kern::version();

    CHECK(major == 0);
    CHECK(minor == 1);
    CHECK(patch == 0);
    CHECK(kern::version_string() == std::string_view{"0.1.0"});
    CHECK(!kern::version_string().empty());
}
