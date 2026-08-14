//
// Created by Ivan Maieli on 13/08/2026.
//

#include "kern/version.hpp"
#include <string_view>

namespace kern {
    Version version() noexcept {
        return Version{.major = 0, .minor = 1, .patch = 0};
    }

    std::string_view version_string() noexcept {
        return "0.1.0";
    }
}
