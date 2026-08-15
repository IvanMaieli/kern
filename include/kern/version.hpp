//
// Created by Ivan Maieli on 13/08/2026.
//

#pragma once

#include <string_view>

namespace kern {
    struct Version {
        int major;
        int minor;
        int patch;
    };
    [[nodiscard]] Version version() noexcept;
    [[nodiscard]] std::string_view version_string() noexcept;
}