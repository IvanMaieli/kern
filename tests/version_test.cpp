//
// Created by Ivan Maieli on 13/08/2026.
//

#include <kern/version.hpp>
#include <iostream>

int main() {
    const auto [major, minor, patch] = kern::version();

    if (major != 0) {
        std::cerr << "Expected major version 0, got " << major << '\n';
        return 1;
    }

    if (minor != 1) {
        std::cerr << "Expected minor version 1, got " << minor << '\n';
        return 1;
    }

    if (patch != 0) {
        std::cerr << "Expected patch version 0, got " << patch << '\n';
        return 1;
    }

    if (kern::version_string() != "0.1.0") {
        std::cerr << "Expected version string 0.1.0, got " << kern::version_string() << '\n';
        return 1;
    }

    return 0;
}