//
// Created by Ivan Maieli on 13/08/2026.
//

#include <kern/version.hpp>
#include <iostream>

int main() {
    std::cout << "Kern " << kern::version_string() << '\n';
    std::cout << "Small models, close to the hardware.\n";
    return 0;
}