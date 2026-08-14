//
// Created by Ivan Maieli on 13/08/2026.
//

#include <kern/version.hpp>
#include "kern/shape.hpp"
#include <iostream>


int main() {
    std::cout << "Kern " << kern::version_string() << '\n';
    std::cout << "Small models, close to the hardware.\n";

    const kern::Shape shape{2, 3, 4};
    std::cout << "Rank: " << shape.rank() << "\n"; // 3
    std::cout << "Dimension: " << shape.dimension(0) << "\n"; // 2
    std::cout << "Element count: " << shape.element_count() << "\n"; // 24
    return 0;
}
