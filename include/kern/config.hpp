#pragma once
#include <cstddef>

namespace kern {
    static constexpr size_t ALIGNMENT = 64UL;
    static constexpr size_t TILE_SIZE = 32UL; // Configurable tile size for MatMul
}
