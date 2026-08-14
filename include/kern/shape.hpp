//
// Created by Ivan Maieli on 14/08/2026.
//

#pragma once

#pragma once

#include <cstddef>
#include <initializer_list>
#include <vector>

namespace kern {
    class Shape {
    public:
        Shape() = default;
        Shape(std::initializer_list<std::size_t> dimensions);

        [[nodiscard]] std::size_t rank() const noexcept;
        [[nodiscard]] bool empty() const noexcept;
        [[nodiscard]] std::size_t dimension(std::size_t axis) const;
        [[nodiscard]] std::size_t element_count() const;
    private:
        std::vector<std::size_t> dimensions_;
    };

} // namespace kern


