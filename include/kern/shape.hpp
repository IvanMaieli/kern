//
// Created by Ivan Maieli on 14/08/2026.
//

#pragma once

#include <array>
#include <cstddef>
#include <initializer_list>

namespace kern {
    class Shape {
    public:
        using Dimension = std::size_t;
        static constexpr std::size_t maximum_rank = 4;

        Shape() noexcept = default;
        Shape(std::initializer_list<Dimension> dimensions);

        [[nodiscard]] std::size_t rank() const noexcept;
        [[nodiscard]] bool empty() const noexcept;
        [[nodiscard]] Dimension dimension(std::size_t axis) const;
        [[nodiscard]] std::size_t element_count() const;

    private:
        std::array<Dimension, maximum_rank> dimensions_{};
        std::size_t rank_{0};
    };

} // namespace kern