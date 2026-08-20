//
// Created by Ivan Maieli on 14/08/2026.
//

#pragma once

#include <array>
#include <cstddef>
#include <initializer_list>
#include <vector>

namespace kern {
    class Shape {
    public:
        using Dimension = std::size_t;
        static constexpr std::size_t maximum_rank = 4;

        Shape() noexcept = default;
        Shape(std::initializer_list<Dimension> dimensions);
        explicit Shape(const std::vector<Dimension>& dimensions);

        [[nodiscard]] std::size_t rank() const noexcept { return rank_; }
        [[nodiscard]] bool empty() const noexcept { return rank_ == 0; }
        [[nodiscard]] Dimension dimension(std::size_t axis) const;
        [[nodiscard]] std::size_t element_count() const;
        [[nodiscard]] bool operator==(const Shape& other) const noexcept;

        // Strides methods
        [[nodiscard]] Dimension stride(std::size_t axis) const;
        [[nodiscard]] std::size_t linear_index(const std::array<Dimension, maximum_rank>& coords) const;
        [[nodiscard]] bool is_contiguous() const noexcept;

        // Metadata-only view operations: dimensions and strides are updated
        // together so the shape keeps describing the same buffer layout.
        void swap_dimensions(std::size_t axis1, std::size_t axis2);
        void apply_permutation(const std::vector<std::size_t>& order);

    private:
        std::array<Dimension, maximum_rank> dimensions_{};
        std::array<Dimension, maximum_rank> strides_{};
        std::size_t rank_{0};
        void compute_strides();
    };

} // namespace kern
