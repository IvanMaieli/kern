//
// Created by Ivan Maieli on 14/08/2026.
//

#include <kern/shape.hpp>
#include <limits>
#include <stdexcept>

namespace kern {

    Shape::Shape(std::initializer_list<std::size_t> dimensions)
        : dimensions_{dimensions} {
    }

    std::size_t Shape::rank() const noexcept {
        return dimensions_.size();
    }

    bool Shape::empty() const noexcept {
        return dimensions_.empty();
    }

    std::size_t Shape::dimension(const std::size_t axis) const {
        return dimensions_.at(axis);
    }

    std::size_t Shape::element_count() const {
        for (const std::size_t dimension : dimensions_) if (dimension == 0) return 0;

        std::size_t count = 1;
        for (const std::size_t dimension : dimensions_) {
            if (constexpr std::size_t maximum = std::numeric_limits<std::size_t>::max();
                count > maximum / dimension) {
                throw std::overflow_error{"Shape element count overflow"};
                }
            count *= dimension;
        }
        return count;
    }
} // namespace kern