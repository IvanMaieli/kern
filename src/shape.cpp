#include <kern/shape.hpp>
#include <limits>
#include <stdexcept>

namespace kern {

    Shape::Shape(const std::initializer_list<Dimension> dimensions) {
        if (dimensions.size() > maximum_rank)
            throw std::invalid_argument{"Shape rank exceeds the maximum supported rank"};

        rank_ = dimensions.size();
        std::size_t axis = 0;
        for (const Dimension dimension : dimensions) {
            dimensions_[axis] = dimension;
            ++axis;
        }
    }

    std::size_t Shape::rank() const noexcept {
        return rank_;
    }

    bool Shape::empty() const noexcept {
        return rank_ == 0;
    }

    Shape::Dimension Shape::dimension(const std::size_t axis) const {
        if (axis >= rank_)
            throw std::out_of_range{"Shape axis is out of range"};
        return dimensions_[axis];
    }

    std::size_t Shape::element_count() const {
        if (empty()) return 1;

        std::size_t count = 1;
        for (std::size_t axis = 0; axis < rank_; ++axis) {
            const Dimension current_dimension = dimensions_[axis];
            if (current_dimension == 0) return 0;
            if (constexpr std::size_t maximum = std::numeric_limits<std::size_t>::max(); count > maximum / current_dimension)
                throw std::overflow_error{"Shape element count overflow"};
            count *= current_dimension;
        }
        return count;
    }

} // namespace kern