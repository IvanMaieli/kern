//
// Created by Ivan Maieli on 14/08/2026.
//

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
        compute_strides();
    }

    Shape::Shape(const std::vector<Dimension>& dimensions) {
        if (dimensions.size() > maximum_rank)
            throw std::invalid_argument{"Shape rank exceeds the maximum supported rank"};

        rank_ = dimensions.size();
        for (std::size_t axis = 0; axis < rank_; ++axis)
            dimensions_[axis] = dimensions[axis];
        compute_strides();
    }

    void Shape::compute_strides() {
        std::size_t s = 1;
        for (std::size_t i = rank_; i > 0; --i) {
            const std::size_t axis = i - 1;
            strides_[axis] = s;
            s *= dimensions_[axis];
        }
    }

    Shape::Dimension Shape::dimension(const std::size_t axis) const {
        if (axis >= rank_)
            throw std::out_of_range{"Shape axis is out of range"};
        return dimensions_[axis];
    }

    Shape::Dimension Shape::stride(const std::size_t axis) const {
        if (axis >= rank_)
            throw std::out_of_range{"Shape axis is out of range"};
        return strides_[axis];
    }

    std::size_t Shape::element_count() const {
        if (empty()) return 1;
        std::size_t count = 1;
        for (std::size_t axis = 0; axis < rank_; ++axis)
            count *= dimensions_[axis];
        return count;
    }

    bool Shape::operator==(const Shape& other) const noexcept {
        if (rank_ != other.rank_) return false;
        for (std::size_t i = 0; i < rank_; ++i)
            if (dimensions_[i] != other.dimensions_[i]) return false;
        return true;
    }

    std::size_t Shape::linear_index(const std::array<Dimension, maximum_rank>& coords) const {
        std::size_t index = 0;
        for (std::size_t axis = 0; axis < rank_; ++axis)
            index += coords[axis] * strides_[axis];
        return index;
    }

    void Shape::swap_dimensions(std::size_t axis1, std::size_t axis2) {
        if (axis1 >= rank_ || axis2 >= rank_)
            throw std::out_of_range{"Shape axis is out of range"};

        std::swap(dimensions_[axis1], dimensions_[axis2]);
        // Swap strides instead of recomputing them: the buffer is unchanged,
        // so the view must keep mapping coordinates to the same elements.
        std::swap(strides_[axis1], strides_[axis2]);
    }

    void Shape::apply_permutation(const std::vector<std::size_t>& order) {
        if (order.size() != rank_)
            throw std::invalid_argument{"Permutation order must match rank."};

        std::array<Dimension, maximum_rank> new_dims{};
        std::array<Dimension, maximum_rank> new_strides{};
        std::array<bool, maximum_rank> seen{};
        for (std::size_t i = 0; i < rank_; ++i) {
            const std::size_t source = order[i];
            if (source >= rank_)
                throw std::out_of_range{"Invalid axis in permutation order."};
            if (seen[source])
                throw std::invalid_argument{"Permutation order must not repeat an axis."};
            seen[source] = true;
            new_dims[i] = dimensions_[source];
            new_strides[i] = strides_[source];
        }
        dimensions_ = new_dims;
        strides_ = new_strides;
    }

    bool Shape::is_contiguous() const noexcept {
        // Row-major layout: matches what compute_strides() would produce.
        // Size-1 axes are exempt because their coordinate is always 0.
        std::size_t expected = 1;
        for (std::size_t i = rank_; i > 0; --i) {
            const std::size_t axis = i - 1;
            if (dimensions_[axis] != 1 && strides_[axis] != expected)
                return false;
            expected *= dimensions_[axis];
        }
        return true;
    }

} // namespace kern

