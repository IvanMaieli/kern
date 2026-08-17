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
        compute_strides();
    }

    void Shape::apply_permutation(const std::vector<std::size_t>& order) {
        if (order.size() != rank_)
            throw std::invalid_argument{"Permutation order must match rank."};

        std::array<Dimension, maximum_rank> new_dims;
        for (std::size_t i = 0; i < rank_; ++i) {
            if (order[i] >= rank_)
                throw std::out_of_range{"Invalid axis in permutation order."};
            new_dims[i] = dimensions_[order[i]];
        }
        dimensions_ = new_dims;
        compute_strides();
    }

} // namespace kern

