#include <kern/ops.hpp>
#include <stdexcept>
#include <vector>
#include <algorithm>

namespace kern::ops {

    Shape GetBroadcastShape(const Shape& a, const Shape& b) {
        const size_t rank_a = a.rank();
        const size_t rank_b = b.rank();
        const size_t max_rank = std::max(rank_a, rank_b);
        
        std::vector<Shape::Dimension> new_dims(max_rank);
        
        for (size_t i = 0; i < max_rank; ++i) {
            // Indices starting from the end
            const int idx_a = static_cast<int>(rank_a) - 1 - static_cast<int>(i);
            const int idx_b = static_cast<int>(rank_b) - 1 - static_cast<int>(i);

            const Shape::Dimension dim_a = (idx_a >= 0) ? a.dimension(static_cast<size_t>(idx_a)) : 1;
            Shape::Dimension dim_b = (idx_b >= 0) ? b.dimension(static_cast<size_t>(idx_b)) : 1;

            if      (dim_a == dim_b)  new_dims[max_rank - 1 - i] = dim_a;
            else if (dim_a == 1)      new_dims[max_rank - 1 - i] = dim_b;
            else if (dim_b == 1)      new_dims[max_rank - 1 - i] = dim_a;
            else throw std::invalid_argument("Incompatible shapes for broadcasting.");

        }
        return Shape(new_dims);
    }

    static std::array<Shape::Dimension, Shape::maximum_rank> get_coords(size_t index, const Shape& shape) {
        std::array<Shape::Dimension, Shape::maximum_rank> coords{};
        for (int i = static_cast<int>(shape.rank()) - 1; i >= 0; --i) {
            coords[static_cast<size_t>(i)] = index % shape.dimension(static_cast<size_t>(i));
            index /= shape.dimension(static_cast<size_t>(i));
        }
        return coords;
    }

    void Add(const Tensor& a, const Tensor& b, Tensor& out) {
        // Validate if shapes broadcast to out.shape()
        if (GetBroadcastShape(a.shape(), b.shape()) == out.shape()) {
             // Continue
        } else {
             throw std::invalid_argument("Tensors shapes are not broadcastable to out.");
        }

        if (a.dtype() != b.dtype() || a.dtype() != out.dtype())
            throw std::invalid_argument("Tensors must have the same dtype for Add operation.");

        if (a.dtype() == DataType::float32) {
            const auto a_ptr = static_cast<const float*>(a.data());
            const auto b_ptr = static_cast<const float*>(b.data());
            const auto out_ptr = static_cast<float*>(out.data());
            const size_t n = out.shape().element_count();
            
            for (size_t i = 0; i < n; ++i) {
                auto coords = get_coords(i, out.shape());
                
                // Need to adjust coords for a and b if their rank is smaller
                std::array<Shape::Dimension, Shape::maximum_rank> a_coords{};
                std::array<Shape::Dimension, Shape::maximum_rank> b_coords{};
                
                for(size_t axis = 0; axis < a.shape().rank(); ++axis) {
                    a_coords[axis] = coords[axis + (out.shape().rank() - a.shape().rank())];
                }
                for(size_t axis = 0; axis < b.shape().rank(); ++axis) {
                    b_coords[axis] = coords[axis + (out.shape().rank() - b.shape().rank())];
                }
                
                out_ptr[i] = a_ptr[a.shape().linear_index(a_coords)] + b_ptr[b.shape().linear_index(b_coords)];
            }
        } else throw std::runtime_error("Unsupported dtype for Add operation.");
    }

    void Reshape(const Tensor& in, const Shape& new_shape, Tensor& out) {
        if (in.shape().element_count() != new_shape.element_count())
            throw std::invalid_argument("Reshape: Total element count must match.");
        
        out.set_shape(new_shape);
        out.set_buffer(in.buffer());
    }

    void Transpose(const Tensor& in, const std::size_t axis1, const std::size_t axis2, Tensor& out) {
        std::vector<std::size_t> order(in.shape().rank());
        for (std::size_t i = 0; i < order.size(); ++i) order[i] = i;
        std::swap(order[axis1], order[axis2]);
        Permute(in, order, out);
    }

    void Permute(const Tensor& in, const std::vector<std::size_t>& order, Tensor& out) {
        Shape new_shape = in.shape();
        new_shape.apply_permutation(order);
        
        out.set_shape(new_shape);
        out.set_buffer(in.buffer());
    }
} // namespace kern::ops
