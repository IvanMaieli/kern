#include <kern/ops.hpp>
#include <stdexcept>
#include <vector>
#include <algorithm>
#include <cmath>
#include <arm_neon.h>
#include <thread>

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
        if (GetBroadcastShape(a.shape(), b.shape()) != out.shape())
             throw std::invalid_argument("Tensors shapes are not broadcastable to out.");

        if (a.dtype() != b.dtype() || a.dtype() != out.dtype())
            throw std::invalid_argument("Tensors must have the same dtype for Add operation.");

        // Fast SIMD path: requires contiguous and non-aliased buffers for safe vectorization
        if (a.dtype() == DataType::float32 && a.shape() == b.shape() && a.shape() == out.shape() &&
            !Tensor::is_aliased(a, out) && !Tensor::is_aliased(b, out)) {
            auto a_ptr = static_cast<const float*>(a.data());
            auto b_ptr = static_cast<const float*>(b.data());
            auto* __restrict out_ptr = static_cast<float*>(out.data());
            const size_t n = out.shape().element_count();

            size_t i = 0;
            for (; i + 3 < n; i += 4) {
                const float32x4_t va = vld1q_f32(a_ptr + i);
                const float32x4_t vb = vld1q_f32(b_ptr + i);
                const float32x4_t vout = vaddq_f32(va, vb);
                vst1q_f32(out_ptr + i, vout);
            }
            for (; i < n; ++i) out_ptr[i] = a_ptr[i] + b_ptr[i];

        } 
 
 

        else if (a.dtype() == DataType::float32) {
            const auto a_ptr = static_cast<const float*>(a.data());
            const auto b_ptr = static_cast<const float*>(b.data());
            auto* __restrict out_ptr = static_cast<float*>(out.data());
            const size_t n = out.shape().element_count();
            
            for (size_t i = 0; i < n; ++i) {
                auto coords = get_coords(i, out.shape());
                
                std::array<Shape::Dimension, Shape::maximum_rank> a_coords{};
                std::array<Shape::Dimension, Shape::maximum_rank> b_coords{};
                
                for(size_t axis = 0; axis < a.shape().rank(); ++axis) {
                    const size_t out_axis = axis + (out.shape().rank() - a.shape().rank());
                    a_coords[axis] = (a.shape().dimension(axis) == 1) ? 0 : coords[out_axis];
                }
                for(size_t axis = 0; axis < b.shape().rank(); ++axis) {
                    const size_t out_axis = axis + (out.shape().rank() - b.shape().rank());
                    b_coords[axis] = (b.shape().dimension(axis) == 1) ? 0 : coords[out_axis];
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

    void MatMul(const Tensor& a, const Tensor& b, Tensor& out) {
        if (a.shape().rank() != 2 || b.shape().rank() != 2 || out.shape().rank() != 2)
            throw std::invalid_argument("MatMul supports only 2D tensors for now.");

        const size_t M = a.shape().dimension(0);
        const size_t K = a.shape().dimension(1);
        const size_t K_b = b.shape().dimension(0);
        const size_t N = b.shape().dimension(1);

        if (K != K_b)
            throw std::invalid_argument("MatMul: Inner dimensions must match.");
        if (M != out.shape().dimension(0) || N != out.shape().dimension(1))
            throw std::invalid_argument("MatMul: Output shape is incompatible.");

        if (a.dtype() == DataType::float32) {
            const auto a_ptr = static_cast<const float*>(a.data());
            auto b_ptr = static_cast<const float*>(b.data());
            auto* __restrict out_ptr = static_cast<float*>(out.data());

            unsigned int num_threads = std::thread::hardware_concurrency();
            if (num_threads == 0) num_threads = 2;
            std::vector<std::jthread> threads;

            // Parallelize over M rows
            const size_t rows_per_thread = (M + num_threads - 1) / num_threads;

            for (unsigned int t = 0; t < num_threads; ++t) {
                const size_t m_start = t * rows_per_thread;
                const size_t m_end = std::min(m_start + rows_per_thread, M);

                if (m_start >= m_end) break;

                threads.emplace_back([=, &a, &b, &out]() {
                    // Tiled + SIMD MatMul implementation for a range of m
                    for (size_t m_tile = m_start; m_tile < m_end; m_tile += TILE_SIZE) {
                        for (size_t k_tile = 0; k_tile < K; k_tile += TILE_SIZE) {
                            for (size_t n_tile = 0; n_tile < N; n_tile += TILE_SIZE) {

                                for (size_t m = m_tile; m < std::min(m_tile + TILE_SIZE, m_end); ++m) {
                                    for (size_t k = k_tile; k < std::min(k_tile + TILE_SIZE, K); ++k) {
                                        const float a_val = a_ptr[a.shape().linear_index({m, k})];
                                        const float32x4_t va_val = vdupq_n_f32(a_val);

                                        size_t n = n_tile;
                                        const size_t n_end = std::min(n_tile + TILE_SIZE, N);

                                        // Vectorized loop over n - Unrolled by 2
                                        for (; n + 7 < n_end; n += 8) {
                                            // Pair 1
                                            const float32x4_t vb1 = vld1q_f32(b_ptr + b.shape().linear_index({k, n}));
                                            float32x4_t vout1 = vld1q_f32(out_ptr + out.shape().linear_index({m, n}));
                                            vout1 = vfmaq_f32(vout1, va_val, vb1);
                                            vst1q_f32(out_ptr + out.shape().linear_index({m, n}), vout1);

                                            // Pair 2
                                            const float32x4_t vb2 = vld1q_f32(b_ptr + b.shape().linear_index({k, n + 4}));
                                            float32x4_t vout2 = vld1q_f32(out_ptr + out.shape().linear_index({m, n + 4}));
                                            vout2 = vfmaq_f32(vout2, va_val, vb2);
                                            vst1q_f32(out_ptr + out.shape().linear_index({m, n + 4}), vout2);
                                        }
                                        // Cleanup loop
                                        for (; n < n_end; ++n) {
                                            out_ptr[out.shape().linear_index({m, n})] += a_val * 
                                                   b_ptr[b.shape().linear_index({k, n})];
                                        }
                                    }
                                }
                            }
                        }
                    }
                });
            }
        } else throw std::runtime_error("Unsupported dtype for MatMul operation.");
    }

    void ReLU(const Tensor& in, Tensor& out) {
        if (!(in.shape() == out.shape()))
            throw std::invalid_argument("ReLU: Shapes must match.");

        if (in.dtype() == DataType::float32) {
            const auto in_ptr = static_cast<const float*>(in.data());
            auto* __restrict out_ptr = static_cast<float*>(out.data());
            const size_t n = in.shape().element_count();
            for (size_t i = 0; i < n; ++i)
                out_ptr[i] = std::max(0.0f, in_ptr[i]);
        } else throw std::runtime_error("Unsupported dtype for ReLU operation.");
    }

    void GELU(const Tensor& in, Tensor& out) {
        if (!(in.shape() == out.shape()))
            throw std::invalid_argument("GELU: Shapes must match.");

        if (in.dtype() == DataType::float32) {
            const auto in_ptr = static_cast<const float*>(in.data());
            auto* __restrict out_ptr = static_cast<float*>(out.data());
            const size_t n = in.shape().element_count();
            const float k1 = 0.7978845608f; // sqrt(2/pi)
            const float k2 = 0.044715f;

            for (size_t i = 0; i < n; ++i) {
                const float x = in_ptr[i];
                const float x3 = x * x * x;
                const float tanh_arg = k1 * (x + k2 * x3);
                out_ptr[i] = 0.5f * x * (1.0f + std::tanh(tanh_arg));
            }
        } else throw std::runtime_error("Unsupported dtype for GELU operation.");
    }

    void Softmax(const Tensor& in, Tensor& out) {
        if (!(in.shape() == out.shape()))
            throw std::invalid_argument("Softmax: Shapes must match.");

        const size_t rank = in.shape().rank();
        const size_t last_dim = in.shape().dimension(rank - 1);
        const size_t num_rows = in.shape().element_count() / last_dim;

        const auto in_ptr = static_cast<const float*>(in.data());
        auto* __restrict out_ptr = static_cast<float*>(out.data());

        for (size_t i = 0; i < num_rows; ++i) {
            const float* row_in = in_ptr + i * last_dim;
            float* row_out = out_ptr + i * last_dim;

            float max_val = -std::numeric_limits<float>::infinity();
            for (size_t j = 0; j < last_dim; ++j) max_val = std::max(max_val, row_in[j]);

            float sum = 0.0f;
            for (size_t j = 0; j < last_dim; ++j) {
                row_out[j] = std::exp(row_in[j] - max_val);
                sum += row_out[j];
            }

            for (size_t j = 0; j < last_dim; ++j) row_out[j] /= sum;
        }
    }

    void LayerNorm(const Tensor& in, Tensor& out) {
        if (!(in.shape() == out.shape()))
            throw std::invalid_argument("LayerNorm: Shapes must match.");

        const size_t rank = in.shape().rank();
        const size_t last_dim = in.shape().dimension(rank - 1);
        const size_t num_rows = in.shape().element_count() / last_dim;

        const auto in_ptr = static_cast<const float*>(in.data());
        auto* __restrict out_ptr = static_cast<float*>(out.data());

        const float eps = 1e-5f;

        for (size_t i = 0; i < num_rows; ++i) {
            const float* row_in = in_ptr + i * last_dim;
            float* row_out = out_ptr + i * last_dim;

            float mean = 0.0f;
            for (size_t j = 0; j < last_dim; ++j) mean += row_in[j];
            mean /= static_cast<float>(last_dim);

            float var = 0.0f;
            for (size_t j = 0; j < last_dim; ++j) {
                const float diff = row_in[j] - mean;
                var += diff * diff;
            }
            var /= static_cast<float>(last_dim);

            const float inv_std = 1.0f / std::sqrt(var + eps);
            for (size_t j = 0; j < last_dim; ++j)
                row_out[j] = (row_in[j] - mean) * inv_std;
        }
    }
} // namespace kern::ops
