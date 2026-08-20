#include <kern/ops.hpp>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include <arm_neon.h>
#include <thread>
#include <mutex>
#include <condition_variable>

namespace kern::ops {

    static ThreadPool& GetGlobalThreadPool() {
        static ThreadPool pool(std::thread::hardware_concurrency());
        return pool;
    }

    ThreadPool& GetThreadPool() {
        return GetGlobalThreadPool();
    }

    // Kernels that iterate data pointers flat (row-major) must reject views
    // with permuted strides instead of silently reading the wrong elements.
    static void require_contiguous(const Tensor& tensor, const std::string& op_name) {
        if (!tensor.shape().is_contiguous())
            throw std::invalid_argument(op_name + ": tensor must be contiguous.");
    }

    Shape GetBroadcastShape(const Shape& a, const Shape& b) {
        const size_t rank_a = a.rank();
        const size_t rank_b = b.rank();
        const size_t max_rank = std::max(rank_a, rank_b);
        std::vector<Shape::Dimension> new_dims(max_rank);
        for (size_t i = 0; i < max_rank; ++i) {
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

    void Add(const Tensor& a, const Tensor& b, Tensor& out) {
        if (GetBroadcastShape(a.shape(), b.shape()) != out.shape())
             throw std::invalid_argument("Tensors shapes are not broadcastable to out.");
        if (a.dtype() != b.dtype() || a.dtype() != out.dtype())
            throw std::invalid_argument("Tensors must have the same dtype for Add operation.");
        // The broadcast path below walks the operands through their own
        // strides, but out is written flat, so only the output must be
        // contiguous. In-place is only safe when the aliased operand is walked
        // in exactly the output's order (same shape, contiguous).
        require_contiguous(out, "Add");
        if (Tensor::is_aliased(a, out) && !(a.shape() == out.shape() && a.shape().is_contiguous()))
            throw std::invalid_argument("Add: in-place use requires the aliased input to match out.");
        if (Tensor::is_aliased(b, out) && !(b.shape() == out.shape() && b.shape().is_contiguous()))
            throw std::invalid_argument("Add: in-place use requires the aliased input to match out.");

        if (a.dtype() == DataType::float32 && a.shape() == b.shape() && a.shape() == out.shape() &&
            a.shape().is_contiguous() && b.shape().is_contiguous() &&
            !Tensor::is_aliased(a, out) && !Tensor::is_aliased(b, out)) {
            const float* __restrict a_ptr = static_cast<const float*>(a.data());
            const float* __restrict b_ptr = static_cast<const float*>(b.data());
            float* __restrict out_ptr = static_cast<float*>(out.data());
            const size_t n = out.shape().element_count();
            size_t i = 0;
            for (; i + 3 < n; i += 4) {
                float32x4_t va = vld1q_f32(a_ptr + i);
                float32x4_t vb = vld1q_f32(b_ptr + i);
                float32x4_t vout = vaddq_f32(va, vb);
                vst1q_f32(out_ptr + i, vout);
            }
            for (; i < n; ++i) out_ptr[i] = a_ptr[i] + b_ptr[i];
        } else if (a.dtype() == DataType::float32) {
            const float* __restrict a_ptr = static_cast<const float*>(a.data());
            const float* __restrict b_ptr = static_cast<const float*>(b.data());
            float* __restrict out_ptr = static_cast<float*>(out.data());

            const Shape& out_shape = out.shape();
            const size_t rank = out_shape.rank();
            if (out_shape.element_count() == 0) return;
            if (rank == 0) { // both operands are scalars
                out_ptr[0] = a_ptr[0] + b_ptr[0];
                return;
            }

            // Stride-zero broadcasting: right-align each operand against the
            // output rank and set its stride to 0 on broadcast axes. A zero
            // stride "freezes" the operand pointer on that axis, so the loops
            // below need no per-element div/mod or coordinate clamping.
            std::array<Shape::Dimension, Shape::maximum_rank> strides_a{};
            std::array<Shape::Dimension, Shape::maximum_rank> strides_b{};
            for (size_t axis = 0; axis < a.shape().rank(); ++axis) {
                const size_t out_axis = axis + (rank - a.shape().rank());
                strides_a[out_axis] = (a.shape().dimension(axis) == 1) ? 0 : a.shape().stride(axis);
            }
            for (size_t axis = 0; axis < b.shape().rank(); ++axis) {
                const size_t out_axis = axis + (rank - b.shape().rank());
                strides_b[out_axis] = (b.shape().dimension(axis) == 1) ? 0 : b.shape().stride(axis);
            }

            // Walk the output with an odometer over the leading axes; the
            // last axis is the inner loop, vectorized when both operands have
            // unit stride there.
            const size_t last = rank - 1;
            const size_t inner = out_shape.dimension(last);
            const size_t outer = out_shape.element_count() / inner;
            const size_t stride_a = strides_a[last];
            const size_t stride_b = strides_b[last];

            std::array<Shape::Dimension, Shape::maximum_rank> coords{};
            for (size_t o = 0; o < outer; ++o) {
                size_t offset_a = 0;
                size_t offset_b = 0;
                for (size_t axis = 0; axis < last; ++axis) {
                    offset_a += coords[axis] * strides_a[axis];
                    offset_b += coords[axis] * strides_b[axis];
                }
                const float* pa = a_ptr + offset_a;
                const float* pb = b_ptr + offset_b;
                float* po = out_ptr + o * inner;

                if (stride_a == 1 && stride_b == 1) {
                    size_t j = 0;
                    for (; j + 3 < inner; j += 4) {
                        const float32x4_t va = vld1q_f32(pa + j);
                        const float32x4_t vb = vld1q_f32(pb + j);
                        vst1q_f32(po + j, vaddq_f32(va, vb));
                    }
                    for (; j < inner; ++j) po[j] = pa[j] + pb[j];
                } else {
                    for (size_t j = 0; j < inner; ++j)
                        po[j] = pa[j * stride_a] + pb[j * stride_b];
                }

                // Odometer carry across the leading axes.
                for (size_t axis = last; axis > 0; --axis) {
                    const size_t dd = axis - 1;
                    if (++coords[dd] < out_shape.dimension(dd)) break;
                    coords[dd] = 0;
                }
            }
        } else throw std::runtime_error("Unsupported dtype for Add operation.");
    }

    void Reshape(const Tensor& in, const Shape& new_shape, Tensor& out) {
        if (in.shape().element_count() != new_shape.element_count())
            throw std::invalid_argument("Reshape: Total element count must match.");
        require_contiguous(in, "Reshape");
        out.set_shape(new_shape);
        out.set_buffer(in.buffer());
    }

    void Transpose(const Tensor& in, const std::size_t axis1, const std::size_t axis2, Tensor& out) {
        if (axis1 >= in.shape().rank() || axis2 >= in.shape().rank())
            throw std::out_of_range("Transpose: axis out of range.");
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
        
        if (K != K_b) throw std::invalid_argument("MatMul: Inner dimensions must match.");
        if (M != out.shape().dimension(0) || N != out.shape().dimension(1))
            throw std::invalid_argument("MatMul: Output shape is incompatible.");
        require_contiguous(a, "MatMul");
        require_contiguous(b, "MatMul");
        require_contiguous(out, "MatMul");
        if (Tensor::is_aliased(a, out) || Tensor::is_aliased(b, out))
            throw std::invalid_argument("MatMul: output must not alias an input.");

        if (a.dtype() == DataType::float32) {
            const float* __restrict a_ptr = static_cast<const float*>(a.data());
            const float* __restrict b_ptr = static_cast<const float*>(b.data());
            float* __restrict out_ptr = static_cast<float*>(out.data());

            unsigned int num_threads = std::thread::hardware_concurrency();
            if (num_threads == 0) num_threads = 2;
            std::condition_variable cv; std::mutex mtx; size_t tasks_completed = 0;
            size_t rows_per_thread = (M + num_threads - 1) / num_threads;
            unsigned int actual_tasks = 0;
            for (unsigned int t = 0; t < num_threads; ++t) {
                size_t m_start = t * rows_per_thread;
                size_t m_end = std::min(m_start + rows_per_thread, M);
                if (m_start >= m_end) break;
                actual_tasks++;
            }
            for (unsigned int t = 0; t < actual_tasks; ++t) {
                size_t m_start = t * rows_per_thread;
                size_t m_end = std::min(m_start + rows_per_thread, M);
                GetThreadPool().enqueue([=, &tasks_completed, &cv, &mtx]() {
                    // M-N-K tiling for better cache locality
                    for (size_t m_tile = m_start; m_tile < m_end; m_tile += TILE_SIZE) {
                        for (size_t n_tile = 0; n_tile < N; n_tile += TILE_SIZE) {
                            const size_t m_max = std::min(m_tile + TILE_SIZE, m_end);
                            const size_t n_max = std::min(n_tile + TILE_SIZE, N);

                            // The k-tile passes below accumulate into out, so the
                            // tile must start from zero: the caller's buffer may
                            // hold a previous result or garbage (aligned_alloc
                            // does not zero-initialize).
                            for (size_t m = m_tile; m < m_max; ++m)
                                std::fill(out_ptr + m * N + n_tile, out_ptr + m * N + n_max, 0.0f);

                            for (size_t k_tile = 0; k_tile < K; k_tile += TILE_SIZE) {
                                const size_t k_max = std::min(k_tile + TILE_SIZE, K);

                                for (size_t m = m_tile; m < m_max; ++m) {
                                    const float* a_row = a_ptr + m * K;
                                    float* out_row = out_ptr + m * N;
                                    for (size_t k = k_tile; k < k_max; ++k) {
                                        const float a_val = a_row[k];
                                        const float32x4_t va_val = vdupq_n_f32(a_val);
                                        const float* b_row = b_ptr + k * N;
                                        
                                        size_t n = n_tile;
                                        for (; n + 7 < n_max; n += 8) {
                                            float32x4_t vb1 = vld1q_f32(b_row + n);
                                            float32x4_t vout1 = vld1q_f32(out_row + n);
                                            vout1 = vfmaq_f32(vout1, va_val, vb1);
                                            vst1q_f32(out_row + n, vout1);
                                            
                                            float32x4_t vb2 = vld1q_f32(b_row + n + 4);
                                            float32x4_t vout2 = vld1q_f32(out_row + n + 4);
                                            vout2 = vfmaq_f32(vout2, va_val, vb2);
                                            vst1q_f32(out_row + n + 4, vout2);
                                        }
                                        for (; n < n_max; ++n) out_row[n] += a_val * b_row[n];
                                    }
                                }
                            }
                        }
                    }
                    std::lock_guard<std::mutex> lock(mtx);
                    if (++tasks_completed == actual_tasks) cv.notify_one();
                });
            }
            std::unique_lock<std::mutex> lock(mtx);
            cv.wait(lock, [&] { return tasks_completed == actual_tasks; });
        } else throw std::runtime_error("Unsupported dtype for MatMul operation.");
    }

    void MatMulTransposed(const Tensor& a, const Tensor& b_t, Tensor& out) {
        if (a.shape().rank() != 2 || b_t.shape().rank() != 2 || out.shape().rank() != 2)
            throw std::invalid_argument("MatMulTransposed supports only 2D tensors.");
        const size_t M = a.shape().dimension(0);
        const size_t K = a.shape().dimension(1);
        const size_t N = b_t.shape().dimension(0);
        const size_t K_b = b_t.shape().dimension(1);
        if (K != K_b) throw std::invalid_argument("MatMulTransposed: Inner dimensions must match.");
        if (M != out.shape().dimension(0) || N != out.shape().dimension(1))
            throw std::invalid_argument("MatMulTransposed: Output shape is incompatible.");
        require_contiguous(a, "MatMulTransposed");
        require_contiguous(b_t, "MatMulTransposed");
        require_contiguous(out, "MatMulTransposed");
        if (Tensor::is_aliased(a, out) || Tensor::is_aliased(b_t, out))
            throw std::invalid_argument("MatMulTransposed: output must not alias an input.");

        if (a.dtype() == DataType::float32) {
            const float* __restrict a_ptr = static_cast<const float*>(a.data());
            const float* __restrict b_ptr = static_cast<const float*>(b_t.data());
            float* __restrict out_ptr = static_cast<float*>(out.data());
            unsigned int num_threads = std::thread::hardware_concurrency();
            if (num_threads == 0) num_threads = 2;
            std::condition_variable cv; std::mutex mtx; size_t tasks_completed = 0;
            size_t rows_per_thread = (M + num_threads - 1) / num_threads;
            unsigned int actual_tasks = 0;
            for (unsigned int t = 0; t < num_threads; ++t) {
                size_t m_start = t * rows_per_thread;
                size_t m_end = std::min(m_start + rows_per_thread, M);
                if (m_start >= m_end) break;
                actual_tasks++;
            }
            for (unsigned int t = 0; t < actual_tasks; ++t) {
                size_t m_start = t * rows_per_thread;
                size_t m_end = std::min(m_start + rows_per_thread, M);
                GetThreadPool().enqueue([=, &tasks_completed, &cv, &mtx]() {
                    // Contiguity is guaranteed by the guards above, so plain
                    // row pointers replace the per-access linear_index calls.
                    for (size_t m = m_start; m < m_end; ++m) {
                        const float* a_row = a_ptr + m * K;
                        float* out_row = out_ptr + m * N;
                        for (size_t n = 0; n < N; ++n) {
                            const float* b_row = b_ptr + n * K;
                            float32x4_t acc = vdupq_n_f32(0.0f);
                            size_t k = 0;
                            for (; k + 3 < K; k += 4)
                                acc = vfmaq_f32(acc, vld1q_f32(a_row + k), vld1q_f32(b_row + k));
                            float sum = vaddvq_f32(acc);
                            for (; k < K; ++k) sum += a_row[k] * b_row[k];
                            out_row[n] = sum;
                        }
                    }
                    std::lock_guard<std::mutex> lock(mtx);
                    if (++tasks_completed == actual_tasks) cv.notify_one();
                });
            }
            std::unique_lock<std::mutex> lock(mtx);
            cv.wait(lock, [&] { return tasks_completed == actual_tasks; });
        } else throw std::runtime_error("Unsupported dtype for MatMulTransposed operation.");
    }

    void ReLU(const Tensor& in, Tensor& out) {
        if (!(in.shape() == out.shape())) throw std::invalid_argument("ReLU: Shapes must match.");
        require_contiguous(in, "ReLU");
        require_contiguous(out, "ReLU");
        if (in.dtype() == DataType::float32 && out.dtype() == DataType::float32) {
            const float* __restrict in_ptr = static_cast<const float*>(in.data());
            float* __restrict out_ptr = static_cast<float*>(out.data());
            const size_t n = in.shape().element_count();
            unsigned int num_threads = std::thread::hardware_concurrency();
            if (num_threads == 0) num_threads = 2;
            std::condition_variable cv; std::mutex mtx; size_t tasks_completed = 0;
            size_t chunk_size = (n + num_threads - 1) / num_threads;
            unsigned int actual_tasks = 0;
            for (unsigned int t = 0; t < num_threads; ++t) {
                size_t start = t * chunk_size;
                size_t end = std::min(start + chunk_size, n);
                if (start >= end) break;
                actual_tasks++;
            }
            for (unsigned int t = 0; t < actual_tasks; ++t) {
                size_t start = t * chunk_size;
                size_t end = std::min(start + chunk_size, n);
                GetThreadPool().enqueue([=, &tasks_completed, &cv, &mtx]() {
                    const float32x4_t vzero = vdupq_n_f32(0.0f);
                    size_t i = start;
                    for (; i + 3 < end; i += 4)
                        vst1q_f32(out_ptr + i, vmaxq_f32(vld1q_f32(in_ptr + i), vzero));
                    for (; i < end; ++i) out_ptr[i] = std::max(0.0f, in_ptr[i]);
                    std::lock_guard<std::mutex> lock(mtx);
                    if (++tasks_completed == actual_tasks) cv.notify_one();
                });
            }
            std::unique_lock<std::mutex> lock(mtx);
            cv.wait(lock, [&] { return tasks_completed == actual_tasks; });
        } else throw std::runtime_error("Unsupported dtype for ReLU operation.");
    }

    // Vector tanh for GELU: a [5/4] Pade approximant evaluated at y/4 (it
    // matches tanh's series through y^5), then two applications of the
    // double-angle identity tanh(2t) = 2t / (1 + t^2). Clamping the rational
    // result makes large arguments saturate to +/-1 exactly, which the
    // double-angle maps to +/-1 while compressing the error near the boundary.
    static float32x4_t neon_tanh(float32x4_t y) {
        const float32x4_t y4 = vmulq_n_f32(y, 0.25f);
        const float32x4_t y2 = vmulq_f32(y4, y4);
        // r = y4 * (y4^4 + 105*y4^2 + 945) / (15*y4^4 + 420*y4^2 + 945)
        const float32x4_t num = vfmaq_f32(vdupq_n_f32(945.0f), vaddq_f32(y2, vdupq_n_f32(105.0f)), y2);
        const float32x4_t den = vfmaq_f32(vdupq_n_f32(945.0f), y2, vfmaq_n_f32(vdupq_n_f32(420.0f), y2, 15.0f));
        const float32x4_t unit = vdupq_n_f32(1.0f);
        const float32x4_t r = vminq_f32(vmaxq_f32(vdivq_f32(vmulq_f32(y4, num), den), vdupq_n_f32(-1.0f)), unit);
        float32x4_t t = vdivq_f32(vaddq_f32(r, r), vfmaq_f32(unit, r, r));
        t = vdivq_f32(vaddq_f32(t, t), vfmaq_f32(unit, t, t));
        return t;
    }

    void GELU(const Tensor& in, Tensor& out) {
        if (!(in.shape() == out.shape())) throw std::invalid_argument("GELU: Shapes must match.");
        require_contiguous(in, "GELU");
        require_contiguous(out, "GELU");
        if (in.dtype() == DataType::float32 && out.dtype() == DataType::float32) {
            const float* __restrict in_ptr = static_cast<const float*>(in.data());
            float* __restrict out_ptr = static_cast<float*>(out.data());
            const size_t n = in.shape().element_count();
            unsigned int num_threads = std::thread::hardware_concurrency();
            if (num_threads == 0) num_threads = 2;
            std::condition_variable cv; std::mutex mtx; size_t tasks_completed = 0;
            size_t chunk_size = (n + num_threads - 1) / num_threads;
            unsigned int actual_tasks = 0;
            for (unsigned int t = 0; t < num_threads; ++t) {
                size_t start = t * chunk_size;
                size_t end = std::min(start + chunk_size, n);
                if (start >= end) break;
                actual_tasks++;
            }
            for (unsigned int t = 0; t < actual_tasks; ++t) {
                size_t start = t * chunk_size;
                size_t end = std::min(start + chunk_size, n);
                GetThreadPool().enqueue([=, &tasks_completed, &cv, &mtx]() {
                    const float k1 = 0.7978845608f;
                    const float k2 = 0.044715f;
                    const float k13 = k1 * k2; // x * (k1 + k13*x^2) == k1 * (x + k2*x^3)
                    size_t i = start;
                    for (; i + 3 < end; i += 4) {
                        const float32x4_t vx = vld1q_f32(in_ptr + i);
                        const float32x4_t vx2 = vmulq_f32(vx, vx);
                        const float32x4_t vu = vmulq_f32(vx, vfmaq_f32(vdupq_n_f32(k1), vx2, vdupq_n_f32(k13)));
                        const float32x4_t vt = neon_tanh(vu);
                        const float32x4_t vres = vmulq_f32(vx, vaddq_f32(vdupq_n_f32(1.0f), vt));
                        vst1q_f32(out_ptr + i, vmulq_n_f32(vres, 0.5f));
                    }
                    for (; i < end; ++i) {
                        const float x = in_ptr[i];
                        out_ptr[i] = 0.5f * x * (1.0f + std::tanh(k1 * (x + k2 * x * x * x)));
                    }
                    std::lock_guard<std::mutex> lock(mtx);
                    if (++tasks_completed == actual_tasks) cv.notify_one();
                });
            }
            std::unique_lock<std::mutex> lock(mtx);
            cv.wait(lock, [&] { return tasks_completed == actual_tasks; });
        } else throw std::runtime_error("Unsupported dtype for GELU operation.");
    }

    void Softmax(const Tensor& in, Tensor& out) {
        if (!(in.shape() == out.shape())) throw std::invalid_argument("Softmax: Shapes must match.");
        if (in.dtype() != DataType::float32 || out.dtype() != DataType::float32)
            throw std::runtime_error("Unsupported dtype for Softmax operation.");
        require_contiguous(in, "Softmax");
        require_contiguous(out, "Softmax");
        const size_t last_dim = in.shape().dimension(in.shape().rank() - 1);
        const size_t num_rows = in.shape().element_count() / last_dim;
        const float* __restrict in_ptr = static_cast<const float*>(in.data());
        float* __restrict out_ptr = static_cast<float*>(out.data());

        unsigned int num_threads = std::thread::hardware_concurrency();
        if (num_threads == 0) num_threads = 2;
        std::condition_variable cv; std::mutex mtx; size_t tasks_completed = 0;
        size_t rows_per_thread = (num_rows + num_threads - 1) / num_threads;
        unsigned int actual_tasks = 0;
        for (unsigned int t = 0; t < num_threads; ++t) {
            size_t start_row = t * rows_per_thread;
            size_t end_row = std::min(start_row + rows_per_thread, num_rows);
            if (start_row >= end_row) break;
            actual_tasks++;
        }
        for (unsigned int t = 0; t < actual_tasks; ++t) {
            size_t start_row = t * rows_per_thread;
            size_t end_row = std::min(start_row + rows_per_thread, num_rows);
            GetThreadPool().enqueue([=, &tasks_completed, &cv, &mtx]() {
                for (size_t i = start_row; i < end_row; ++i) {
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
                std::lock_guard<std::mutex> lock(mtx);
                if (++tasks_completed == actual_tasks) cv.notify_one();
            });
        }
        std::unique_lock<std::mutex> lock(mtx);
        cv.wait(lock, [&] { return tasks_completed == actual_tasks; });
    }

    void LayerNorm(const Tensor& in, Tensor& out) {
        if (!(in.shape() == out.shape())) throw std::invalid_argument("LayerNorm: Shapes must match.");
        if (in.dtype() != DataType::float32 || out.dtype() != DataType::float32)
            throw std::runtime_error("Unsupported dtype for LayerNorm operation.");
        require_contiguous(in, "LayerNorm");
        require_contiguous(out, "LayerNorm");
        const size_t last_dim = in.shape().dimension(in.shape().rank() - 1);
        const size_t num_rows = in.shape().element_count() / last_dim;
        const float* __restrict in_ptr = static_cast<const float*>(in.data());
        float* __restrict out_ptr = static_cast<float*>(out.data());
        const float eps = 1e-5f;

        unsigned int num_threads = std::thread::hardware_concurrency();
        if (num_threads == 0) num_threads = 2;
        std::condition_variable cv; std::mutex mtx; size_t tasks_completed = 0;
        size_t rows_per_thread = (num_rows + num_threads - 1) / num_threads;
        unsigned int actual_tasks = 0;
        for (unsigned int t = 0; t < num_threads; ++t) {
            size_t start_row = t * rows_per_thread;
            size_t end_row = std::min(start_row + rows_per_thread, num_rows);
            if (start_row >= end_row) break;
            actual_tasks++;
        }
        for (unsigned int t = 0; t < actual_tasks; ++t) {
            size_t start_row = t * rows_per_thread;
            size_t end_row = std::min(start_row + rows_per_thread, num_rows);
            GetThreadPool().enqueue([=, &tasks_completed, &cv, &mtx]() {
                for (size_t i = start_row; i < end_row; ++i) {
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
                    for (size_t j = 0; j < last_dim; ++j) row_out[j] = (row_in[j] - mean) * inv_std;
                }
                std::lock_guard<std::mutex> lock(mtx);
                if (++tasks_completed == actual_tasks) cv.notify_one();
            });
        }
        std::unique_lock<std::mutex> lock(mtx);
        cv.wait(lock, [&] { return tasks_completed == actual_tasks; });
    }
} // namespace kern::ops
