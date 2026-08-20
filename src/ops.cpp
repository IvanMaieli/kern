#include <kern/ops.hpp>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <arm_neon.h>
#include <thread>
#include <latch>

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

    // Minimum work units per parallel task below which the operation runs
    // single-threaded to avoid threading overhead dominating.
    static constexpr size_t kMinChunk = 1024;

    // Split [0, total) into at-most-hardware-concurrency chunks and dispatch
    // each to the global thread pool.  Falls back to inline single-threaded
    // execution when the per-task work is below min_work.
    template <typename F>
    static void parallel_for(size_t total, size_t min_work, F body) {
        if (total == 0) return;

        unsigned int hw = std::thread::hardware_concurrency();
        if (hw == 0) hw = 2;

        const size_t chunk = (total + hw - 1) / hw;

        // Single-threaded when per-task work is too small.
        if (chunk < min_work) {
            body(0, total);
            return;
        }

        // Count actual tasks (trailing threads may have no work).
        unsigned int actual = 0;
        for (unsigned int t = 0; t < hw; ++t)
            if (t * chunk < total) ++actual;

        std::latch done(actual);

        for (unsigned int t = 0; t < actual; ++t) {
            const size_t start = t * chunk;
            const size_t end = std::min(start + chunk, total);
            GetThreadPool().enqueue([=, &done]() {
                body(start, end);
                done.count_down();
            });
        }

        done.wait();
    }

    // ------------------------------------------------------------------
    // Dtype traits: one vector loop body, two storage formats.
    //
    // float16 tensors store weights at half the memory footprint (the win:
    // every kernel here is bandwidth-bound at least partially), but the
    // arithmetic always runs in float32 lanes. Loading converts f16 -> f32,
    // storing converts back, so accumulation precision is identical to the
    // float32 kernels and only the storage changes.
    // ------------------------------------------------------------------
    struct F32Traits {
        using T = float;
        static float32x4_t load4(const T* p) { return vld1q_f32(p); }
        static void store4(T* p, float32x4_t v) { vst1q_f32(p, v); }
        static float load1(const T* p) { return *p; }
        static void store1(T* p, float v) { *p = v; }
    };

    struct F16Traits {
        using T = __fp16;
        static float32x4_t load4(const T* p) { return vcvt_f32_f16(vld1_f16(p)); }
        static void store4(T* p, float32x4_t v) { vst1_f16(p, vcvt_f16_f32(v)); }
        static float load1(const T* p) { return static_cast<float>(*p); }
        static void store1(T* p, float v) { *p = static_cast<__fp16>(v); }
    };

    template <typename Tr>
    static float dot_range(const typename Tr::T* x, const typename Tr::T* y, size_t n) {
        float32x4_t acc = vdupq_n_f32(0.0f);
        size_t k = 0;
        for (; k + 3 < n; k += 4)
            acc = vfmaq_f32(acc, Tr::load4(x + k), Tr::load4(y + k));
        float sum = vaddvq_f32(acc);
        for (; k < n; ++k) sum += Tr::load1(x + k) * Tr::load1(y + k);
        return sum;
    }

    // Element-wise a+b over a flat range. In-order processing reads each
    // element before writing it, so out aliasing a or b is safe.
    template <typename Tr>
    static void add_flat(const typename Tr::T* a, const typename Tr::T* b,
                         typename Tr::T* out, size_t n) {
        size_t i = 0;
        for (; i + 3 < n; i += 4)
            Tr::store4(out + i, vaddq_f32(Tr::load4(a + i), Tr::load4(b + i)));
        for (; i < n; ++i) Tr::store1(out + i, Tr::load1(a + i) + Tr::load1(b + i));
    }

    template <typename Tr>
    static void relu_range(const typename Tr::T* in, typename Tr::T* out,
                           size_t start, size_t end) {
        const float32x4_t vzero = vdupq_n_f32(0.0f);
        size_t i = start;
        for (; i + 3 < end; i += 4)
            Tr::store4(out + i, vmaxq_f32(Tr::load4(in + i), vzero));
        for (; i < end; ++i) Tr::store1(out + i, std::max(0.0f, Tr::load1(in + i)));
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
        const float32x4_t den = vfmaq_f32(vdupq_n_f32(945.0f), y2,
                                          vfmaq_n_f32(vdupq_n_f32(420.0f), y2, 15.0f));
        const float32x4_t unit = vdupq_n_f32(1.0f);
        const float32x4_t r = vminq_f32(vmaxq_f32(vdivq_f32(vmulq_f32(y4, num), den), vdupq_n_f32(-1.0f)), unit);
        float32x4_t t = vdivq_f32(vaddq_f32(r, r), vfmaq_f32(unit, r, r));
        t = vdivq_f32(vaddq_f32(t, t), vfmaq_f32(unit, t, t));
        return t;
    }

    template <typename Tr>
    static void gelu_range(const typename Tr::T* in, typename Tr::T* out,
                           size_t start, size_t end) {
        const float k1 = 0.7978845608f;
        const float k2 = 0.044715f;
        const float k13 = k1 * k2; // x * (k1 + k13*x^2) == k1 * (x + k2*x^3)
        size_t i = start;
        for (; i + 3 < end; i += 4) {
            const float32x4_t vx = Tr::load4(in + i);
            const float32x4_t vx2 = vmulq_f32(vx, vx);
            const float32x4_t vu = vmulq_f32(vx, vfmaq_f32(vdupq_n_f32(k1), vx2, vdupq_n_f32(k13)));
            const float32x4_t vt = neon_tanh(vu);
            const float32x4_t vres = vmulq_f32(vx, vaddq_f32(vdupq_n_f32(1.0f), vt));
            Tr::store4(out + i, vmulq_n_f32(vres, 0.5f));
        }
        for (; i < end; ++i) {
            const float x = Tr::load1(in + i);
            Tr::store1(out + i, 0.5f * x * (1.0f + std::tanh(k1 * (x + k2 * x * x * x))));
        }
    }

    template <typename Tr>
    static void softmax_rows(const typename Tr::T* in, typename Tr::T* out,
                             size_t last_dim, size_t r0, size_t r1) {
        for (size_t i = r0; i < r1; ++i) {
            const typename Tr::T* row_in = in + i * last_dim;
            typename Tr::T* row_out = out + i * last_dim;

            // Pass 1: NEON max reduction.
            float max_val = -std::numeric_limits<float>::infinity();
            size_t j = 0;
            if (last_dim >= 4) {
                float32x4_t vmax = Tr::load4(row_in);
                j = 4;
                for (; j + 3 < last_dim; j += 4)
                    vmax = vmaxq_f32(vmax, Tr::load4(row_in + j));
                max_val = vmaxvq_f32(vmax);
            }
            for (; j < last_dim; ++j) max_val = std::max(max_val, Tr::load1(row_in + j));

            // Pass 2: exp + sum (scalar: no NEON exp on this platform).
            float sum = 0.0f;
            for (j = 0; j < last_dim; ++j) {
                const float e = std::exp(Tr::load1(row_in + j) - max_val);
                Tr::store1(row_out + j, e);
                sum += e;
            }

            // Pass 3: NEON normalization by reciprocal product.
            const float32x4_t vinv = vdupq_n_f32(1.0f / sum);
            j = 0;
            for (; j + 3 < last_dim; ++j += 4)
                Tr::store4(row_out + j, vmulq_f32(Tr::load4(row_out + j), vinv));
            for (; j < last_dim; ++j)
                Tr::store1(row_out + j, Tr::load1(row_out + j) / sum);
        }
    }

    template <typename Tr>
    static void layernorm_rows(const typename Tr::T* in, typename Tr::T* out,
                               size_t last_dim, size_t r0, size_t r1) {
        constexpr float eps = 1e-5f;
        for (size_t i = r0; i < r1; ++i) {
            const typename Tr::T* row_in = in + i * last_dim;
            typename Tr::T* row_out = out + i * last_dim;

            // Pass 1: NEON mean.
            float sum = 0.0f;
            size_t j = 0;
            if (last_dim >= 4) {
                float32x4_t vsum = vdupq_n_f32(0.0f);
                for (; j + 3 < last_dim; j += 4)
                    vsum = vaddq_f32(vsum, Tr::load4(row_in + j));
                sum = vaddvq_f32(vsum);
            }
            for (; j < last_dim; ++j) sum += Tr::load1(row_in + j);
            const float mean = sum / static_cast<float>(last_dim);

            // Pass 2: NEON variance.
            const float32x4_t vmean = vdupq_n_f32(mean);
            float var = 0.0f;
            j = 0;
            if (last_dim >= 4) {
                float32x4_t vvar = vdupq_n_f32(0.0f);
                for (; j + 3 < last_dim; j += 4) {
                    const float32x4_t d = vsubq_f32(Tr::load4(row_in + j), vmean);
                    vvar = vfmaq_f32(vvar, d, d);
                }
                var = vaddvq_f32(vvar);
            }
            for (; j < last_dim; ++j) {
                const float diff = Tr::load1(row_in + j) - mean;
                var += diff * diff;
            }
            var /= static_cast<float>(last_dim);
            const float inv_std = 1.0f / std::sqrt(var + eps);

            // Pass 3: NEON normalization.
            const float32x4_t vinv_std = vdupq_n_f32(inv_std);
            j = 0;
            for (; j + 3 < last_dim; j += 4)
                Tr::store4(row_out + j, vmulq_f32(vsubq_f32(Tr::load4(row_in + j), vmean), vinv_std));
            for (; j < last_dim; ++j)
                Tr::store1(row_out + j, (Tr::load1(row_in + j) - mean) * inv_std);
        }
    }

    // Tiled M-N-K matmul over a row range; the template only changes how
    // elements are loaded/stored, all arithmetic stays in f32 registers.
    template <typename Tr>
    static void matmul_rows(const typename Tr::T* a_ptr, const typename Tr::T* b_ptr,
                            typename Tr::T* out_ptr,
                            size_t K, size_t N, size_t m_start, size_t m_end) {
        for (size_t m_tile = m_start; m_tile < m_end; m_tile += TILE_SIZE) {
            for (size_t n_tile = 0; n_tile < N; n_tile += TILE_SIZE) {
                const size_t m_max = std::min(m_tile + TILE_SIZE, m_end);
                const size_t n_max = std::min(n_tile + TILE_SIZE, N);

                // Zero the output tile so the accumulation below produces
                // correct results even when the output buffer holds stale
                // data. +0.0 is an all-zero byte pattern for both f32 and
                // f16, so memset is valid for both.
                for (size_t m = m_tile; m < m_max; ++m)
                    std::memset(out_ptr + m * N + n_tile, 0,
                                (n_max - n_tile) * sizeof(typename Tr::T));

                for (size_t k_tile = 0; k_tile < K; k_tile += TILE_SIZE) {
                    const size_t k_max = std::min(k_tile + TILE_SIZE, K);

                    for (size_t m = m_tile; m < m_max; ++m) {
                        const typename Tr::T* a_row = a_ptr + m * K;
                        typename Tr::T* out_row = out_ptr + m * N;
                        for (size_t k = k_tile; k < k_max; ++k) {
                            const float a_val = Tr::load1(a_row + k);
                            const float32x4_t va_val = vdupq_n_f32(a_val);
                            const typename Tr::T* b_row = b_ptr + k * N;

                            size_t n = n_tile;
                            for (; n + 7 < n_max; n += 8) {
                                float32x4_t vout1 = Tr::load4(out_row + n);
                                vout1 = vfmaq_f32(vout1, va_val, Tr::load4(b_row + n));
                                Tr::store4(out_row + n, vout1);

                                float32x4_t vout2 = Tr::load4(out_row + n + 4);
                                vout2 = vfmaq_f32(vout2, va_val, Tr::load4(b_row + n + 4));
                                Tr::store4(out_row + n + 4, vout2);
                            }
                            for (; n < n_max; ++n)
                                Tr::store1(out_row + n,
                                           Tr::load1(out_row + n) + a_val * Tr::load1(b_row + n));
                        }
                    }
                }
            }
        }
    }

    // out = A . B^T with B^T supplied row-major [N, K]: one dot product per
    // output element, both operands stream with unit stride.
    template <typename Tr>
    static void matmul_transposed_rows(const typename Tr::T* a_ptr, const typename Tr::T* b_ptr,
                                       typename Tr::T* out_ptr,
                                       size_t K, size_t N, size_t m_start, size_t m_end) {
        for (size_t m = m_start; m < m_end; ++m) {
            const typename Tr::T* a_row = a_ptr + m * K;
            typename Tr::T* out_row = out_ptr + m * N;
            for (size_t n = 0; n < N; ++n)
                Tr::store1(out_row + n, dot_range<Tr>(a_row, b_ptr + n * K, K));
        }
    }

    // out[M] = A[M,K] . v[K]. Two parallelization strategies:
    //  - M fills the pool: parallel over rows, each row one dot product.
    //  - M is small (decode has M == 1): rows cannot fill the pool, so K is
    //    split into chunks; every chunk computes partial dots into a
    //    disjoint scratch column (no atomics), and the caller reduces.
    template <typename Tr>
    static void matvec_impl(const typename Tr::T* a, const typename Tr::T* v,
                            typename Tr::T* out, size_t M, size_t K) {
        unsigned int hw = std::thread::hardware_concurrency();
        if (hw == 0) hw = 2;

        if (M >= hw) {
            parallel_for(M, 1, [&](size_t s, size_t e) {
                for (size_t m = s; m < e; ++m)
                    Tr::store1(out + m, dot_range<Tr>(a + m * K, v, K));
            });
            return;
        }

        const size_t max_chunks = K / kMinChunk;
        if (max_chunks < 2) {
            for (size_t m = 0; m < M; ++m)
                Tr::store1(out + m, dot_range<Tr>(a + m * K, v, K));
            return;
        }

        const size_t nchunks = std::min<size_t>(hw, max_chunks);
        std::vector<float> partial(M * nchunks);
        parallel_for(nchunks, 1, [&](size_t cs, size_t ce) {
            for (size_t c = cs; c < ce; ++c) {
                const size_t k0 = c * K / nchunks;
                const size_t k1 = (c + 1) * K / nchunks;
                for (size_t m = 0; m < M; ++m)
                    partial[m * nchunks + c] = dot_range<Tr>(a + m * K + k0, v + k0, k1 - k0);
            }
        });
        for (size_t m = 0; m < M; ++m) {
            float sum = 0.0f;
            for (size_t c = 0; c < nchunks; ++c) sum += partial[m * nchunks + c];
            Tr::store1(out + m, sum);
        }
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

        const bool same_shape = a.shape() == b.shape() && a.shape() == out.shape();

        if (a.dtype() == DataType::float32) {
            if (same_shape && a.shape().is_contiguous() && b.shape().is_contiguous()) {
                // add_flat processes elements in order (read both, then
                // write), so it also covers the legal in-place case.
                add_flat<F32Traits>(static_cast<const float*>(a.data()),
                                    static_cast<const float*>(b.data()),
                                    static_cast<float*>(out.data()),
                                    out.shape().element_count());
                return;
            }
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

            // Stride-zero broadcasting: right align each operand against the
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
                    add_flat<F32Traits>(pa, pb, po, inner);
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
        } else if (a.dtype() == DataType::float16) {
            if (!same_shape || !a.shape().is_contiguous() || !b.shape().is_contiguous())
                throw std::invalid_argument("Add: float16 supports same-shape contiguous tensors only.");
            add_flat<F16Traits>(static_cast<const __fp16*>(a.data()),
                                static_cast<const __fp16*>(b.data()),
                                static_cast<__fp16*>(out.data()),
                                out.shape().element_count());
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

        // Each row already does O(K*N) work, so one row per task is enough
        // to justify the dispatch overhead. Raw pointers are hoisted out of
        // the lambdas: a by-value capture would copy the Tensor handle as
        // const and hand back a const void*.
        if (a.dtype() == DataType::float32) {
            const float* a_ptr = static_cast<const float*>(a.data());
            const float* b_ptr = static_cast<const float*>(b.data());
            float* out_ptr = static_cast<float*>(out.data());
            parallel_for(M, 1, [=](size_t s, size_t e) {
                matmul_rows<F32Traits>(a_ptr, b_ptr, out_ptr, K, N, s, e);
            });
        } else if (a.dtype() == DataType::float16) {
            const __fp16* a_ptr = static_cast<const __fp16*>(a.data());
            const __fp16* b_ptr = static_cast<const __fp16*>(b.data());
            __fp16* out_ptr = static_cast<__fp16*>(out.data());
            parallel_for(M, 1, [=](size_t s, size_t e) {
                matmul_rows<F16Traits>(a_ptr, b_ptr, out_ptr, K, N, s, e);
            });
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
            const float* a_ptr = static_cast<const float*>(a.data());
            const float* b_ptr = static_cast<const float*>(b_t.data());
            float* out_ptr = static_cast<float*>(out.data());
            parallel_for(M, 1, [=](size_t s, size_t e) {
                matmul_transposed_rows<F32Traits>(a_ptr, b_ptr, out_ptr, K, N, s, e);
            });
        } else if (a.dtype() == DataType::float16) {
            const __fp16* a_ptr = static_cast<const __fp16*>(a.data());
            const __fp16* b_ptr = static_cast<const __fp16*>(b_t.data());
            __fp16* out_ptr = static_cast<__fp16*>(out.data());
            parallel_for(M, 1, [=](size_t s, size_t e) {
                matmul_transposed_rows<F16Traits>(a_ptr, b_ptr, out_ptr, K, N, s, e);
            });
        } else throw std::runtime_error("Unsupported dtype for MatMulTransposed operation.");
    }

    void MatVec(const Tensor& a, const Tensor& v, Tensor& out) {
        if (a.shape().rank() != 2 || v.shape().rank() != 1 || out.shape().rank() != 1)
            throw std::invalid_argument("MatVec: expected a 2D matrix, a 1D vector and a 1D output.");
        const size_t M = a.shape().dimension(0);
        const size_t K = a.shape().dimension(1);
        if (v.shape().dimension(0) != K)
            throw std::invalid_argument("MatVec: Inner dimensions must match.");
        if (out.shape().dimension(0) != M)
            throw std::invalid_argument("MatVec: Output shape is incompatible.");
        require_contiguous(a, "MatVec");
        require_contiguous(v, "MatVec");
        require_contiguous(out, "MatVec");
        if (Tensor::is_aliased(a, out) || Tensor::is_aliased(v, out))
            throw std::invalid_argument("MatVec: output must not alias an input.");

        if (a.dtype() == DataType::float32) {
            matvec_impl<F32Traits>(static_cast<const float*>(a.data()),
                                   static_cast<const float*>(v.data()),
                                   static_cast<float*>(out.data()), M, K);
        } else if (a.dtype() == DataType::float16) {
            matvec_impl<F16Traits>(static_cast<const __fp16*>(a.data()),
                                   static_cast<const __fp16*>(v.data()),
                                   static_cast<__fp16*>(out.data()), M, K);
        } else throw std::runtime_error("Unsupported dtype for MatVec operation.");
    }

    void ReLU(const Tensor& in, Tensor& out) {
        if (!(in.shape() == out.shape())) throw std::invalid_argument("ReLU: Shapes must match.");
        require_contiguous(in, "ReLU");
        require_contiguous(out, "ReLU");
        const size_t n = in.shape().element_count();
        if (in.dtype() == DataType::float32) {
            const float* in_ptr = static_cast<const float*>(in.data());
            float* out_ptr = static_cast<float*>(out.data());
            parallel_for(n, kMinChunk, [=](size_t s, size_t e) {
                relu_range<F32Traits>(in_ptr, out_ptr, s, e);
            });
        } else if (in.dtype() == DataType::float16) {
            const __fp16* in_ptr = static_cast<const __fp16*>(in.data());
            __fp16* out_ptr = static_cast<__fp16*>(out.data());
            parallel_for(n, kMinChunk, [=](size_t s, size_t e) {
                relu_range<F16Traits>(in_ptr, out_ptr, s, e);
            });
        } else throw std::runtime_error("Unsupported dtype for ReLU operation.");
    }

    void GELU(const Tensor& in, Tensor& out) {
        if (!(in.shape() == out.shape())) throw std::invalid_argument("GELU: Shapes must match.");
        require_contiguous(in, "GELU");
        require_contiguous(out, "GELU");
        const size_t n = in.shape().element_count();
        if (in.dtype() == DataType::float32) {
            const float* in_ptr = static_cast<const float*>(in.data());
            float* out_ptr = static_cast<float*>(out.data());
            parallel_for(n, kMinChunk, [=](size_t s, size_t e) {
                gelu_range<F32Traits>(in_ptr, out_ptr, s, e);
            });
        } else if (in.dtype() == DataType::float16) {
            const __fp16* in_ptr = static_cast<const __fp16*>(in.data());
            __fp16* out_ptr = static_cast<__fp16*>(out.data());
            parallel_for(n, kMinChunk, [=](size_t s, size_t e) {
                gelu_range<F16Traits>(in_ptr, out_ptr, s, e);
            });
        } else throw std::runtime_error("Unsupported dtype for GELU operation.");
    }

    void Softmax(const Tensor& in, Tensor& out) {
        if (!(in.shape() == out.shape())) throw std::invalid_argument("Softmax: Shapes must match.");
        require_contiguous(in, "Softmax");
        require_contiguous(out, "Softmax");
        const size_t last_dim = in.shape().dimension(in.shape().rank() - 1);
        const size_t num_rows = in.shape().element_count() / last_dim;

        // Each row does O(last_dim) work — parallelize over rows.
        if (in.dtype() == DataType::float32) {
            const float* in_ptr = static_cast<const float*>(in.data());
            float* out_ptr = static_cast<float*>(out.data());
            parallel_for(num_rows, 1, [=](size_t s, size_t e) {
                softmax_rows<F32Traits>(in_ptr, out_ptr, last_dim, s, e);
            });
        } else if (in.dtype() == DataType::float16) {
            const __fp16* in_ptr = static_cast<const __fp16*>(in.data());
            __fp16* out_ptr = static_cast<__fp16*>(out.data());
            parallel_for(num_rows, 1, [=](size_t s, size_t e) {
                softmax_rows<F16Traits>(in_ptr, out_ptr, last_dim, s, e);
            });
        } else throw std::runtime_error("Unsupported dtype for Softmax operation.");
    }

    void LayerNorm(const Tensor& in, Tensor& out) {
        if (!(in.shape() == out.shape())) throw std::invalid_argument("LayerNorm: Shapes must match.");
        require_contiguous(in, "LayerNorm");
        require_contiguous(out, "LayerNorm");
        const size_t last_dim = in.shape().dimension(in.shape().rank() - 1);
        const size_t num_rows = in.shape().element_count() / last_dim;

        if (in.dtype() == DataType::float32) {
            const float* in_ptr = static_cast<const float*>(in.data());
            float* out_ptr = static_cast<float*>(out.data());
            parallel_for(num_rows, 1, [=](size_t s, size_t e) {
                layernorm_rows<F32Traits>(in_ptr, out_ptr, last_dim, s, e);
            });
        } else if (in.dtype() == DataType::float16) {
            const __fp16* in_ptr = static_cast<const __fp16*>(in.data());
            __fp16* out_ptr = static_cast<__fp16*>(out.data());
            parallel_for(num_rows, 1, [=](size_t s, size_t e) {
                layernorm_rows<F16Traits>(in_ptr, out_ptr, last_dim, s, e);
            });
        } else throw std::runtime_error("Unsupported dtype for LayerNorm operation.");
    }

} // namespace kern::ops
