#include <kern/ops.hpp>
#include <stdexcept>

namespace kern::ops {

    void Add(const Tensor& a, const Tensor& b, Tensor& out) {
        if (!(a.shape() == b.shape() && a.shape() == out.shape()))
            throw std::invalid_argument("Tensors must have the same shape for Add operation.");

        if (a.dtype() != b.dtype() || a.dtype() != out.dtype())
            throw std::invalid_argument("Tensors must have the same dtype for Add operation.");

        if (a.dtype() == DataType::float32) {
            const auto a_ptr = static_cast<const float*>(a.data());
            const auto b_ptr = static_cast<const float*>(b.data());
            const auto out_ptr = static_cast<float*>(out.data());
            const size_t n = a.shape().element_count();
            for (size_t i = 0; i < n; ++i) out_ptr[i] = a_ptr[i] + b_ptr[i];
        } else throw std::runtime_error("Unsupported dtype for Add operation.");
    }
} // namespace kern::ops
