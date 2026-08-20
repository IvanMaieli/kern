//
// Created by Ivan Maieli on 15/08/2026.
//

#pragma once

#include <kern/tensor.hpp>
#include <kern/thread_pool.hpp>

namespace kern::ops {
    ThreadPool& GetThreadPool();
    void Add(const Tensor& a, const Tensor& b, Tensor& out);
    void Reshape(const Tensor& in, const Shape& new_shape, Tensor& out);
    void Transpose(const Tensor& in, std::size_t axis1, std::size_t axis2, Tensor& out);
    void Permute(const Tensor& in, const std::vector<std::size_t>& order, Tensor& out);
    void MatMul(const Tensor& a, const Tensor& b, Tensor& out);
    void MatMulTransposed(const Tensor& a, const Tensor& b_t, Tensor& out);
    void ReLU(const Tensor& in, Tensor& out);
    void GELU(const Tensor& in, Tensor& out);
    void Softmax(const Tensor& in, Tensor& out);
    void LayerNorm(const Tensor& in, Tensor& out);
    Shape GetBroadcastShape(const Shape& a, const Shape& b);

} // namespace kern::ops
