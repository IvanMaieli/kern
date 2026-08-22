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
    // out[M] = A[M,K] . v[K]. With M == 1 this is the decode-path GEMV; the
    // K axis is split across threads when there are too few rows to fill the
    // pool. float32 and float16 are supported.
    void MatVec(const Tensor& a, const Tensor& v, Tensor& out);
    // Copy x [M, C] into cache rows [base, base + M). The caller owns the
    // write cursor: appending the current token's K/V before Attention makes
    // token i (absolute position base+i) attend cache rows [0, base+i],
    // itself included.
    void KVCacheAppend(const Tensor& x, std::size_t base, Tensor& cache);
    // Multi-head causal attention of M query tokens over a KV cache.
    //   q:       [M, n_heads * head_dim]
    //   k_cache: [max_seq, n_kv_heads * head_dim], rows [0, base + M) valid
    //   v_cache: same shape as k_cache
    //   out:     [M, n_heads * head_dim]
    // Causality is loop bounds, not a mask tensor. GQA: query head h reads
    // kv head h / (n_heads / n_kv_heads). Scores are scaled by
    // 1/sqrt(head_dim); the softmax runs streaming (online max/sum), so each
    // cache row is read once and no score matrix is materialized. With M == 1
    // this is the decode path. float32 and float16 are supported.
    void Attention(const Tensor& q, const Tensor& k_cache, const Tensor& v_cache,
                   std::size_t base, std::size_t n_heads, std::size_t head_dim,
                   Tensor& out);
    void ReLU(const Tensor& in, Tensor& out);
    void GELU(const Tensor& in, Tensor& out);
    void Softmax(const Tensor& in, Tensor& out);
    void LayerNorm(const Tensor& in, Tensor& out);
    Shape GetBroadcastShape(const Shape& a, const Shape& b);

} // namespace kern::ops
