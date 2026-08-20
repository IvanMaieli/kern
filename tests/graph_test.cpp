#include <kern/graph.hpp>
#include <kern/ops.hpp>
#include <kern/tensor.hpp>
#include <kern/dtype.hpp>
#include "test_macros.hpp"
#include "tests.hpp"
#include <cmath>
#include <vector>

void test_graph_matvec_mlp() {
    // x[4] -> MatVec(W1[8,4]) -> GELU -> MatVec(W2[2,8]) -> out[2]
    kern::Graph g;
    const kern::TensorId x = g.add_input(kern::Shape{4}, kern::DataType::float32);
    const kern::TensorId w1 = g.add_param(kern::Shape{8, 4}, kern::DataType::float32);
    const kern::TensorId w2 = g.add_param(kern::Shape{2, 8}, kern::DataType::float32);
    const kern::TensorId h = g.add_matvec(w1, x);
    const kern::TensorId a = g.add_gelu(h);
    const kern::TensorId o = g.add_matvec(w2, a);
    g.mark_output(o);
    g.build();
    CHECK(g.node_count() == 3); // matvec, gelu, matvec — inputs/params are values, not nodes

    auto* xp = static_cast<float*>(g.value(x).data());
    auto* w1p = static_cast<float*>(g.value(w1).data());
    auto* w2p = static_cast<float*>(g.value(w2).data());
    for (std::size_t i = 0; i < 4; ++i) xp[i] = 0.25f * static_cast<float>(i) - 0.5f;
    for (std::size_t i = 0; i < 32; ++i) w1p[i] = 0.01f * static_cast<float>(i) - 0.15f;
    for (std::size_t i = 0; i < 16; ++i) w2p[i] = 0.02f * static_cast<float>(i) - 0.30f;

    g.run();

    // Reference: same network executed with plain ops.
    kern::Tensor r_x(kern::Shape{4}, kern::DataType::float32);
    kern::Tensor r_w1(kern::Shape{8, 4}, kern::DataType::float32);
    kern::Tensor r_w2(kern::Shape{2, 8}, kern::DataType::float32);
    kern::Tensor r_h(kern::Shape{8}, kern::DataType::float32);
    kern::Tensor r_a(kern::Shape{8}, kern::DataType::float32);
    kern::Tensor r_o(kern::Shape{2}, kern::DataType::float32);
    for (std::size_t i = 0; i < 4; ++i) static_cast<float*>(r_x.data())[i] = xp[i];
    for (std::size_t i = 0; i < 32; ++i) static_cast<float*>(r_w1.data())[i] = w1p[i];
    for (std::size_t i = 0; i < 16; ++i) static_cast<float*>(r_w2.data())[i] = w2p[i];
    kern::ops::MatVec(r_w1, r_x, r_h);
    kern::ops::GELU(r_h, r_a);
    kern::ops::MatVec(r_w2, r_a, r_o);

    const auto* got = static_cast<const float*>(g.value(o).data());
    const auto* ref = static_cast<const float*>(r_o.data());
    for (std::size_t i = 0; i < 2; ++i)
        CHECK(std::abs(got[i] - ref[i]) < 1e-6f);
}

void test_graph_transformer_block() {
    // res = GELU(MatMulTransposed(LayerNorm(MatMulTransposed(x, Wq)), W2)) + x
    constexpr std::size_t seq = 2;
    constexpr std::size_t d = 16;
    kern::Graph g;
    const kern::TensorId x = g.add_input(kern::Shape{seq, d}, kern::DataType::float32);
    const kern::TensorId wq = g.add_param(kern::Shape{d, d}, kern::DataType::float32);
    const kern::TensorId w2 = g.add_param(kern::Shape{d, d}, kern::DataType::float32);
    const kern::TensorId mh = g.add_matmul_transposed(x, wq);
    const kern::TensorId n = g.add_layernorm(mh);
    const kern::TensorId p = g.add_matmul_transposed(n, w2);
    const kern::TensorId a = g.add_gelu(p);
    const kern::TensorId res = g.add_add(a, x);
    g.mark_output(res);
    g.build();

    auto* xp = static_cast<float*>(g.value(x).data());
    auto* wqp = static_cast<float*>(g.value(wq).data());
    auto* w2p = static_cast<float*>(g.value(w2).data());
    for (std::size_t i = 0; i < seq * d; ++i) xp[i] = 0.01f * static_cast<float>(i) - 0.1f;
    for (std::size_t i = 0; i < d * d; ++i) {
        wqp[i] = 0.001f * static_cast<float>((i * 13) % 97) - 0.05f;
        w2p[i] = 0.001f * static_cast<float>((i * 29) % 89) - 0.04f;
    }

    g.run();

    // Reference with plain ops.
    kern::Tensor r_x(kern::Shape{seq, d}, kern::DataType::float32);
    kern::Tensor r_wq(kern::Shape{d, d}, kern::DataType::float32);
    kern::Tensor r_w2(kern::Shape{d, d}, kern::DataType::float32);
    for (std::size_t i = 0; i < seq * d; ++i) static_cast<float*>(r_x.data())[i] = xp[i];
    for (std::size_t i = 0; i < d * d; ++i) {
        static_cast<float*>(r_wq.data())[i] = wqp[i];
        static_cast<float*>(r_w2.data())[i] = w2p[i];
    }
    kern::Tensor r1(kern::Shape{seq, d}, kern::DataType::float32);
    kern::Tensor r2(kern::Shape{seq, d}, kern::DataType::float32);
    kern::Tensor r3(kern::Shape{seq, d}, kern::DataType::float32);
    kern::Tensor r4(kern::Shape{seq, d}, kern::DataType::float32);
    kern::ops::MatMulTransposed(r_x, r_wq, r1);
    kern::ops::LayerNorm(r1, r2);
    kern::ops::MatMulTransposed(r2, r_w2, r3);
    kern::ops::GELU(r3, r4);
    kern::Tensor r_res(kern::Shape{seq, d}, kern::DataType::float32);
    kern::ops::Add(r4, r_x, r_res);

    const auto* got = static_cast<const float*>(g.value(res).data());
    const auto* ref = static_cast<const float*>(r_res.data());
    for (std::size_t i = 0; i < seq * d; ++i)
        CHECK(std::abs(got[i] - ref[i]) < 1e-5f);
}

void test_graph_memory_reuse() {
    // A sequential chain relu -> gelu -> relu over [1,1024]: y1 is dead once
    // y2 exists, so y1's slot is recycled by y3. Arena must hold two buffers,
    // not three.
    kern::Graph g;
    const kern::TensorId x = g.add_input(kern::Shape{1, 1024}, kern::DataType::float32);
    const kern::TensorId y1 = g.add_relu(x);
    const kern::TensorId y2 = g.add_gelu(y1);
    const kern::TensorId y3 = g.add_relu(y2);
    g.mark_output(y3);
    g.build();

    constexpr std::size_t one = 1024 * sizeof(float); // 4096 bytes, already 64-aligned
    CHECK(g.arena_bytes() == 2 * one);

    // And it still computes correctly.
    auto* xp = static_cast<float*>(g.value(x).data());
    for (std::size_t i = 0; i < 1024; ++i) xp[i] = 0.002f * static_cast<float>(i % 1000) - 1.0f;
    g.run();
    const auto* got = static_cast<const float*>(g.value(y3).data());
    for (std::size_t i = 0; i < 1024; ++i) {
        const float v0 = std::max(0.0f, xp[i]);
        const float k1 = 0.7978845608f;
        const float k2 = 0.044715f;
        const float v1 = 0.5f * v0 * (1.0f + std::tanh(k1 * (v0 + k2 * v0 * v0 * v0)));
        const float expected = std::max(0.0f, v1);
        CHECK(std::abs(got[i] - expected) < 1e-5f);
    }
}

void test_graph_rerun() {
    // run() must be repeatable: the second pass overwrites the same arena
    // slots and yields the same values.
    kern::Graph g;
    const kern::TensorId x = g.add_input(kern::Shape{8}, kern::DataType::float32);
    const kern::TensorId w = g.add_param(kern::Shape{4, 8}, kern::DataType::float32);
    const kern::TensorId h = g.add_matvec(w, x);
    const kern::TensorId o = g.add_gelu(h);
    g.mark_output(o);
    g.build();

    auto* xp = static_cast<float*>(g.value(x).data());
    auto* wp = static_cast<float*>(g.value(w).data());
    for (std::size_t i = 0; i < 8; ++i) xp[i] = 0.1f * static_cast<float>(i) - 0.4f;
    for (std::size_t i = 0; i < 32; ++i) wp[i] = 0.01f * static_cast<float>(i % 17) - 0.08f;

    g.run();
    std::vector<float> first(4);
    const auto* op = static_cast<const float*>(g.value(o).data());
    for (std::size_t i = 0; i < 4; ++i) first[i] = op[i];

    g.run();
    for (std::size_t i = 0; i < 4; ++i)
        CHECK(std::abs(op[i] - first[i]) < 1e-7f);
}
