//
// Created by Ivan Maieli on 20/08/2026.
//

#pragma once

#include <kern/tensor.hpp>
#include <kern/shape.hpp>
#include <kern/dtype.hpp>
#include <cstddef>
#include <vector>

namespace kern {

    using TensorId = std::size_t;

    // Small functional compute graph over the kern::ops kernels.
    //
    // The API is SSA-style: every node produces a fresh value id and may only
    // consume ids defined before it, so submission order is automatically a
    // valid topological order — no scheduling pass is needed.
    //
    // build() plans the activation memory: each value's live range is
    // [defining node, last use]; values whose ranges do not overlap share the
    // same arena slot (linear-scan allocation over one aligned Buffer). A
    // slot is recycled only when the new value is defined strictly after the
    // old value's last use, so a node never reads an input through memory
    // already rewritten as its own output.
    class Graph {
    public:
        Graph() = default;

        // External values owned by the caller: fill them via value(id)
        // before run(). Params are conceptually weights, inputs activations.
        TensorId add_input(const Shape& shape, DataType dtype);
        TensorId add_param(const Shape& shape, DataType dtype);

        TensorId add_matmul(TensorId a, TensorId b);
        TensorId add_matmul_transposed(TensorId a, TensorId b_t);
        TensorId add_matvec(TensorId a, TensorId v);
        TensorId add_add(TensorId a, TensorId b);
        TensorId add_relu(TensorId x);
        TensorId add_gelu(TensorId x);
        TensorId add_softmax(TensorId x);
        TensorId add_layernorm(TensorId x);

        // Extends a value's live range to the end of the graph so its arena
        // slot is not recycled by later nodes. Call before build().
        void mark_output(TensorId id);

        // Validates nothing further (shapes were checked at add time) and
        // plans the activation arena.
        void build();

        // Executes the nodes in order. Inputs must have been filled.
        void run();

        [[nodiscard]] Tensor& value(TensorId id);
        [[nodiscard]] const Shape& shape(TensorId id) const;
        // Planned activation footprint (post-reuse), not counting inputs/params.
        [[nodiscard]] std::size_t arena_bytes() const noexcept { return arena_bytes_; }
        [[nodiscard]] std::size_t node_count() const noexcept { return nodes_.size(); }

    private:
        enum class Op { MatMul, MatMulTransposed, MatVec, Add, ReLU, GELU, Softmax, LayerNorm };
        struct Node {
            Op op;
            TensorId a;
            TensorId b; // unused for unary ops (equals a)
            TensorId out;
        };

        TensorId new_value(const Shape& shape, DataType dtype, bool external);
        TensorId add_unary(Op op, TensorId x);
        TensorId add_binary(Op op, TensorId a, TensorId b, const Shape& out_shape);
        void check_id(TensorId id) const;

        std::vector<Node> nodes_;          // compute nodes in topological order
        std::vector<Shape> shapes_;        // per value
        std::vector<DataType> dtypes_;     // per value
        std::vector<Tensor> values_;       // externals are real tensors, activations are arena views
        std::vector<char> kind_;           // 'i' input, 'p' param, 'a' activation
        std::vector<std::size_t> def_node_;// node index that defines each value
        std::vector<bool> is_output_;
        std::shared_ptr<Buffer> arena_;    // one aligned block backing all activations
        std::size_t arena_bytes_ = 0;
        bool built_ = false;
    };

} // namespace kern
