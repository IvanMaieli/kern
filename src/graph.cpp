//
// Created by Ivan Maieli on 20/08/2026.
//

#include <kern/graph.hpp>
#include <kern/ops.hpp>
#include <algorithm>
#include <memory>
#include <stdexcept>

namespace kern {

    namespace {
        std::size_t align64(const std::size_t bytes) {
            return (bytes + ALIGNMENT - 1) & ~(ALIGNMENT - 1);
        }
    } // namespace

    void Graph::check_id(const TensorId id) const {
        if (id >= shapes_.size())
            throw std::out_of_range("Graph: unknown TensorId.");
    }

    TensorId Graph::new_value(const Shape& shape, const DataType dtype, const bool external) {
        if (built_)
            throw std::runtime_error("Graph: cannot add values after build().");
        if (dtype != DataType::float32 && dtype != DataType::float16)
            throw std::invalid_argument("Graph: only float32/float16 values are supported.");
        const TensorId id = shapes_.size();
        shapes_.push_back(shape);
        dtypes_.push_back(dtype);
        def_node_.push_back(0);
        is_output_.push_back(false);
        kind_.push_back(external ? 'e' : 'a');
        if (external)
            values_.emplace_back(shape, dtype);
        else
            values_.emplace_back(); // activation buffer is assigned at build()
        return id;
    }

    TensorId Graph::add_input(const Shape& shape, const DataType dtype) {
        return new_value(shape, dtype, true);
    }

    TensorId Graph::add_param(const Shape& shape, const DataType dtype) {
        return new_value(shape, dtype, true);
    }

    TensorId Graph::add_unary(const Op op, const TensorId x) {
        check_id(x);
        const TensorId out = new_value(shapes_[x], dtypes_[x], false);
        nodes_.push_back({op, x, x, out});
        def_node_[out] = nodes_.size() - 1;
        return out;
    }

    TensorId Graph::add_binary(const Op op, const TensorId a, const TensorId b, const Shape& out_shape) {
        check_id(a);
        check_id(b);
        if (dtypes_[a] != dtypes_[b])
            throw std::invalid_argument("Graph: operand dtypes must match.");
        const TensorId out = new_value(out_shape, dtypes_[a], false);
        nodes_.push_back({op, a, b, out});
        def_node_[out] = nodes_.size() - 1;
        return out;
    }

    TensorId Graph::add_matmul(const TensorId a, const TensorId b) {
        check_id(a);
        check_id(b);
        if (shapes_[a].rank() != 2 || shapes_[b].rank() != 2)
            throw std::invalid_argument("Graph::add_matmul: operands must be 2D.");
        const std::size_t K = shapes_[a].dimension(1);
        if (shapes_[b].dimension(0) != K)
            throw std::invalid_argument("Graph::add_matmul: inner dimensions must match.");
        return add_binary(Op::MatMul, a, b, Shape{shapes_[a].dimension(0), shapes_[b].dimension(1)});
    }

    TensorId Graph::add_matmul_transposed(const TensorId a, const TensorId b_t) {
        check_id(a);
        check_id(b_t);
        if (shapes_[a].rank() != 2 || shapes_[b_t].rank() != 2)
            throw std::invalid_argument("Graph::add_matmul_transposed: operands must be 2D.");
        const std::size_t K = shapes_[a].dimension(1);
        if (shapes_[b_t].dimension(1) != K)
            throw std::invalid_argument("Graph::add_matmul_transposed: inner dimensions must match.");
        return add_binary(Op::MatMulTransposed, a, b_t,
                          Shape{shapes_[a].dimension(0), shapes_[b_t].dimension(0)});
    }

    TensorId Graph::add_matvec(const TensorId a, const TensorId v) {
        check_id(a);
        check_id(v);
        if (shapes_[a].rank() != 2 || shapes_[v].rank() != 1)
            throw std::invalid_argument("Graph::add_matvec: expected a 2D matrix and a 1D vector.");
        const std::size_t K = shapes_[a].dimension(1);
        if (shapes_[v].dimension(0) != K)
            throw std::invalid_argument("Graph::add_matvec: inner dimensions must match.");
        return add_binary(Op::MatVec, a, v, Shape{shapes_[a].dimension(0)});
    }

    TensorId Graph::add_add(const TensorId a, const TensorId b) {
        check_id(a);
        check_id(b);
        return add_binary(Op::Add, a, b, ops::GetBroadcastShape(shapes_[a], shapes_[b]));
    }

    TensorId Graph::add_relu(const TensorId x) { return add_unary(Op::ReLU, x); }
    TensorId Graph::add_gelu(const TensorId x) { return add_unary(Op::GELU, x); }
    TensorId Graph::add_softmax(const TensorId x) { return add_unary(Op::Softmax, x); }
    TensorId Graph::add_layernorm(const TensorId x) { return add_unary(Op::LayerNorm, x); }

    void Graph::mark_output(const TensorId id) {
        check_id(id);
        if (built_)
            throw std::runtime_error("Graph: cannot mark outputs after build().");
        is_output_[id] = true;
    }

    void Graph::build() {
        if (built_)
            throw std::runtime_error("Graph: already built.");
        built_ = true;

        const std::size_t n = nodes_.size();

        // Live range of each value: [def node, last use]. Graph outputs live
        // until the end so run() leaves them readable.
        std::vector<std::size_t> last_use(values_.size(), 0);
        for (std::size_t i = 0; i < n; ++i) {
            const Node& nd = nodes_[i];
            last_use[nd.a] = std::max(last_use[nd.a], i);
            last_use[nd.b] = std::max(last_use[nd.b], i);
            last_use[nd.out] = i; // at least its own definition
        }
        for (TensorId id = 0; id < values_.size(); ++id)
            if (is_output_[id]) last_use[id] = n;

        // Linear-scan slot allocation over one arena. Activations are visited
        // in id order, which is definition order. A slot is reusable when the
        // incoming value is defined at or after the previous tenant's
        // last_use + 1 — strictly after the node that last read it.
        struct Slot {
            std::size_t offset;
            std::size_t size;
            std::size_t free_at; // first node index that may reuse this slot
        };
        std::vector<Slot> slots;
        std::vector<std::size_t> offset(values_.size(), 0);
        std::vector<std::size_t> bytes(values_.size(), 0);

        for (TensorId id = 0; id < values_.size(); ++id) {
            if (kind_[id] != 'a') continue;
            bytes[id] = align64(shapes_[id].element_count() * size_bytes(dtypes_[id]));
            if (bytes[id] == 0) continue;

            const std::size_t def = def_node_[id];
            Slot* chosen = nullptr;
            for (Slot& s : slots)
                if (s.free_at <= def && s.size >= bytes[id]) {
                    chosen = &s;
                    break;
                }
            if (!chosen) {
                slots.push_back({arena_bytes_, bytes[id], 0});
                arena_bytes_ += bytes[id];
                chosen = &slots.back();
            }
            chosen->free_at = last_use[id] + 1;
            offset[id] = chosen->offset;
        }

        // Materialize the activations as non-owning views into the arena.
        if (arena_bytes_ > 0) {
            arena_ = std::make_shared<Buffer>(arena_bytes_);
            for (TensorId id = 0; id < values_.size(); ++id) {
                if (kind_[id] != 'a' || bytes[id] == 0) continue;
                void* ptr = static_cast<char*>(arena_->data()) + offset[id];
                values_[id] = Tensor(shapes_[id], dtypes_[id],
                                     std::make_shared<Buffer>(ptr, bytes[id]));
            }
        }
    }

    void Graph::run() {
        if (!built_)
            throw std::runtime_error("Graph: call build() before run().");
        for (const Node& nd : nodes_) {
            Tensor& a = values_[nd.a];
            Tensor& b = values_[nd.b];
            Tensor& o = values_[nd.out];
            switch (nd.op) {
                case Op::MatMul:          ops::MatMul(a, b, o); break;
                case Op::MatMulTransposed:ops::MatMulTransposed(a, b, o); break;
                case Op::MatVec:          ops::MatVec(a, b, o); break;
                case Op::Add:             ops::Add(a, b, o); break;
                case Op::ReLU:            ops::ReLU(a, o); break;
                case Op::GELU:            ops::GELU(a, o); break;
                case Op::Softmax:         ops::Softmax(a, o); break;
                case Op::LayerNorm:       ops::LayerNorm(a, o); break;
            }
        }
    }

    Tensor& Graph::value(const TensorId id) {
        check_id(id);
        return values_[id];
    }

    const Shape& Graph::shape(const TensorId id) const {
        check_id(id);
        return shapes_[id];
    }

} // namespace kern
