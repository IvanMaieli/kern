#include "tests.hpp"
#include <cstdio>
#include <cstdlib>

int main() {
    std::printf("Running version...\n"); test_version();
    std::printf("Running shape...\n"); test_shape();
    std::printf("Running dtype...\n"); test_dtype();
    std::printf("Running tensor default...\n"); test_tensor_default();
    std::printf("Running tensor basics...\n"); test_tensor_basics();
    std::printf("Running tensor alignment...\n"); test_tensor_alignment();
    std::printf("Running tensor move...\n"); test_tensor_move();
    std::printf("Running tensor with pool...\n"); test_tensor_with_pool();
    std::printf("Running ops add...\n"); test_ops_add();
    std::printf("Running ops add mismatched...\n"); test_ops_add_mismatched_shape();
    std::printf("Running ops matmul...\n"); test_ops_matmul();
    std::printf("Running ops matmul output reuse...\n"); test_ops_matmul_output_reuse();
    std::printf("Running ops matmul non contiguous...\n"); test_ops_matmul_non_contiguous_rejected();
    std::printf("Running ops matmul transposed...\n"); test_ops_matmul_transposed();
    std::printf("Running ops relu...\n"); test_ops_relu();
    std::printf("Running ops relu vectorized...\n"); test_ops_relu_vectorized();
    std::printf("Running ops gelu...\n"); test_ops_gelu();
    std::printf("Running ops gelu accuracy...\n"); test_ops_gelu_accuracy();
    std::printf("Running ops softmax...\n"); test_ops_softmax();
    std::printf("Running ops layernorm...\n"); test_ops_layernorm();
    std::printf("Running ops broadcast add...\n"); test_ops_broadcast_add();
    std::printf("Running ops broadcast add rank mismatch...\n"); test_ops_broadcast_add_rank_mismatch();
    std::printf("Running ops broadcast add scalar...\n"); test_ops_broadcast_add_scalar();
    std::printf("Running ops add inplace...\n"); test_ops_add_inplace();
    std::printf("Running ops add inplace broadcast rejected...\n"); test_ops_add_inplace_broadcast_rejected();
    std::printf("Running bench matmul...\n"); test_bench_matmul();
    std::printf("Running bench add broadcast...\n"); test_bench_add_broadcast();
    std::printf("Running bench matmul transposed...\n"); test_bench_matmul_transposed();
    std::printf("Running bench gelu...\n"); test_bench_gelu();
    std::printf("Running ops reshape...\n"); test_ops_reshape();
    std::printf("Running ops permute...\n"); test_ops_permute();
    std::printf("Running ops permute data...\n"); test_ops_permute_data();
    std::printf("Running ops transpose data...\n"); test_ops_transpose_data();
    std::printf("Running pool alignment...\n"); test_memory_pool_alignment();
    std::printf("Running pool reset...\n"); test_memory_pool_reset();
    std::printf("Running pool oom...\n"); test_memory_pool_oom();

    if (failures != 0) {
        std::fprintf(stderr, "%d check(s) failed.\n", failures);
        return EXIT_FAILURE;
    }

    std::printf("All checks passed.\n");
    return EXIT_SUCCESS;
}


