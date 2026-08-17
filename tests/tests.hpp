#pragma once

inline int failures = 0;

void test_version();
void test_shape();
void test_dtype();
void test_tensor_basics();
void test_tensor_alignment();
void test_tensor_move();
void test_tensor_with_pool();
void test_ops_add();
void test_ops_add_mismatched_shape();
void test_ops_reshape();
void test_ops_permute();
void test_ops_matmul();
void test_ops_broadcast_add();
void test_memory_pool_alignment();
void test_memory_pool_reset();
void test_memory_pool_oom();
