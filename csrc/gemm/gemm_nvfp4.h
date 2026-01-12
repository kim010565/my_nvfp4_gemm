#pragma once

// https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html
#include "../common/common.h"

namespace my_plugins {
// mma指令的维度(mma.m16n8k32)
constexpr int MMA_M = 16;
constexpr int MMA_N = 8;
constexpr int MMA_K = 64;

// Block分块（尽量用满共享内存，50系列最大99KB；M、N尽量大且接近，优化global memory->shared memory的重复load）
constexpr int BLK_M = 128;
constexpr int BLK_N = 128;
constexpr int BLK_M_TILES = BLK_M / MMA_M;
constexpr int BLK_N_TILES = BLK_N / MMA_N;

// Warp分块（尽量用满reg，每个线程最大255个reg；M、N尽量大且接近，优化shared memory->reg的重复load）
constexpr int WARP_M = 64;
constexpr int WARP_N = 64;
constexpr int BLK_M_WARPS = BLK_M / WARP_M;
constexpr int BLK_N_WARPS = BLK_N / WARP_N;
constexpr int WARP_M_TILES = WARP_M / MMA_M;
constexpr int WARP_N_TILES = WARP_N / MMA_N;

// 一个block内的线程数
constexpr int WARPS_PER_BLK = BLK_M_WARPS * BLK_N_WARPS;
constexpr int THREADS_PER_BLK = WARPS_PER_BLK * WARP_SIZE;

// multi-stage，在K方向上，一个block需要分成多少个阶段来计算（通过multi-stage来优化隐藏访存/计算）
constexpr int K_NUM_STAGE = 6;
constexpr int K_SF_NUM_STAGE = (K_NUM_STAGE + 3) / 4 + 1;

// f(M,N,K,S)=(M*N*2+(M*K/2+N*K/2+M*K*4/16+N*K*4/16+2*8)*S)/1024
// 需要占用的shared memory大小
constexpr int SMEM_D_SIZE = BLK_M * BLK_N * sizeof(__nv_bfloat16);
constexpr int SMEM_A_SIZE = BLK_M * MMA_K * sizeof(__nv_fp4x2_e2m1) / 2;
constexpr int SMEM_B_SIZE = BLK_N * MMA_K * sizeof(__nv_fp4x2_e2m1) / 2;
constexpr int SMEM_A_SF_SIZE = BLK_M * MMA_K * 4 * sizeof(__nv_fp8_e4m3) / 16;
constexpr int SMEM_B_SF_SIZE = BLK_N * MMA_K * 4 * sizeof(__nv_fp8_e4m3) / 16;
constexpr int SMEM_MBAR_SIZE = WARPS_PER_BLK * sizeof(uint64_t);
constexpr int MAX_SMEN_SIZE = SMEM_D_SIZE + (SMEM_A_SIZE + SMEM_B_SIZE + SMEM_MBAR_SIZE) * K_NUM_STAGE + (SMEM_A_SF_SIZE + SMEM_B_SF_SIZE) * K_SF_NUM_STAGE;

void gemm_nvfp4(const __nv_fp4x2_e2m1 *A, const __nv_fp4x2_e2m1 *B,
                const __nv_fp8_e4m3 *A_sf, const __nv_fp8_e4m3 *B_sf,
                __nv_bfloat16 *D, const float scale,
                const int M, const int N, const int K, cudaStream_t stream);
}  // namespace my_plugins