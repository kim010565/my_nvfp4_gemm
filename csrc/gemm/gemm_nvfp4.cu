#include "gemm_nvfp4.h"

namespace my_plugins {
// __restrict__是CUDA中的一个重要关键字，用于向编译器提供额外的内存别名信息，从而帮助编译器生成更优化的代码
__global__ void gemm_nvfp4_kernel(const __nv_fp4x2_e2m1 *__restrict__ A, const __nv_fp4x2_e2m1 *__restrict__ B,
                                  const __nv_fp8_e4m3 *__restrict__ A_sf, const __nv_fp8_e4m3 *__restrict__ B_sf,
                                  __nv_bfloat16 *__restrict__ D,
                                  const int M, const int N, const int K, const float scale) {
  const int blk_m_start = blockIdx.x * BLK_M;
  const int blk_n_start = blockIdx.y * BLK_N;
  if (blk_m_start >= M || blk_n_start >= N) {
    return;
  }
  uint32_t warpid, laneid;
  asm("mov.u32 %0, %%warpid;" : "=r"(warpid) :);
  asm("mov.u32 %0, %%laneid;" : "=r"(laneid) :);

  const int blk_m_end = min(M, blk_m_start + BLK_M);
  const int blk_m_tiles = ceil_div(blk_m_end - blk_m_start, MMA_M);
  const int blk_n_end = min(N, blk_n_start + BLK_N);
  const int blk_n_tiles = ceil_div(blk_n_end - blk_n_start, MMA_N);
  const int mma_k_tiles = K / MMA_K;

  extern __shared__ uint8_t smem_buffer[];
  __nv_bfloat16 *smem_d = (__nv_bfloat16 *)smem_buffer;
  __nv_fp4x2_e2m1 *smem_a[K_NUM_STAGE];
  __nv_fp4x2_e2m1 *smem_b[K_NUM_STAGE];
  uint64_t *load_full_mbar[K_NUM_STAGE];
  uint32_t *smem_a_sf[K_SF_NUM_STAGE];
  uint32_t *smem_b_sf[K_SF_NUM_STAGE];
  int mma_consumer_phase[K_NUM_STAGE] = {0};
#pragma unroll
  for (int i = 0; i < K_NUM_STAGE; ++i) {
    smem_a[i] = (__nv_fp4x2_e2m1 *)(smem_buffer + SMEM_D_SIZE + i * SMEM_A_SIZE);
    smem_b[i] = (__nv_fp4x2_e2m1 *)(smem_buffer + SMEM_D_SIZE + K_NUM_STAGE * SMEM_A_SIZE + i * SMEM_B_SIZE);
    load_full_mbar[i] = (uint64_t *)(smem_buffer + SMEM_D_SIZE + K_NUM_STAGE * (SMEM_A_SIZE + SMEM_B_SIZE) + i * SMEM_MBAR_SIZE + warpid * sizeof(uint64_t));
  }
#pragma unroll
  for (int i = 0; i < K_SF_NUM_STAGE; ++i) {
    smem_a_sf[i] = (uint32_t *)(smem_buffer + SMEM_D_SIZE + K_NUM_STAGE * (SMEM_A_SIZE + SMEM_B_SIZE + SMEM_MBAR_SIZE) + i * SMEM_A_SF_SIZE);
    smem_b_sf[i] = (uint32_t *)(smem_buffer + SMEM_D_SIZE + K_NUM_STAGE * (SMEM_A_SIZE + SMEM_B_SIZE + SMEM_MBAR_SIZE) + K_SF_NUM_STAGE * SMEM_A_SF_SIZE + i * SMEM_B_SF_SIZE);
  }

  uint32_t Ra[4];
  uint32_t Rb[2];
  float Rc[WARP_M_TILES][WARP_N_TILES][4];
  uint32_t sf_a, sf_b;
  __nv_bfloat162 Rd[2];

  if (0 == laneid) {
#pragma unroll
    for (int i = 0; i < K_NUM_STAGE; ++i) {
      // Init full_mbarrier with number of producers
      init_mbarrier(load_full_mbar[i], 32 + 16 + 16 + 8);
    }
  }
  __syncthreads();

  const int warp_m = warpid / BLK_N_WARPS;
  const int warp_m_start = blk_m_start + warp_m * WARP_M;
  const int warp_m_end = min(blk_m_end, warp_m_start + WARP_M);
  const int warp_n = warpid % BLK_N_WARPS;
  const int warp_n_start = blk_n_start + warp_n * WARP_N;
  const int warp_n_end = min(blk_n_end, warp_n_start + WARP_N);
  if (warp_m_start < warp_m_end && warp_n_start < warp_n_end) {
    const int warp_m_tiles = ceil_div(warp_m_end - warp_m_start, MMA_M);
    const int warp_n_tiles = ceil_div(warp_n_end - warp_n_start, MMA_N);
    // 初始化Rc寄存器
#pragma unroll
    for (int mma_m = 0; mma_m < warp_m_tiles; ++mma_m) {
#pragma unroll
      for (int mma_n = 0; mma_n < warp_n_tiles; ++mma_n) {
        Rc[mma_m][mma_n][0] = 0.0f;
        Rc[mma_m][mma_n][1] = 0.0f;
        Rc[mma_m][mma_n][2] = 0.0f;
        Rc[mma_m][mma_n][3] = 0.0f;
      }
    }

    // 0~(K_NUM_STAGE-1)，Load A/B/A_sf/B_sf
    int load_buf_id = 0;
    int load_buf_sf_id = 0;
#pragma unroll
    for (int mma_k = 0; mma_k < min(K_NUM_STAGE, mma_k_tiles); ++mma_k) {
      //// Load A
      arrive_and_expect_tx_mbarrier(load_full_mbar[load_buf_id], warp_m_tiles * 16);
#pragma unroll
      for (int mma_m = 0; mma_m < warp_m_tiles; ++mma_m) {
        // 虽然sm_120a也能支持cp.async.bulk.tensor，但是，消费卡没有TMA硬件。所以，这里选择使用cp.async.bulk来传输数据
        // swizzle参考：https://blog.csdn.net/qq_40672115/article/details/151025091
        cp_async(&A[(warp_m_start + mma_m * MMA_M + laneid / 2) * (K / 2) + (mma_k * MMA_K / 2) + (laneid % 2) * 16], &smem_a[load_buf_id][swizzle<1, 4, 3>((warp_m * warp_m_tiles + mma_m) * MMA_M * MMA_K / 2 + laneid * 16)], load_full_mbar[load_buf_id], 16);
      }
      if (laneid < 16) {
        //// Load B
        arrive_and_expect_tx_mbarrier(load_full_mbar[load_buf_id], warp_n_tiles * 16);
#pragma unroll
        for (int mma_n = 0; mma_n < warp_n_tiles; ++mma_n) {
          cp_async(&B[(warp_n_start + mma_n * MMA_N + laneid / 2) * (K / 2) + (mma_k * MMA_K / 2) + (laneid % 2) * 16], &smem_b[load_buf_id][swizzle<1, 4, 3>((warp_n * warp_n_tiles + mma_n) * MMA_N * MMA_K / 2 + laneid * 16)], load_full_mbar[load_buf_id], 16);
        }
        //// Load A_sf
        if (0 == (mma_k & 3)) {
          arrive_and_expect_tx_mbarrier(load_full_mbar[load_buf_id], warp_m_tiles * 16);
#pragma unroll
          for (int mma_m = 0; mma_m < warp_m_tiles; ++mma_m) {
            cp_async(&A_sf[(warp_m_start + mma_m * MMA_M + laneid) * (K / 16) + mma_k * 4], &smem_a_sf[load_buf_sf_id][((warp_m * warp_m_tiles + mma_m) * MMA_M + laneid) * 4], load_full_mbar[load_buf_id], 16);
          }
        } else {
          arrive_and_expect_tx_mbarrier(load_full_mbar[load_buf_id], 0);
        }
      }
      if (laneid < 8) {
        //// Load B_sf
        if (0 == (mma_k & 3)) {
          arrive_and_expect_tx_mbarrier(load_full_mbar[load_buf_id], warp_n_tiles * 16);
#pragma unroll
          for (int mma_n = 0; mma_n < warp_n_tiles; ++mma_n) {
            cp_async(&B_sf[(warp_n_start + mma_n * MMA_N + laneid) * (K / 16) + mma_k * 4], &smem_b_sf[load_buf_sf_id][((warp_n * warp_n_tiles + mma_n) * MMA_N + laneid) * 4], load_full_mbar[load_buf_id], 16);
          }
        } else {
          arrive_and_expect_tx_mbarrier(load_full_mbar[load_buf_id], 0);
        }
      }
      load_buf_id = (load_buf_id + 1) % K_NUM_STAGE;
      if (3 == (mma_k & 3)) {
        load_buf_sf_id = (load_buf_sf_id + 1) % K_SF_NUM_STAGE;
      }
    }

    // 0~(mma_k_tiles-K_NUM_STAGE-1)，MMA; K_NUM_STAGE~(mma_k_tiles-1)，Load A/B/A_sf/B_sf
    int mma_buf_id = 0;
    int mma_buf_sf_id = 0;
#pragma unroll
    for (int mma_k = 0; mma_k < (mma_k_tiles - K_NUM_STAGE); ++mma_k) {
      int mma_k_load = mma_k + K_NUM_STAGE;
      wait_mbarrier(load_full_mbar[mma_buf_id], mma_consumer_phase[mma_buf_id]);
      //// ldmatrix & MMA
#pragma unroll
      for (int mma_m = 0; mma_m < warp_m_tiles; ++mma_m) {
        // ldmatrix & stmatrix参考：https://zhuanlan.zhihu.com/p/1892538766796780877
        ldmatrix(&smem_a[mma_buf_id][swizzle<1, 4, 3>((warp_m * warp_m_tiles + mma_m) * MMA_M * MMA_K / 2 + (laneid % 16) * 32 + (laneid / 16) * 16)], Ra[0], Ra[1], Ra[2], Ra[3]);
        // sf layout参考：https://zhuanlan.zhihu.com/p/1933211176004727485
        sf_a = smem_a_sf[mma_buf_sf_id][((warp_m * warp_m_tiles + mma_m) * MMA_M + ((laneid % 2) * 8 + laneid / 4)) * 4 + (mma_k & 3)];
#pragma unroll
        for (int mma_n = 0; mma_n < warp_n_tiles; ++mma_n) {
          ldmatrix(&smem_b[mma_buf_id][swizzle<1, 4, 3>((warp_n * warp_n_tiles + mma_n) * MMA_N * MMA_K / 2 + (laneid % 8) * 32 + (laneid / 8) * 16)], Rb[0], Rb[1]);
          sf_b = smem_b_sf[mma_buf_sf_id][((warp_n * warp_n_tiles + mma_n) * MMA_N + (laneid / 4)) * 4 + (mma_k & 3)];
          mma_nvfp4(Rc[mma_m][mma_n][0], Rc[mma_m][mma_n][1], Rc[mma_m][mma_n][2], Rc[mma_m][mma_n][3], Ra[0], Ra[1], Ra[2], Ra[3], Rb[0], Rb[1], Rc[mma_m][mma_n][0], Rc[mma_m][mma_n][1], Rc[mma_m][mma_n][2], Rc[mma_m][mma_n][3], sf_a, sf_b);
        }
      }
      mma_consumer_phase[mma_buf_id] ^= 1;

      //// Load A
      arrive_and_expect_tx_mbarrier(load_full_mbar[load_buf_id], warp_m_tiles * 16);
#pragma unroll
      for (int mma_m = 0; mma_m < warp_m_tiles; ++mma_m) {
        // 虽然sm_120a也能支持cp.async.bulk.tensor，但是，消费卡没有TMA硬件。所以，这里选择使用cp.async.bulk来传输数据
        // swizzle参考：https://blog.csdn.net/qq_40672115/article/details/151025091
        cp_async(&A[(warp_m_start + mma_m * MMA_M + laneid / 2) * (K / 2) + (mma_k_load * MMA_K / 2) + (laneid % 2) * 16], &smem_a[load_buf_id][swizzle<1, 4, 3>((warp_m * warp_m_tiles + mma_m) * MMA_M * MMA_K / 2 + laneid * 16)], load_full_mbar[load_buf_id], 16);
      }
      if (laneid < 16) {
        //// Load B
        arrive_and_expect_tx_mbarrier(load_full_mbar[load_buf_id], warp_n_tiles * 16);
#pragma unroll
        for (int mma_n = 0; mma_n < warp_n_tiles; ++mma_n) {
          cp_async(&B[(warp_n_start + mma_n * MMA_N + laneid / 2) * (K / 2) + (mma_k_load * MMA_K / 2) + (laneid % 2) * 16], &smem_b[load_buf_id][swizzle<1, 4, 3>((warp_n * warp_n_tiles + mma_n) * MMA_N * MMA_K / 2 + laneid * 16)], load_full_mbar[load_buf_id], 16);
        }
        // Load A_sf
        if (0 == (mma_k_load & 3)) {
          arrive_and_expect_tx_mbarrier(load_full_mbar[load_buf_id], warp_m_tiles * 16);
#pragma unroll
          for (int mma_m = 0; mma_m < warp_m_tiles; ++mma_m) {
            cp_async(&A_sf[(warp_m_start + mma_m * MMA_M + laneid) * (K / 16) + mma_k_load * 4], &smem_a_sf[load_buf_sf_id][((warp_m * warp_m_tiles + mma_m) * MMA_M + laneid) * 4], load_full_mbar[load_buf_id], 16);
          }
        } else {
          arrive_and_expect_tx_mbarrier(load_full_mbar[load_buf_id], 0);
        }
      }
      if (laneid < 8) {
        // Load B_sf
        if (0 == (mma_k_load & 3)) {
          arrive_and_expect_tx_mbarrier(load_full_mbar[load_buf_id], warp_n_tiles * 16);
#pragma unroll
          for (int mma_n = 0; mma_n < warp_n_tiles; ++mma_n) {
            cp_async(&B_sf[(warp_n_start + mma_n * MMA_N + laneid) * (K / 16) + mma_k_load * 4], &smem_b_sf[load_buf_sf_id][((warp_n * warp_n_tiles + mma_n) * MMA_N + laneid) * 4], load_full_mbar[load_buf_id], 16);
          }
        } else {
          arrive_and_expect_tx_mbarrier(load_full_mbar[load_buf_id], 0);
        }
      }
      mma_buf_id = (mma_buf_id + 1) % K_NUM_STAGE;
      if (3 == (mma_k & 3)) {
        mma_buf_sf_id = (mma_buf_sf_id + 1) % K_SF_NUM_STAGE;
      }
      load_buf_id = (load_buf_id + 1) % K_NUM_STAGE;
      if (3 == (mma_k_load & 3)) {
        load_buf_sf_id = (load_buf_sf_id + 1) % K_SF_NUM_STAGE;
      }
    }

    // (mma_k_tiles-K_NUM_STAGE)~(mma_k_tiles-1)，MMA
#pragma unroll
    for (int mma_k = 0; mma_k < min(K_NUM_STAGE, mma_k_tiles); ++mma_k) {
      int mma_k_calc = mma_k + max(0, mma_k_tiles - K_NUM_STAGE);
      wait_mbarrier(load_full_mbar[mma_buf_id], mma_consumer_phase[mma_buf_id]);
      //// ldmatrix & MMA
#pragma unroll
      for (int mma_m = 0; mma_m < warp_m_tiles; ++mma_m) {
        // ldmatrix & stmatrix参考：https://zhuanlan.zhihu.com/p/1892538766796780877
        ldmatrix(&smem_a[mma_buf_id][swizzle<1, 4, 3>((warp_m * warp_m_tiles + mma_m) * MMA_M * MMA_K / 2 + (laneid % 16) * 32 + (laneid / 16) * 16)], Ra[0], Ra[1], Ra[2], Ra[3]);
        // sf layout参考：https://zhuanlan.zhihu.com/p/1933211176004727485
        sf_a = smem_a_sf[mma_buf_sf_id][((warp_m * warp_m_tiles + mma_m) * MMA_M + ((laneid % 2) * 8 + laneid / 4)) * 4 + (mma_k_calc & 3)];
#pragma unroll
        for (int mma_n = 0; mma_n < warp_n_tiles; ++mma_n) {
          ldmatrix(&smem_b[mma_buf_id][swizzle<1, 4, 3>((warp_n * warp_n_tiles + mma_n) * MMA_N * MMA_K / 2 + (laneid % 8) * 32 + (laneid / 8) * 16)], Rb[0], Rb[1]);
          sf_b = smem_b_sf[mma_buf_sf_id][((warp_n * warp_n_tiles + mma_n) * MMA_N + (laneid / 4)) * 4 + (mma_k_calc & 3)];
          mma_nvfp4(Rc[mma_m][mma_n][0], Rc[mma_m][mma_n][1], Rc[mma_m][mma_n][2], Rc[mma_m][mma_n][3], Ra[0], Ra[1], Ra[2], Ra[3], Rb[0], Rb[1], Rc[mma_m][mma_n][0], Rc[mma_m][mma_n][1], Rc[mma_m][mma_n][2], Rc[mma_m][mma_n][3], sf_a, sf_b);
        }
      }
      mma_consumer_phase[mma_buf_id] ^= 1;
      mma_buf_id = (mma_buf_id + 1) % K_NUM_STAGE;
      if (3 == (mma_k_calc & 3)) {
        mma_buf_sf_id = (mma_buf_sf_id + 1) % K_SF_NUM_STAGE;
      }
    }
    cp_async_wait_group<0>();
#pragma unroll
    for (int mma_m = 0; mma_m < warp_m_tiles; ++mma_m) {
#pragma unroll
      for (int mma_n = 0; mma_n < warp_n_tiles; ++mma_n) {
        Rd[0] = __float22bfloat162_rn({Rc[mma_m][mma_n][0] / scale, Rc[mma_m][mma_n][1] / scale});
        Rd[1] = __float22bfloat162_rn({Rc[mma_m][mma_n][2] / scale, Rc[mma_m][mma_n][3] / scale});
        stmatrix(((uint32_t *)Rd)[0], ((uint32_t *)Rd)[1], &(smem_d[((warp_m * warp_m_tiles + mma_m) * BLK_N_TILES + (warp_n * warp_n_tiles + mma_n)) * MMA_M * MMA_N + laneid * 8]));
        if (laneid < 16 && (warp_m_start + mma_m * MMA_M + laneid) < M) {
          cp_async_bulk_group(&smem_d[((warp_m * warp_m_tiles + mma_m) * BLK_N_TILES + (warp_n * warp_n_tiles + mma_n)) * MMA_M * MMA_N + laneid * 8], &D[(warp_m_start + mma_m * MMA_M + laneid) * N + (warp_n_start + mma_n * MMA_N)], 16);
        }
      }
    }
    cp_async_commit_group();
  }
}

void gemm_nvfp4(const __nv_fp4x2_e2m1 *A, const __nv_fp4x2_e2m1 *B,
                const __nv_fp8_e4m3 *A_sf, const __nv_fp8_e4m3 *B_sf,
                __nv_bfloat16 *D, const float scale,
                const int M, const int N, const int K, cudaStream_t stream) {
  assert(0 == (K % 256) && 0 == (N % 8));
  dim3 grid(ceil_div(M, BLK_M), ceil_div(N, BLK_N));
  dim3 block(THREADS_PER_BLK);
  cudaFuncSetAttribute(gemm_nvfp4_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, MAX_SMEN_SIZE);
  gemm_nvfp4_kernel<<<grid, block, MAX_SMEN_SIZE, stream>>>(A, B, A_sf, B_sf, D, M, N, K, scale);
  LAST_KERNEL_CHECK();
}
}  // namespace my_plugins