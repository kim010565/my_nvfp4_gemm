#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <mma.h>
#if defined(__CUDACC__)
#include <cuda/barrier>
#endif
#include <cuda/std/array>
#include <iostream>

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

#ifndef NUM_SMS
#define NUM_SMS 46
#endif

#ifndef MAX_SHARED_MEMORY
#define MAX_SHARED_MEMORY 101376
#endif

namespace my_plugins {
#if defined(__CUDACC__)
#define HOST_DEVICE_INLINE __host__ __device__ __forceinline__
#define DEVICE_INLINE __device__ __forceinline__
#define HOST_INLINE __host__ __forceinline__
#else
#define HOST_DEVICE_INLINE inline
#define DEVICE_INLINE inline
#define HOST_INLINE inline
#endif

#define CUDA_CHECK(call) cuda_check(call, __FILE__, __LINE__)  //  定义CUDA错误检查宏，用于检查CUDA API调用是否成功
#define LAST_KERNEL_CHECK() kernel_check(__FILE__, __LINE__)   //  定义最后一次内核调用检查宏，用于检查最后一次执行的CUDA内核是否成功

HOST_INLINE void cuda_check(cudaError_t err, const char *file, const int line) {
  if (err != cudaSuccess) {
    printf("cuda ERROR: %s:%d, ", file, line);
    printf("CODE:%s, DETAIL:%s\n", cudaGetErrorName(err), cudaGetErrorString(err));
    exit(1);
  }
}

HOST_INLINE void kernel_check(const char *file, const int line) {
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("kernel ERROR: %s:%d, ", file, line);
    printf("CODE:%s, DETAIL:%s\n", cudaGetErrorName(err), cudaGetErrorString(err));
    exit(1);
  }
}

template <typename T>
HOST_DEVICE_INLINE T ceil_div(T a, T b) {
  return (a + b - 1) / b;
}

#if defined(__CUDACC__)
DEVICE_INLINE uint32_t convert_fp32_to_e2m1(const float *source) {
  uint32_t out;
  asm volatile(
      "{\n"
      ".reg .b8 byte0;\n"
      ".reg .b8 byte1;\n"
      ".reg .b8 byte2;\n"
      ".reg .b8 byte3;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte0, %2, %1;\n"  // satfinite表示超过max_norm，则保护为max_norm
      "cvt.rn.satfinite.e2m1x2.f32   byte1, %4, %3;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte2, %6, %5;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte3, %8, %7;\n"
      "mov.b32 %0, {byte0, byte1, byte2, byte3};\n"
      "}"
      : "=r"(out)
      : "f"(source[0]), "f"(source[1]), "f"(source[2]), "f"(source[3]),
        "f"(source[4]), "f"(source[5]), "f"(source[6]), "f"(source[7]));
  return out;
}

DEVICE_INLINE __nv_fp8_e4m3 convert_fp32_to_e4m3(float const &x) {
  uint16_t tmp;
  asm volatile(
      "cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;\n"
      : "=h"(tmp)
      : "f"(0.0f), "f"(x));
  return *(__nv_fp8_e4m3 *)(&tmp);
}

DEVICE_INLINE cuda::std::array<float, 8> convert_e2m1_to_fp32(uint32_t const &source) {
  uint32_t out_fp16[4];
  asm volatile(
      "{\n"
      ".reg .b8 byte0, byte1, byte2, byte3;\n"
      "mov.b32 {byte0, byte1, byte2, byte3}, %4;\n"
      "cvt.rn.f16x2.e2m1x2 %0, byte0;\n"
      "cvt.rn.f16x2.e2m1x2 %1, byte1;\n"
      "cvt.rn.f16x2.e2m1x2 %2, byte2;\n"
      "cvt.rn.f16x2.e2m1x2 %3, byte3;\n"
      "}\n" : "=r"(out_fp16[0]), "=r"(out_fp16[1]), "=r"(out_fp16[2]), "=r"(out_fp16[3]) : "r"(source));  // cvt不支持直接从e2m1转fp32，所以，先将e2m1转换为f16，再转换为fp32
  float2 res0 = __half22float2(reinterpret_cast<__half2 &>(out_fp16[0]));
  float2 res1 = __half22float2(reinterpret_cast<__half2 &>(out_fp16[1]));
  float2 res2 = __half22float2(reinterpret_cast<__half2 &>(out_fp16[2]));
  float2 res3 = __half22float2(reinterpret_cast<__half2 &>(out_fp16[3]));
  cuda::std::array<float, 8> out;
  out[0] = res0.x;
  out[1] = res0.y;
  out[2] = res1.x;
  out[3] = res1.y;
  out[4] = res2.x;
  out[5] = res2.y;
  out[6] = res3.x;
  out[7] = res3.y;
  return out;
}

DEVICE_INLINE float convert_e4m3_to_fp32(__nv_fp8_e4m3 const &x) {
  uint16_t bits = x.__x;
  uint32_t packed;
  asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;\n" : "=r"(packed) : "h"(bits));  // cvt不支持直接从e4m3转fp32，所以，先将e4m3转换为f16，再转换为fp32
  return __half2float(reinterpret_cast<half2 const &>(packed).x);
}

// Initializing the current phase to 0.
// Initializing the expected arrival count to count. (1 ~ 2^20-1)
// Initializing the pending arrival count to count. (0 ~ 2^20-1)
// Initializing the tx-count to 0. (-(2^20-1) ~ 2^20-1)
DEVICE_INLINE void init_mbarrier(uint64_t *mbar, int count) {
  uint32_t mbar_int = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(mbar_int), "r"(count));
}

// the pending arrival count is decremented by count.
// If the current phase has been completed then the mbarrier transitions to the next phase. (The pending arrival count is reinitialized to the expected arrival count.)
DEVICE_INLINE void arrive_mbarrier(uint64_t *mbar) {
  uint32_t mbar_int = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
  asm volatile(
      "{\n"
      ".reg .b64 state; \n"
      "mbarrier.arrive.shared::cta.b64 state, [%0];\n"
      "}\n" ::"r"(mbar_int));
}

// an expect-tx operation is performed prior to the arrive-on operation.
// increases the tx-count of an mbarrierobject by the value specified by expectCount
DEVICE_INLINE void arrive_and_expect_tx_mbarrier(uint64_t *mbar, uint32_t expect_count) {
  uint32_t mbar_int = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(mbar_int), "r"(expect_count));
}

// mbarrier.try_wait is a potentially blocking instruction which tests for the completion of the phase.
// If the phase is not complete, the executing thread may be suspended.
// the valid values of phaseParity operand are 0 and 1.
DEVICE_INLINE void wait_mbarrier(uint64_t *mbar, int phase) {
  uint32_t mbar_int = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
  asm volatile(
      "{\n"
      ".reg .pred P1;\n"
      "LAB_WAIT:\n"
      "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
      "@P1 bra DONE;\n"
      "bra LAB_WAIT;\n"
      "DONE:\n"
      "}\n" ::"r"(mbar_int),
      "r"(phase));
}

// tx-count is decremented by completeCount
DEVICE_INLINE void complete_tx_mbarrier(uint64_t *mbar, int32_t tx_ount = 1) {
  uint32_t mbar_int = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
  asm volatile(
      "mbarrier.complete_tx.shared::cta.b64 [%0], %1;\n"
      :
      : "r"(mbar_int), "r"(tx_ount));
}

// tx-count is decremented by completeCount(equal to amount of data copied in bytes)
// If the current phase has been completed then the mbarrier transitions to the next phase.
DEVICE_INLINE void cp_async(void const *gmem, void *smem, uint64_t *mbar, int32_t bytes) {
  uint32_t mbar_int = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
  uint32_t smem_int = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
               :
               : "r"(smem_int), "l"(gmem), "r"(bytes), "r"(mbar_int)
               : "memory");
}

DEVICE_INLINE void cp_async_bulk_group(void const *smem, void *gmem, int32_t bytes) {
  uint32_t smem_int = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;\n"
               :
               : "l"(gmem), "r"(smem_int), "r"(bytes)
               : "memory");
}

DEVICE_INLINE void cp_async_commit_group() {
  asm volatile("cp.async.bulk.commit_group;");
}

template <int count>
DEVICE_INLINE void cp_async_wait_group() {
  asm volatile(
      "cp.async.bulk.wait_group.read %0;"
      :
      : "n"(count)
      : "memory");
}

DEVICE_INLINE void ldmatrix(void const *smem_src, uint32_t &dst0, uint32_t &dst1, uint32_t &dst2, uint32_t &dst3) {
  uint32_t smem_int = static_cast<uint32_t>(__cvta_generic_to_shared(smem_src));
  asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(dst0), "=r"(dst1), "=r"(dst2), "=r"(dst3)
               : "r"(smem_int));
}

DEVICE_INLINE void ldmatrix(void const *smem_src, uint32_t &dst0, uint32_t &dst1) {
  uint32_t smem_int = static_cast<uint32_t>(__cvta_generic_to_shared(smem_src));
  asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n"
               : "=r"(dst0), "=r"(dst1)
               : "r"(smem_int));
}

DEVICE_INLINE void stmatrix(uint32_t const &src0, uint32_t const &src1, void *smem_dst) {
  uint32_t smem_int = static_cast<uint32_t>(__cvta_generic_to_shared(smem_dst));
  asm volatile("stmatrix.sync.aligned.x2.m8n8.shared.b16 [%0], {%1, %2};\n" ::"r"(smem_int),
               "r"(src0), "r"(src1));
}

DEVICE_INLINE void mma_nvfp4(float &d0, float &d1, float &d2, float &d3,
                             uint32_t const &a0, uint32_t const &a1, uint32_t const &a2, uint32_t const &a3,
                             uint32_t const &b0, uint32_t const &b1,
                             float const &c0, float const &c1, float const &c2, float const &c3,
                             uint32_t const &sfa0,
                             uint32_t const &sfb0) {
  constexpr uint16_t tidA = 0;
  constexpr uint16_t bidA = 0;
  constexpr uint16_t tidB = 0;
  constexpr uint16_t bidB = 0;
  asm volatile(
      "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5,  %6,  %7},"
      "{%8,  %9},"
      "{%10, %11, %12, %13},"
      "{%14},"
      "{%15, %16},"
      "{%17},"
      "{%18, %19};\n"
      : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
        "r"(b0), "r"(b1),
        "f"(c0), "f"(c1), "f"(c2), "f"(c3),
        "r"(uint32_t(sfa0)), "h"(bidA), "h"(tidA),
        "r"(uint32_t(sfb0)), "h"(bidB), "h"(tidB));
}

// 参考cutlass\include\cute\swizzle.hpp
// A generic Swizzle functor
/* 0bxxxxxxxxxxxxxxxYYYxxxxxxxZZZxxxx
 *                               ^--^ M is the number of least-sig bits to keep constant
 *                  ^-^       ^-^     B is the number of bits in the mask
 *                    ^---------^     S is the distance to shift the YYY mask
 *                                    (pos shifts YYY to the right, neg shifts YYY to the left)
 *
 * e.g. Given
 * 0bxxxxxxxxxxxxxxxxYYxxxxxxxxxZZxxx
 * the result is
 * 0bxxxxxxxxxxxxxxxxYYxxxxxxxxxAAxxx where AA = ZZ xor YY
 */
// 32 byte swizzle mode:  Swizzle<1, 4, 3> @ Byte
// 64 byte swizzle mode:  Swizzle<2, 4, 3> @ Byte
// 128 byte swizzle mode: Swizzle<3, 4, 3> @ Byte
template <uint32_t B, uint32_t M, uint32_t S>
DEVICE_INLINE uint32_t swizzle(uint32_t const &offset) {
  constexpr uint32_t bit_msk = (1 << B) - 1;
  constexpr uint32_t yyy_msk = bit_msk << (M + S);
  return offset ^ ((offset & yyy_msk) >> S);  // ZZZ ^= YYY
}
#endif
}  // namespace my_plugins

using namespace my_plugins;  //  使用my_plugins命名空间
