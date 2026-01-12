#include "quantize_nvfp4.h"

namespace my_plugins {
// quantize
// __restrict__是CUDA中的一个重要关键字，用于向编译器提供额外的内存别名信息，从而帮助编译器生成更优化的代码
__global__ void quantize_nvfp4_kernel(const __nv_bfloat162 *__restrict__ input, const float input_scale, const int input_numel, uint8_t *__restrict__ output, __nv_fp8_e4m3 *__restrict__ output_sf) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= (input_numel / 8)) {
    return;
  }
  constexpr float e2m1_max_norm = 6.0f;
  constexpr float rcp_e2m1_max_norm = 1.0f / e2m1_max_norm;
  float2 tmp_fp32[8];
  __nv_bfloat162 tmp_amax;
  ((uint16_t *)(&tmp_amax.x))[0] = 1;  // 非0保护
  ((uint16_t *)(&tmp_amax.x))[1] = 1;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    __nv_bfloat162 tmp = input[idx * 8 + i];
    tmp_amax = __hmax2_nan(tmp_amax, __habs2(tmp));
    tmp_fp32[i] = __bfloat1622float2(tmp);  // cvt不支持直接从bf16转e2m1，所以，先将bf16转换为fp32，再转换为e2m1;
  }
  __nv_bfloat16 input_amax = __hmax_nan(tmp_amax.x, tmp_amax.y);
  float sf = __bfloat162float(input_amax) * input_scale * rcp_e2m1_max_norm;
  __nv_fp8_e4m3 sf_tmp = convert_fp32_to_e4m3(sf);
  output_sf[idx] = sf_tmp;
  float output_scale = __frcp_rn(convert_e4m3_to_fp32(sf_tmp) * __frcp_rn(input_scale));
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    tmp_fp32[i].x *= output_scale;
    tmp_fp32[i].y *= output_scale;
  }
  ((uint32_t *)output)[idx * 2] = convert_fp32_to_e2m1((const float *)&tmp_fp32[0]);
  ((uint32_t *)output)[idx * 2 + 1] = convert_fp32_to_e2m1((const float *)&tmp_fp32[4]);
}

void quantize_nvfp4(const __nv_bfloat162 *input, const float input_scale, const int input_numel, uint8_t *output, __nv_fp8_e4m3 *output_sf, cudaStream_t stream) {
  constexpr int block_size = 256;
  const int grid_size = ceil_div(input_numel, block_size * 8);  // 为了避免线程间通信，每个线程处理一个Block Scaling(16个元素)
  quantize_nvfp4_kernel<<<grid_size, block_size, 0, stream>>>(input, input_scale, input_numel, output, output_sf);
}

// de_quantize
__global__ void de_quantize_nvfp4_kernel(const uint8_t *__restrict__ input, const float input_scale, const __nv_fp8_e4m3 *__restrict__ input_sf, const int input_numel, __nv_bfloat162 *__restrict__ output) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= (input_numel / 8)) {
    return;
  }
  float scale = convert_e4m3_to_fp32(input_sf[idx]) * __frcp_rn(input_scale);
  cuda::std::array<float, 8> tmp0 = convert_e2m1_to_fp32(((uint32_t *)input)[idx * 2]);
  cuda::std::array<float, 8> tmp1 = convert_e2m1_to_fp32(((uint32_t *)input)[idx * 2 + 1]);
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    tmp0[i] *= scale;
    tmp1[i] *= scale;
  }
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    output[idx * 8 + i] = __float22bfloat162_rn({tmp0[2 * i], tmp0[2 * i + 1]});
    output[idx * 8 + i + 4] = __float22bfloat162_rn({tmp1[2 * i], tmp1[2 * i + 1]});
  }
}

void de_quantize_nvfp4(const uint8_t *input, const float input_scale, const __nv_fp8_e4m3 *input_sf, const int input_numel, __nv_bfloat162 *output, cudaStream_t stream) {
  constexpr int block_size = 256;
  const int grid_size = ceil_div(input_numel, block_size * 8);  // 为了避免线程间通信，每个线程处理一个Block Scaling(16个元素)
  de_quantize_nvfp4_kernel<<<grid_size, block_size, 0, stream>>>(input, input_scale, input_sf, input_numel, output);
}
}  // namespace my_plugins