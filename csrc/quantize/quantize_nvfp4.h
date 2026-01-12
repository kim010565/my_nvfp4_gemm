#pragma once

#include "../common/common.h"

namespace my_plugins {
void quantize_nvfp4(const __nv_bfloat162 *input, const float input_scale, const int input_numel, uint8_t *output, __nv_fp8_e4m3 *output_sf, cudaStream_t stream);
void de_quantize_nvfp4(const uint8_t *input, const float input_scale, const __nv_fp8_e4m3 *input_sf, const int input_numel, __nv_bfloat162 *output, cudaStream_t stream);
}