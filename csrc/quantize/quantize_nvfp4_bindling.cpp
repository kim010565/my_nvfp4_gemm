#include "quantize_nvfp4_bindling.h"

// bf16 -> nvfp4量化
std::tuple<torch::Tensor, torch::Tensor> py_quantize_nvfp4(torch::Tensor const &input, torch::Tensor const &input_scale) {
  TORCH_CHECK(input.dtype() == torch::kBFloat16 && input.is_contiguous(), "input must be bf16 and contiguous");
  TORCH_CHECK(input_scale.dtype() == torch::kFloat32 && 0 == input_scale.dim(), "input_scale must be float32 and scalar");
  TORCH_CHECK(input.device().is_cuda(), "input must be on cuda device");

  auto output_sizes = input.sizes().vec();
  auto output_sf_sizes = input.sizes().vec();
  output_sizes[input.dim() - 1] /= 2;
  output_sf_sizes[input.dim() - 1] /= 16;
  auto output = torch::empty(output_sizes, torch::dtype(torch::kUInt8).device(input.device()));
  auto output_sf = torch::empty(output_sf_sizes, torch::dtype(torch::kFloat8_e4m3fn).device(input.device()));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(input.device().index());

  quantize_nvfp4((const __nv_bfloat162 *)input.data_ptr(), input_scale.item<float>(), input.numel() / 2, (uint8_t *)output.data_ptr(), (__nv_fp8_e4m3 *)output_sf.data_ptr(), stream);

  return std::make_tuple(output, output_sf);
}

// nvfp4 -> bf16反量化
torch::Tensor py_de_quantize_nvfp4(torch::Tensor const &input, torch::Tensor const &input_scale, torch::Tensor const &input_sf) {
  TORCH_CHECK(input.dtype() == torch::kUInt8 && input.is_contiguous(), "input must be uint8 and contiguous");
  TORCH_CHECK(input_scale.dtype() == torch::kFloat32 && 0 == input_scale.dim(), "input_scale must be float32 and scalar");
  TORCH_CHECK(input_sf.dtype() == torch::kFloat8_e4m3fn && input_sf.is_contiguous(), "input_sf must be float8_e4m3 and contiguous");
  TORCH_CHECK(input.device().is_cuda() && input_sf.device().is_cuda(), "input&input_sf must be on cuda device");

  auto output_sizes = input.sizes().vec();
  output_sizes[input.dim() - 1] *= 2;
  auto output = torch::empty(output_sizes, torch::dtype(torch::kBFloat16).device(input.device()));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(input.device().index());

  de_quantize_nvfp4((const uint8_t *)input.data_ptr(), input_scale.item<float>(), (const __nv_fp8_e4m3 *)input_sf.data_ptr(), input.numel(), (__nv_bfloat162 *)output.data_ptr(), stream);

  return output;
}

//  quantize_nvfp4绑定函数，用于Python模块
void quantize_nvfp4_bindling(py::module &m) {
  //  为Python模块添加quantize_nvfp4量化、反量化相关的函数
  //  py::overload_cast是pybind11中用于处理C++函数重载的工具，主要解决Python调用C++函数时参数类型不匹配的问题
  m.def("quantize_nvfp4", py::overload_cast<torch::Tensor const &, torch::Tensor const &>(&py_quantize_nvfp4), "quantize_nvfp4");
  m.def("de_quantize_nvfp4", py::overload_cast<torch::Tensor const &, torch::Tensor const &, torch::Tensor const &>(&py_de_quantize_nvfp4), "de_quantize_nvfp4");
}