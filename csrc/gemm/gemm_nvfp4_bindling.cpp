#include "gemm_nvfp4_bindling.h"

// D(bf16) = scale(fp32) * (A(nvfp4)*A_sf(fp8_e4m3) * B(nvfp4)*B_sf(fp8_e4m3))
torch::Tensor py_gemm_nvfp4(torch::Tensor const &A, torch::Tensor const &B, torch::Tensor const &A_sf, torch::Tensor const &B_sf, torch::Tensor const &scale) {
  TORCH_CHECK(A.dtype() == torch::kUInt8 && A.is_contiguous(), "A must be uint8 and contiguous");
  TORCH_CHECK(B.dtype() == torch::kUInt8 && B.is_contiguous(), "B must be uint8 and contiguous");
  TORCH_CHECK(2 == A.dim() && 2 == B.dim(), "A and B must be 2D");
  TORCH_CHECK(A_sf.dtype() == torch::kFloat8_e4m3fn && A_sf.is_contiguous(), "A_sf must be float8_e4m3 and contiguous");
  TORCH_CHECK(B_sf.dtype() == torch::kFloat8_e4m3fn && B_sf.is_contiguous(), "B_sf must be float8_e4m3 and contiguous");
  TORCH_CHECK(scale.dtype() == torch::kFloat32 && 0 == scale.dim(), "scale must be float32 and scalar");
  TORCH_CHECK(A.device().is_cuda() && B.device().is_cuda() && A_sf.device().is_cuda() && B_sf.device().is_cuda(), "A&B&A_sf&B_sf must be on cuda device");

  const int M = A.size(0);
  const int N = B.size(0);
  const int K = A.size(1) * 2;
  auto D = torch::empty({M, N}, torch::dtype(torch::kBFloat16).device(A.device()));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(A.device().index());

  gemm_nvfp4((const __nv_fp4x2_e2m1 *)A.data_ptr(), (const __nv_fp4x2_e2m1 *)B.data_ptr(),
             (const __nv_fp8_e4m3 *)A_sf.data_ptr(), (const __nv_fp8_e4m3 *)B_sf.data_ptr(),
             (__nv_bfloat16 *)D.data_ptr(), scale.item<float>(),
             M, N, K, stream);

  return D;
}

//  gemm_nvfp4绑定函数，用于Python模块
void gemm_nvfp4_bindling(py::module &m) {
  //  为Python模块添加gemm_nvfp4量化、反量化相关的函数
  //  py::overload_cast是pybind11中用于处理C++函数重载的工具，主要解决Python调用C++函数时参数类型不匹配的问题
  m.def("gemm_nvfp4", py::overload_cast<torch::Tensor const &, torch::Tensor const &, torch::Tensor const &, torch::Tensor const &, torch::Tensor const &>(&py_gemm_nvfp4), "gemm_nvfp4");
}