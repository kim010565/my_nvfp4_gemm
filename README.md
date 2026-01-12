# my_nvfp4_gemm
本项目为基于NVIDIA 50系列显卡实现NVFP4精度的矩阵乘法：
1、通过pybind11将C++代码封装为python接口，方便调用
2、在SM120a上，使用mbarrier的cp.async.bulk+mma实现矩阵乘法gemm_nvfp4（由于仅cp支持异步，所以只有full_mbar，无empty_mbar）
3、实现了bf16 <-> nvfp4的转换quantize_nvfp4和de_quantize_nvfp4