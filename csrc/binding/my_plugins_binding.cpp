
#include "../gemm/gemm_nvfp4_bindling.h"
#include "../quantize/quantize_nvfp4_bindling.h"

PYBIND11_MODULE(my_plugins, m) {
  m.doc() = "my_plugins";  //  设置模块文档字符串

  quantize_nvfp4_bindling(m);  //  调用quantize函数的绑定函数，将quantize相关功能暴露给Python
  gemm_nvfp4_bindling(m);
}