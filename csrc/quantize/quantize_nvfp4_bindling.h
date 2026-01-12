#pragma once

#include <Python.h>  //  Python API头文件
#include <pybind11/pybind11.h>
#include <torch/extension.h>  //  PyTorch扩展头文件
#include <tuple>
#include "quantize_nvfp4.h"

void quantize_nvfp4_bindling(py::module &m);  //  声明Python绑定函数，用于将C++函数绑定到Python模块