# pybind11的setup.py文件用于编译C++代码生成动态链接库，并安装为Python扩展模块
from setuptools import setup  #  导入setuptools模块，用于Python包的安装和分发
from torch.utils import cpp_extension  #  导入PyTorch的C++扩展工具，用于编译C++代码
import os, glob  #  导入os和glob模块，用于文件和目录操作
import torch  #  导入PyTorch库

current_dir = os.path.dirname(os.path.realpath(__file__))  #  获取当前文件所在的目录


def find_files(dir_lisr):  #  搜索源文件列表
    op_files = []
    for p in dir_lisr:
        current_dir_p = current_dir + "/" + p
        op_files = op_files + list(
            glob.iglob(f"{current_dir_p}/**/*.cu", recursive=True)
        )
        op_files = op_files + list(
            glob.iglob(f"{current_dir_p}/**/*.cpp", recursive=True)
        )
    return op_files


def get_extensions():
    extensions = []
    ext_name = "my_plugins"  #  定义扩展模块的名称
    os.environ.setdefault("MAX_JOBS", "8")  #  设置并行编译的最大任务数
    define_macros = []  #  定义预处理器宏
    include_path = [  #  定义头文件搜索路径列表
        current_dir,  #  当前目录
    ]

    op_files = find_files(["binding", "quantize", "gemm"])
    define_macros += [
        (
            "WARP_SIZE",
            str(torch.cuda.get_device_properties(device="cuda").warp_size),
        )
    ]  #  添加宏定义
    define_macros += [
        (
            "NUM_SMS",
            str(torch.cuda.get_device_properties(device="cuda").multi_processor_count),
        )
    ]  #  添加宏定义
    define_macros += [
        (
            "MAX_SHARED_MEMORY",
            str(
                torch.cuda.get_device_properties(
                    device="cuda"
                ).shared_memory_per_block_optin
            ),
        )
    ]  #  添加宏定义
    ext_ops = cpp_extension.CUDAExtension(  #  创建CUDA扩展模块
        name=ext_name,  #  扩展模块的名称
        sources=op_files,  #  源文件列表
        include_dirs=include_path,  #  头文件路径列表
        define_macros=define_macros,  #  宏定义列表
        extra_compile_args={
            "cxx": [
                "-O0",
                "-std=c++17",
                "-march=native",
            ],  #   C++编译器参数：最高优化级别、C++17标准、针对本地CPU架构优化
            "nvcc": [
                "-O0",
                "-Xcompiler",
                "-std=c++17",
                "-gencode",
                "arch=compute_120a,code=sm_120a",
            ],  #   NVIDIA CUDA编译器参数：最高优化级别、传递C/C++编译选项、C++17标准
        },
    )

    extensions.append(ext_ops)
    return extensions


setup(  #  设置包的配置信息
    name="my_plugins",  #  包的名称
    ext_modules=get_extensions(),  #  扩展模块列表
    include_package_data=False,
    cmdclass={"build_ext": cpp_extension.BuildExtension},  #  自定义构建命令类
)
