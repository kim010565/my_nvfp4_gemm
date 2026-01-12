from csrc import my_plugins
import torch

torch.manual_seed(100)

if __name__ == "__main__":
    A = torch.randn(128, 256 * 4, dtype=torch.bfloat16).to("cuda")
    B = torch.randn(256, 256 * 4, dtype=torch.bfloat16).to("cuda")

    A_scale = 6 * 448 / A.abs().max().to(dtype=torch.float32)
    B_scale = 6 * 448 / B.abs().max().to(dtype=torch.float32)
    a, a_sf = my_plugins.quantize_nvfp4(A, A_scale)
    b, b_sf = my_plugins.quantize_nvfp4(B, B_scale)

    d = my_plugins.gemm_nvfp4(a, b, a_sf, b_sf, A_scale * B_scale)

    a_r = my_plugins.de_quantize_nvfp4(a, A_scale, a_sf)
    b_r = my_plugins.de_quantize_nvfp4(b, B_scale, b_sf)

    D = torch.nn.functional.linear(a_r, b_r)

    print(d)
    print(D)
    print(D - d)
