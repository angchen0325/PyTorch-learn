#include <torch/torch.h>
#include <iostream>

int main() {
  // 使用arange构造一个一维向量，再用reshape变换到5x5的矩阵
  torch::Tensor foo = torch::arange(36).reshape({6, 6});

  // 计算矩阵的迹
  torch::Tensor bar = torch::einsum("ii", foo);

  // 输出矩阵和对应的迹
  std::cout << "==> matrix is:\n " << foo << std::endl;
  std::cout << "==> trace of it is:\n " << bar << std::endl;
}