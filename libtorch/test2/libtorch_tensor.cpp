#include <torch/torch.h>
#include <iostream>


int main() {

    torch::Tensor foo = torch::tensor({1.0, 2.0, 3.0, 7.0});

    float arr[] = {1.0, 2.0, 3.0, 4.0};
    torch::Tensor bar1 = torch::from_blob(arr, {1,4});
    bar1 = torch::from_blob(arr, {2, 2});

    std::vector<float> v = {1.0, 4.0, 5.0, 2.0};
    torch::Tensor bar2 = torch::from_blob(v.data(), {2, 2});


    // std::cout << "==> foo is:\n " << foo << std::endl;
    // std::cout << "==> bar1 is:\n " << bar1 << std::endl;
    // std::cout << "==> bar2 is:\n " << bar2 << std::endl;

    auto device = bar1.device();
    auto dtype = bar1.dtype();
    std::cout << "==> device of bar1 is:\n " << device << std::endl;
    std::cout << "==> dtype of bar1 is:\n " << dtype << std::endl;

    // torch::Tensor foo32 = foo.to(torch::kFloat32);
    auto foo_mps = foo.to(torch::kMPS);
    std::cout << "==> device of foo is:\n " << foo.device() << std::endl;
    std::cout << "==> device of foo_mps is:\n " << foo_mps.device() << std::endl;

    torch::Tensor foo1 = torch::randn({1,2,3});

    auto value1 = foo1[0][1][1];
    std::cout << "==> foo1[0][1][1] is:\n " << value1 << std::endl;
    auto value2 = foo1.index({0,1,1});
    std::cout << "==> foo1.index({0,1,1}) is:\n " << value2 << std::endl;


}