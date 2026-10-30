#include <cmath>
#include <iomanip>
#include <iostream>

#include "../inference_c/MLP/neural_network.h"

int main(int argc, char** argv) {
  try {
    if (argc != 2) throw std::runtime_error("Expected a weight directory");
    const digits::Mlp model(argv[1]);
    std::array<float, 784> input{};
    for (float& value : input)
      if (!(std::cin >> value) || !std::isfinite(value))
        throw std::runtime_error("Expected 784 floats");
    for (float value : model.predict(input)) std::cout << std::setprecision(9) << value << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
