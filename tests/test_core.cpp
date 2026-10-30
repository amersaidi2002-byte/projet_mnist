#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

#include "../inference_c/CNN/neural_net_cnn.h"
#include "../inference_c/MLP/neural_network.h"

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
int main(int argc, char** argv) {
  try {
    require(argc == 2, "Pass the CNN parameter directory");
    const float weights[]{1, 2, 3, 4}, bias[]{0.5f, -0.5f}, input[]{2, 3};
    float output[2]{};
    const DenseLayer layer{2, 2, weights, bias};
    dense_forward(&layer, input, output);
    require(output[0] == 8.5f && output[1] == 17.5f, "Dense layout is wrong");
    float activations[]{-5, 0, 3};
    relu(activations, 3);
    require(activations[0] == 0 && activations[2] == 3, "ReLU is wrong");
    float extreme[]{10000, 10001, 9999};
    softmax(extreme, 3);
    require(std::isfinite(extreme[0]) && std::abs(extreme[0] + extreme[1] + extreme[2] - 1) < 1e-6f,
            "Softmax is not stable");
    require(argmax(extreme, 3) == 1, "Argmax is wrong");
    const digits::Cnn cnn(argv[1]);
    digits::Input image{};
    for (int i = 0; i < 784; ++i) image[i] = static_cast<float>(i % 31) / 31;
    const auto first = cnn.predict(image), second = cnn.predict(image);
    require(first == second, "CNN must not retain mutable inference state");
    for (float value : first) require(std::isfinite(value), "CNN produced a non-finite logit");
    bool rejected = false;
    try {
      const digits::Cnn invalid(std::filesystem::path(argv[1]) / "missing");
    } catch (const std::runtime_error&) {
      rejected = true;
    }
    require(rejected, "Missing parameters must be rejected");
    std::cout << "Core checks passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
