#include "neural_net_cnn.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>

#include "../common/weights.hpp"

namespace digits {
namespace {
Input convolve_relu(const Input& input, const std::vector<float>& weights, float bias) {
  Input output{};
  for (int y = 0; y < kImageWidth; ++y) {
    for (int x = 0; x < kImageWidth; ++x) {
      float value = bias;
      for (int ky = 0; ky < 5; ++ky) {
        for (int kx = 0; kx < 5; ++kx) {
          const int row = y + ky - 2, col = x + kx - 2;
          if (row >= 0 && row < kImageWidth && col >= 0 && col < kImageWidth) {
            value += input[row * kImageWidth + col] * weights[ky * 5 + kx];
          }
        }
      }
      output[y * kImageWidth + x] = std::max(0.0f, value);
    }
  }
  return output;
}
}  // namespace

Cnn::Cnn(const std::filesystem::path& directory)
    : conv1_weights_(read_parameters(directory / "layer1.0.weight.bin", 25)),
      conv1_bias_(read_parameters(directory / "layer1.0.bias.bin", 1)),
      conv2_weights_(read_parameters(directory / "layer2.0.weight.bin", 25)),
      conv2_bias_(read_parameters(directory / "layer2.0.bias.bin", 1)),
      linear_weights_(read_parameters(directory / "layer3.weight.bin", kImageSize * kClasses)),
      linear_bias_(read_parameters(directory / "layer3.bias.bin", kClasses)) {}

Logits Cnn::predict(const Input& input) const {
  const auto hidden1 = convolve_relu(input, conv1_weights_, conv1_bias_[0]);
  const auto hidden2 = convolve_relu(hidden1, conv2_weights_, conv2_bias_[0]);
  Logits output{};
  for (int digit = 0; digit < kClasses; ++digit) {
    float value = linear_bias_[digit];
    for (int i = 0; i < kImageSize; ++i) {
      value += hidden2[i] * linear_weights_[digit * kImageSize + i];
    }
    output[digit] = value;
  }
  return output;
}
}  // namespace digits
