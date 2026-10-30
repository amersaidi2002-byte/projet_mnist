#include "neural_network.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "../common/weights.hpp"

void dense_forward(const DenseLayer* layer, const float* input, float* output) {
  if (!layer || !input || !output || !layer->weights || !layer->bias || layer->in_size < 1 ||
      layer->out_size < 1)
    throw std::invalid_argument("Invalid dense layer buffers or dimensions");
  for (int row = 0; row < layer->out_size; ++row) {
    float sum = layer->bias[row];
    for (int col = 0; col < layer->in_size; ++col)
      sum += input[col] * layer->weights[row * layer->in_size + col];
    output[row] = sum;
  }
}
void relu(float* values, int size) {
  if (!values || size < 1) throw std::invalid_argument("Invalid ReLU input");
  for (int i = 0; i < size; ++i) values[i] = std::max(0.0f, values[i]);
}
void softmax(float* values, int size) {
  if (!values || size < 1) throw std::invalid_argument("Invalid softmax input");
  for (int i = 0; i < size; ++i)
    if (!std::isfinite(values[i])) throw std::invalid_argument("Non-finite softmax input");
  const float maximum = *std::max_element(values, values + size);
  float sum = 0;
  for (int i = 0; i < size; ++i) {
    values[i] = std::exp(values[i] - maximum);
    sum += values[i];
  }
  for (int i = 0; i < size; ++i) values[i] /= sum;
}
int argmax(const float* values, int size) {
  if (!values || size < 1) throw std::invalid_argument("Invalid argmax input");
  return std::max_element(values, values + size) - values;
}
namespace digits {
Mlp::Mlp(const std::filesystem::path& directory)
    : w1_(read_parameters(directory / "fc1.weight.bin", INPUT_SIZE * FC1_SIZE)),
      b1_(read_parameters(directory / "fc1.bias.bin", FC1_SIZE)),
      w2_(read_parameters(directory / "fc2.weight.bin", FC1_SIZE * FC2_SIZE)),
      b2_(read_parameters(directory / "fc2.bias.bin", FC2_SIZE)),
      w3_(read_parameters(directory / "fc3.weight.bin", FC2_SIZE * OUTPUT_SIZE)),
      b3_(read_parameters(directory / "fc3.bias.bin", OUTPUT_SIZE)) {}
std::array<float, 10> Mlp::predict(const std::array<float, 784>& input) const {
  const DenseLayer fc1{INPUT_SIZE, FC1_SIZE, w1_.data(), b1_.data()};
  const DenseLayer fc2{FC1_SIZE, FC2_SIZE, w2_.data(), b2_.data()};
  const DenseLayer fc3{FC2_SIZE, OUTPUT_SIZE, w3_.data(), b3_.data()};
  std::array<float, FC1_SIZE> hidden1{};
  std::array<float, FC2_SIZE> hidden2{};
  std::array<float, OUTPUT_SIZE> output{};
  dense_forward(&fc1, input.data(), hidden1.data());
  relu(hidden1.data(), FC1_SIZE);
  dense_forward(&fc2, hidden1.data(), hidden2.data());
  relu(hidden2.data(), FC2_SIZE);
  dense_forward(&fc3, hidden2.data(), output.data());
  return output;
}
}  // namespace digits
