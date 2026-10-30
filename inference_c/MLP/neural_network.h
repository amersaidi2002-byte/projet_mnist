#pragma once
#include <array>
#include <filesystem>
#include <vector>

inline constexpr int INPUT_SIZE = 784;
inline constexpr int FC1_SIZE = 128;
inline constexpr int FC2_SIZE = 64;
inline constexpr int OUTPUT_SIZE = 10;
struct DenseLayer {
  int in_size, out_size;
  const float* weights;
  const float* bias;
};
void dense_forward(const DenseLayer* layer, const float* input, float* output);
void relu(float* values, int size);
void softmax(float* values, int size);
int argmax(const float* values, int size);

namespace digits {
class Mlp {
 public:
  explicit Mlp(const std::filesystem::path& directory);
  std::array<float, 10> predict(const std::array<float, 784>& input) const;

 private:
  std::vector<float> w1_, b1_, w2_, b2_, w3_, b3_;
};
}  // namespace digits
