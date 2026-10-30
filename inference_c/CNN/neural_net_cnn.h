#pragma once

#include <array>
#include <filesystem>
#include <vector>

namespace digits {
inline constexpr int kImageWidth = 28;
inline constexpr int kImageSize = 784;
inline constexpr int kClasses = 10;
using Input = std::array<float, kImageSize>;
using Logits = std::array<float, kClasses>;

// Parameters belong to the instance; prediction uses local scratch buffers.
class Cnn {
 public:
  explicit Cnn(const std::filesystem::path& weight_directory);
  Logits predict(const Input& input) const;

 private:
  std::vector<float> conv1_weights_, conv1_bias_, conv2_weights_, conv2_bias_;
  std::vector<float> linear_weights_, linear_bias_;
};
}  // namespace digits
