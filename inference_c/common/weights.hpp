#pragma once
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace digits {
inline std::vector<float> read_parameters(const std::filesystem::path& path, std::size_t count) {
  static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559,
                "Weights require IEEE-754 float32");
  const std::uint32_t probe = 1;
  if (*reinterpret_cast<const unsigned char*>(&probe) != 1)
    throw std::runtime_error("Binary weights require a little-endian host");
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream || stream.tellg() != static_cast<std::streamoff>(count * sizeof(float)))
    throw std::runtime_error("Missing or incorrectly sized weight file: " + path.string());
  stream.seekg(0);
  std::vector<float> values(count);
  if (!stream.read(reinterpret_cast<char*>(values.data()), count * sizeof(float)))
    throw std::runtime_error("Could not read weights: " + path.string());
  for (float value : values)
    if (!std::isfinite(value)) throw std::runtime_error("Non-finite weight in " + path.string());
  return values;
}
}  // namespace digits
