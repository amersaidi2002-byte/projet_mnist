#pragma once

#include <algorithm>
#include <array>
#include <opencv2/imgproc.hpp>
#include <optional>
#include <stdexcept>
#include <vector>

namespace digits {
struct PreparedDigit {
  std::array<float, 784> pixels{};
  cv::Rect box;
};

// Matches training/preprocess.py. No eligible contour is a normal result.
inline std::optional<PreparedDigit> preprocess(const cv::Mat& image, bool light_foreground) {
  if (image.empty() || image.depth() != CV_8U || (image.channels() != 1 && image.channels() != 3)) {
    throw std::invalid_argument("Expected a nonempty uint8 grayscale or BGR image");
  }
  cv::Mat gray;
  if (image.channels() == 3) {
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
  } else {
    gray = image.clone();
  }
  cv::GaussianBlur(gray, gray, cv::Size(3, 3), 0);
  cv::Mat binary;
  cv::threshold(gray, binary, 125, 255,
                light_foreground ? cv::THRESH_BINARY : cv::THRESH_BINARY_INV);
  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
  const std::vector<cv::Point>* best = nullptr;
  double largest = 0;
  for (const auto& contour : contours) {
    const double area = cv::contourArea(contour);
    if (area >= 4 && area <= 0.9 * binary.total() && area > largest) {
      largest = area;
      best = &contour;
    }
  }
  if (!best) return std::nullopt;
  PreparedDigit result;
  result.box = cv::boundingRect(*best);
  const int side = std::max(result.box.width, result.box.height);
  cv::Mat square = cv::Mat::zeros(side, side, CV_8UC1);
  const cv::Rect center((side - result.box.width) / 2, (side - result.box.height) / 2,
                        result.box.width, result.box.height);
  binary(result.box).copyTo(square(center));
  cv::Mat resized, normalized;
  cv::resize(square, resized, cv::Size(28, 28), 0, 0, cv::INTER_AREA);
  resized.convertTo(normalized, CV_32F, 1.0 / 255.0);
  for (int row = 0; row < 28; ++row) {
    const float* values = normalized.ptr<float>(row);
    std::copy(values, values + 28, result.pixels.begin() + row * 28);
  }
  return result;
}
}  // namespace digits
