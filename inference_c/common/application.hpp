#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <csignal>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <stdexcept>
#include <string>

#include "preprocess.hpp"

namespace digits {
inline volatile std::sig_atomic_t running = 1;
inline void stop(int) { running = 0; }
struct Options {
  std::string image, video, output, dump_input;
  std::filesystem::path weights = "inference_c/CNN";
  int camera = -1, input_port = 5000, output_port = 8554, width = 1280, height = 720;
  bool light_foreground = false, help = false;
};

inline int positive_integer(const std::string& value, int maximum) {
  std::size_t consumed = 0;
  const int result = std::stoi(value, &consumed);
  if (consumed != value.size() || result < 1 || result > maximum)
    throw std::invalid_argument("Invalid positive integer: " + value);
  return result;
}

inline Options parse_options(int argc, char** argv,
                             const char* default_weights = "inference_c/CNN") {
  Options options;
  options.weights = default_weights;
  // Preserve the original positional TCP invocation: app 5000 8554 1280 720.
  if (argc > 1 && argv[1][0] != '-') {
    if (argc > 5) throw std::invalid_argument("Expected at most four positional arguments");
    if (argc > 1) options.input_port = positive_integer(argv[1], 65535);
    if (argc > 2) options.output_port = positive_integer(argv[2], 65535);
    if (argc > 3) options.width = positive_integer(argv[3], 16384);
    if (argc > 4) options.height = positive_integer(argv[4], 16384);
    return options;
  }
  for (int i = 1; i < argc; ++i) {
    const std::string name = argv[i];
    if (name == "--help" || name == "-h") {
      options.help = true;
      continue;
    }
    if (name == "--light-foreground") {
      options.light_foreground = true;
      continue;
    }
    if (i + 1 >= argc) throw std::invalid_argument("Missing value for " + name);
    const std::string value = argv[++i];
    if (name == "--image")
      options.image = value;
    else if (name == "--video")
      options.video = value;
    else if (name == "--output")
      options.output = value;
    else if (name == "--dump-input")
      options.dump_input = value;
    else if (name == "--weights")
      options.weights = value;
    else if (name == "--camera") {
      std::size_t consumed = 0;
      options.camera = std::stoi(value, &consumed);
      if (consumed != value.size() || options.camera < 0 || options.camera > 1024)
        throw std::invalid_argument("Invalid camera index: " + value);
    } else if (name == "--input-port")
      options.input_port = positive_integer(value, 65535);
    else if (name == "--output-port")
      options.output_port = positive_integer(value, 65535);
    else if (name == "--width")
      options.width = positive_integer(value, 16384);
    else if (name == "--height")
      options.height = positive_integer(value, 16384);
    else
      throw std::invalid_argument("Unknown option: " + name);
  }
  const int sources = !options.image.empty() + !options.video.empty() + (options.camera >= 0);
  if (sources > 1) throw std::invalid_argument("Choose one of --image, --video or --camera");
  if (!options.dump_input.empty() && options.image.empty())
    throw std::invalid_argument("--dump-input requires --image");
  return options;
}

inline void print_help(const char* program) {
  std::cout << "Usage: " << program
            << " [options]\n"
               "  --image PATH         Classify one image; print JSON logits\n"
               "  --video PATH         Process a video file\n"
               "  --camera NUMBER      Open a local camera (zero-based index)\n"
               "  --output PATH        Save image overlay or annotated MJPG AVI\n"
               "  --weights DIRECTORY  Float32 parameters (default: inference_c/MODEL)\n"
               "  --light-foreground   White digit on a dark background (MNIST/custom set)\n"
               "  --dump-input PATH    Write preprocessed float32 pixels for parity checks\n"
               "  --input-port PORT    H.264 TCP input on localhost (default: 5000)\n"
               "  --output-port PORT   Annotated TCP output on localhost (default: 8554)\n"
               "  --width N --height N TCP stream dimensions (default: 1280 x 720)\n"
               "No image/video/camera option selects the original GStreamer TCP mode.\n";
}

using Predictor = std::function<std::array<float, 10>(const std::array<float, 784>&)>;

inline void overlay(cv::Mat& frame, const PreparedDigit& digit, int prediction) {
  if (frame.channels() == 1) cv::cvtColor(frame, frame, cv::COLOR_GRAY2BGR);
  cv::rectangle(frame, digit.box, cv::Scalar(0, 180, 0), 2);
  cv::putText(frame, "Digit: " + std::to_string(prediction),
              cv::Point(digit.box.x, std::max(20, digit.box.y - 5)), cv::FONT_HERSHEY_SIMPLEX, 0.7,
              cv::Scalar(0, 180, 0), 2);
}

inline int run_application(const Options& options, const Predictor& predict) {
  if (!options.image.empty()) {
    cv::Mat image = cv::imread(options.image, cv::IMREAD_COLOR);
    if (image.empty()) throw std::runtime_error("Cannot read image: " + options.image);
    const auto digit = preprocess(image, options.light_foreground);
    if (!digit) {
      std::cout << "{\"detected\":false}\n";
      return 0;
    }
    if (!options.dump_input.empty()) {
      std::ofstream stream(options.dump_input, std::ios::binary);
      stream.write(reinterpret_cast<const char*>(digit->pixels.data()), 784 * sizeof(float));
      if (!stream) throw std::runtime_error("Could not write preprocessed input");
    }
    const auto start = std::chrono::steady_clock::now();
    const auto logits = predict(digit->pixels);
    const double milliseconds =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    const int prediction = std::max_element(logits.begin(), logits.end()) - logits.begin();
    if (!options.output.empty()) {
      overlay(image, *digit, prediction);
      if (!cv::imwrite(options.output, image)) throw std::runtime_error("Cannot save overlay");
    }
    std::cout << std::setprecision(9) << "{\"detected\":true,\"prediction\":" << prediction
              << ",\"inference_ms\":" << milliseconds << ",\"logits\":[";
    for (std::size_t i = 0; i < logits.size(); ++i) {
      if (i) std::cout << ',';
      std::cout << logits[i];
    }
    std::cout << "]}\n";
    return 0;
  }
  std::signal(SIGINT, stop);
  std::signal(SIGTERM, stop);
  const bool tcp = options.video.empty() && options.camera < 0;
  cv::VideoCapture capture;
  if (!options.video.empty())
    capture.open(options.video);
  else if (options.camera >= 0)
    capture.open(options.camera);
  else
    capture.open("tcpclientsrc host=127.0.0.1 port=" + std::to_string(options.input_port) +
                     " ! h264parse ! avdec_h264 ! videoconvert ! video/x-raw,format=BGR ! appsink "
                     "drop=1 sync=0",
                 cv::CAP_GSTREAMER);
  if (!capture.isOpened()) throw std::runtime_error("Cannot open input; check source and backend");
  cv::VideoWriter writer;
  cv::Mat frame;
  std::size_t frames = 0, detected = 0;
  double total_inference_ms = 0;
  const auto started = std::chrono::steady_clock::now();
  while (running && capture.read(frame)) {
    if (frame.empty()) continue;
    if (tcp && (frame.cols != options.width || frame.rows != options.height))
      throw std::runtime_error("TCP frame dimensions differ from --width/--height");
    if (frame.channels() == 1) cv::cvtColor(frame, frame, cv::COLOR_GRAY2BGR);
    if (!writer.isOpened() && (tcp || !options.output.empty())) {
      if (tcp) {
        const std::string pipeline =
            "appsrc ! videoconvert ! video/x-raw,format=I420 ! "
            "x264enc tune=zerolatency speed-preset=ultrafast bitrate=4000 key-int-max=15 ! "
            "video/x-h264,profile=baseline ! h264parse config-interval=1 ! mpegtsmux ! "
            "tcpserversink host=127.0.0.1 port=" +
            std::to_string(options.output_port);
        writer.open(pipeline, cv::CAP_GSTREAMER, 0, 30, frame.size(), true);
      } else {
        double fps = capture.get(cv::CAP_PROP_FPS);
        if (!std::isfinite(fps) || fps <= 0) fps = 30;
        writer.open(options.output, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, frame.size());
      }
      if (!writer.isOpened()) throw std::runtime_error("Cannot open output; check codec/backend");
    }
    const auto digit = preprocess(frame, options.light_foreground);
    if (digit) {
      const auto start = std::chrono::steady_clock::now();
      const auto logits = predict(digit->pixels);
      total_inference_ms +=
          std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
              .count();
      ++detected;
      overlay(frame, *digit, std::max_element(logits.begin(), logits.end()) - logits.begin());
    }
    if (writer.isOpened()) writer.write(frame);
    ++frames;
  }
  const double seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
  std::cout << "Frames: " << frames << ", detected: " << detected
            << ", throughput_fps: " << (seconds > 0 ? frames / seconds : 0)
            << ", mean_inference_ms: " << (detected ? total_inference_ms / detected : 0) << '\n';
  return 0;
}
}  // namespace digits
