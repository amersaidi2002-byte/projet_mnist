#include <exception>
#include <iostream>

#include "../common/application.hpp"
#include "neural_network.h"

int main(int argc, char** argv) {
  try {
    const auto options = digits::parse_options(argc, argv, "inference_c/MLP");
    if (options.help) {
      digits::print_help(argv[0]);
      return 0;
    }
    const digits::Mlp model(options.weights);
    return digits::run_application(options,
                                   [&model](const auto& input) { return model.predict(input); });
  } catch (const std::exception& error) {
    std::cerr << "Error: " << error.what() << '\n';
    return 1;
  }
}
