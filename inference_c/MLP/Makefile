CXX ?= g++
CPPFLAGS += $(shell pkg-config --cflags opencv4)
CXXFLAGS ?= -O3 -Wall -Wextra -Wpedantic -std=c++17
LDLIBS += $(shell pkg-config --libs opencv4)

app: main.cpp neural_network.cpp neural_network.h ../common/weights.hpp ../common/application.hpp ../common/preprocess.hpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) main.cpp neural_network.cpp -o $@ $(LDLIBS)

.PHONY: clean
clean:
	rm -f app *.o
