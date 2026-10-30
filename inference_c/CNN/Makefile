CXX ?= g++
CPPFLAGS += $(shell pkg-config --cflags opencv4)
CXXFLAGS ?= -O3 -Wall -Wextra -Wpedantic -std=c++17
LDLIBS += $(shell pkg-config --libs opencv4)

app: main.cpp neural_net_cnn.cpp neural_net_cnn.h ../common/application.hpp ../common/preprocess.hpp ../common/weights.hpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) main.cpp neural_net_cnn.cpp -o $@ $(LDLIBS)

.PHONY: clean
clean:
	rm -f app *.o
