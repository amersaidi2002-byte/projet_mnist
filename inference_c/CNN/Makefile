CXX = g++
CXXFLAGS = -O3 -Wall -std=c++17 $(shell pkg-config --cflags opencv4)
LDFLAGS  = $(shell pkg-config --libs opencv4)

SRCS = main.cpp neural_net_cnn.cpp
OBJS = $(SRCS:.cpp=.o)

app: $(OBJS)
	$(CXX) $(CXXFLAGS) $(OBJS) -o $@ $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) app
