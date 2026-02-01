FROM debian:trixie-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    g++ make libopencv-dev \
    gstreamer1.0-tools gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav \
    libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY main.cpp Makefile neural_net_cnn.cpp neural_net_cnn.h ./
COPY *.bin ./
RUN make

ENTRYPOINT ["./app"]
CMD ["5000", "8554", "1280", "720"]
