FROM debian:bookworm-slim
ARG MODEL=cnn
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates g++ make cmake pkg-config libopencv-dev \
    gstreamer1.0-tools gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY CMakeLists.txt ./
COPY inference_c/ inference_c/
RUN test "$MODEL" = cnn -o "$MODEL" = mlp \
    && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF \
    && cmake --build build --target digit_${MODEL}_app --parallel 2 \
    && ln -s /app/build/digit_${MODEL}_app /usr/local/bin/recognize-digit
ENTRYPOINT ["recognize-digit"]
CMD ["--help"]
