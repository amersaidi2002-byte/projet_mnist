#include "neural_net_cnn.h"
#include <iostream>
#include <cstdlib>

// ------------------- RELU -------------------
void ReLU(double* x_in, int length, double* x_out) {
    for (int i = 0; i < length; i++)
        x_out[i] = (x_in[i] > 0.0) ? x_in[i] : 0.0;
}

// ------------------- CONV2D -------------------
void Conv2d(double* x_in, double* x_out,
            int kernel_size, int width, int height,
            double* W, double* b) {

    int h = kernel_size / 2; // padding = 2 pour kernel 5

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {

            double sum = b[0];

            for (int j = -h; j <= h; j++) {
                for (int i = -h; i <= h; i++) {

                    int yy = y + j;
                    int xx = x + i;

                    // padding zéro: si hors image -> contribution 0
                    if (yy >= 0 && yy < height && xx >= 0 && xx < width) {
                        int w_idx = (j + h) * kernel_size + (i + h);
                        sum += x_in[yy * width + xx] * W[w_idx];
                    }
                }
            }

            x_out[y * width + x] = sum;
        }
    }
}

// ------------------- LINEAR -------------------
void Linear(const LinearLayer& layer, double* x_in, double* x_out) {
    for (int o = 0; o < layer.out_features; o++) {
        double sum = layer.b[o];
        for (int i = 0; i < layer.in_features; i++)
            sum += layer.W[o * layer.in_features + i] * x_in[i];
        x_out[o] = sum;
    }
}

// ------------------- LECTURE POIDS -------------------
float* read_weights(const char* filename, int n_element) {
    FILE* f = fopen(filename, "rb");
    if (!f) {
        std::cerr << "Erreur ouverture " << filename << std::endl;
        exit(1);
    }
    float* data = (float*)malloc(sizeof(float) * n_element);
    fread(data, sizeof(float), n_element, f);
    fclose(f);
    return data;
}

// ------------------- BUFFERS GLOBAUX -------------------
double conv1_W[KERNEL_SIZE * KERNEL_SIZE];
double conv1_b[1];

double conv2_W[KERNEL_SIZE * KERNEL_SIZE];
double conv2_b[1];

double x1[IMG_SIZE], x1_relu[IMG_SIZE];
double x2[IMG_SIZE], x2_relu[IMG_SIZE];

LinearLayer fc;

// ------------------- CHARGEMENT POIDS -------------------
void load_weights() {
    float* w;

    w = read_weights("layer1.0.weight.bin", 25);
    for (int i = 0; i < 25; i++) conv1_W[i] = w[i];
    conv1_b[0] = read_weights("layer1.0.bias.bin", 1)[0];

    w = read_weights("layer2.0.weight.bin", 25);
    for (int i = 0; i < 25; i++) conv2_W[i] = w[i];
    conv2_b[0] = read_weights("layer2.0.bias.bin", 1)[0];

    fc.in_features = IMG_SIZE;
    fc.out_features = NUM_CLASSES;
    fc.W = read_weights("layer3.weight.bin", IMG_SIZE * NUM_CLASSES);
    fc.b = read_weights("layer3.bias.bin", NUM_CLASSES);
}

// ------------------- FORWARD COMPLET -------------------
void CNN_forward(double* input, double* out) {
    Conv2d(input, x1, KERNEL_SIZE, IMG_W, IMG_H, conv1_W, conv1_b);
    ReLU(x1, IMG_SIZE, x1_relu);

    Conv2d(x1_relu, x2, KERNEL_SIZE, IMG_W, IMG_H, conv2_W, conv2_b);
    ReLU(x2, IMG_SIZE, x2_relu);

    Linear(fc, x2_relu, out);
}
