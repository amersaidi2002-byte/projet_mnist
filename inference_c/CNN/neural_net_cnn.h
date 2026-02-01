#ifndef NEURAL_NET_CNN_H
#define NEURAL_NET_CNN_H

#include <cstdio>

// ------------------- CONSTANTES -------------------
#define IMG_W 28
#define IMG_H 28
#define IMG_SIZE (IMG_W * IMG_H)
#define NUM_CLASSES 10
#define KERNEL_SIZE 5

// ------------------- STRUCTURES -------------------
struct LinearLayer {
    int in_features;
    int out_features;
    float* W;
    float* b;
};

// ------------------- FONCTIONS -------------------
void ReLU(double* x_in, int length, double* x_out);

void Conv2d(double* x_in, double* x_out,
            int kernel_size, int width, int height,
            double* W, double* b);

void Linear(const LinearLayer& layer, double* x_in, double* x_out);

float* read_weights(const char* filename, int n_element);

void load_weights();

void CNN_forward(double* input, double* out);

#endif // NEURAL_NET_CNN_H
