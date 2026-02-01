#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#define INPUT_SIZE   784
#define FC1_SIZE     128
#define FC2_SIZE     64
#define OUTPUT_SIZE  10

struct DenseLayer {
    int in_size;
    int out_size;
    const float* weights;  // out_size * in_size
    const float* bias;     // out_size
};

// input not modified
void dense_forward(const DenseLayer* layer,
                   const float* input,
                   float* output);

void relu(float* x, int size);
void softmax(float* x, int size);

int argmax(const float* x, int size);

#endif
