#include <algorithm>  // std::max
#include <cmath>      // std::exp
#include <cstddef>    // std::size_t
#include "neural_network.h"

// Dense forward: y = W x + b
void dense_forward(const DenseLayer* layer, const float* input, float* output)
{
    const int in  = layer->in_size;
    const int out = layer->out_size;

    for (int i = 0; i < out; ++i) {
        float acc = layer->bias[i];
        const float* wrow = &layer->weights[i * in];
        for (int j = 0; j < in; ++j) {
            acc += wrow[j] * input[j];
        }
        output[i] = acc;
    }
}

void relu(float* x, int size)
{
    for (int i = 0; i < size; ++i) {
        if (x[i] < 0.0f) x[i] = 0.0f;
    }
}

void softmax(float* x, int size)
{
    // stabilité numérique: soustraire le max
    float m = x[0];
    for (int i = 1; i < size; ++i) {
        if (x[i] > m) m = x[i];
    }

    float sum = 0.0f;
    for (int i = 0; i < size; ++i) {
        x[i] = static_cast<float>(std::exp(x[i] - m));
        sum += x[i];
    }

    // éviter division par zéro
    if (sum <= 0.0f) return;

    const float inv = 1.0f / sum;
    for (int i = 0; i < size; ++i) {
        x[i] *= inv;
    }
}

int argmax(const float* x, int size)
{
    int best = 0;
    float bestv = x[0];
    for (int i = 1; i < size; ++i) {
        if (x[i] > bestv) { bestv = x[i]; best = i; }
    }
    return best;
}
