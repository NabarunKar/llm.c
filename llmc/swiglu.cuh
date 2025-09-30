/*
SwiGLU activation CUDA kernel and launcher.
SwiGLU(x) = x1 * sigmoid(x2), where x1 and x2 are split from input tensor.
Forward: input shape (N, 8*C), output shape (N, 4*C)
Backward: grad wrt input shape (N, 8*C)
*/

#pragma once
#include <cuda_runtime.h>
#include <math.h>
#include "cuda_common.h"
#include "cuda_utils.cuh"

// Forward kernel: out = x1 * sigmoid(x2)
__global__ void swiglu_forward_kernel(floatX* out, const floatX* in, size_t N, size_t C) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = N * 4 * C;
    if (idx < total) {
        size_t n = idx / (4 * C);
        size_t c = idx % (4 * C);
        float x1 = (float)in[n * 8 * C + c];
        float x2 = (float)in[n * 8 * C + 4 * C + c];
        float sigmoid_x2 = 1.0f / (1.0f + expf(-x2));
        out[n * 4 * C + c] = (floatX)(x1 * sigmoid_x2);
    }
}

// Backward kernel: grad_in = [grad_out * sigmoid(x2), grad_out * x1 * sigmoid(x2) * (1 - sigmoid(x2))]
__global__ void swiglu_backward_kernel(floatX* grad_in, const floatX* in, const floatX* out, size_t N, size_t C) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = N * 4 * C;
    if (idx < total) {
        size_t n = idx / (4 * C);
        size_t c = idx % (4 * C);
        float x1 = (float)in[n * 8 * C + c];
        float x2 = (float)in[n * 8 * C + 4 * C + c];
        float sigmoid_x2 = 1.0f / (1.0f + expf(-x2));
        float grad_out = (float)out[n * 4 * C + c];
        // grad wrt x1
        grad_in[n * 8 * C + c] = (floatX)(grad_out * sigmoid_x2);
        // grad wrt x2
        grad_in[n * 8 * C + 4 * C + c] = (floatX)(grad_out * x1 * sigmoid_x2 * (1.0f - sigmoid_x2));
    }
}

// Launcher for forward
inline void swiglu_forward(floatX* out, const floatX* in, size_t N, size_t C, cudaStream_t stream) {
    size_t total = N * 4 * C;
    int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;
    swiglu_forward_kernel<<<grid_size, block_size, 0, stream>>>(out, in, N, C);
    cudaCheck(cudaGetLastError());
}

// Launcher for backward
inline void swiglu_backward(floatX* grad_in, const floatX* in, const floatX* out, size_t N, size_t C, cudaStream_t stream) {
    size_t total = N * 4 * C;
    int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;
    swiglu_backward_kernel<<<grid_size, block_size, 0, stream>>>(grad_in, in, out, N, C);
    cudaCheck(cudaGetLastError());
}
