/*
Rotary Position Encoding (RoPE) implementation for LLM.c.

References:
- RoFormer: Enhanced Transformer with Rotary Position Embedding
  https://arxiv.org/abs/2104.09864
- Used in models like LLaMA, Falcon, MPT, etc.

RoPE encodes position information by applying rotations to query and key vectors
in the attention mechanism, creating relative position awareness directly in the attention.
*/

#ifndef LLMC_ROPE_CUH
#define LLMC_ROPE_CUH

#include "cuda_common.h"
#include "cuda_utils.cuh"

// Calculate cos and sin for RoPE at given position and dimensions
__global__ void precompute_rope_freqs_kernel(float* freqs_cos, float* freqs_sin, int T, int dim, float base) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= T * dim / 2) return;

    int t = idx / (dim / 2);
    int d = idx % (dim / 2);
    
    float theta = powf(base, -(2.0f * d) / (float)dim);
    float pos_theta = theta * t;
    
    freqs_cos[t * dim/2 + d] = cosf(pos_theta);
    freqs_sin[t * dim/2 + d] = sinf(pos_theta);
}

// Apply rotary position embeddings to query and key tensors
// Works on tensors shaped (B, NH, T, HS)
__global__ void apply_rope_kernel(
    floatX* qk_out, const floatX* qk_in, const float* freqs_cos, const float* freqs_sin,
    const int B, const int NH, const int T, const int HS, const int dim_rope
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int total = B * NH * T * HS;
    
    for (int i = idx; i < total; i += stride) {
        int b = i / (NH * T * HS);
        int h = (i / (T * HS)) % NH;
        int t = (i / HS) % T;
        int d = i % HS;
        
        int qk_idx = b * NH * T * HS + h * T * HS + t * HS + d;
        
        // Only apply rotation to dimensions up to dim_rope
        if (d >= dim_rope) {
            qk_out[qk_idx] = qk_in[qk_idx];
            continue;
        }
        
        // Handle pairs of dimensions for rotation
        if (d < dim_rope / 2) {
            int d_pair = d + dim_rope / 2;
            
            // Get the paired values
            float x_d = (float)qk_in[b * NH * T * HS + h * T * HS + t * HS + d];
            float x_d_pair = (float)qk_in[b * NH * T * HS + h * T * HS + t * HS + d_pair];
            
            // Get precomputed cos/sin values
            float cos_theta = freqs_cos[t * dim_rope/2 + d];
            float sin_theta = freqs_sin[t * dim_rope/2 + d];
            
            // Apply rotation
            float rotated_d = x_d * cos_theta - x_d_pair * sin_theta;
            float rotated_d_pair = x_d * sin_theta + x_d_pair * cos_theta;
            
            // Update the values
            qk_out[b * NH * T * HS + h * T * HS + t * HS + d] = (floatX)rotated_d;
            qk_out[b * NH * T * HS + h * T * HS + t * HS + d_pair] = (floatX)rotated_d_pair;
        } else if (d >= dim_rope / 2 && d < dim_rope) {
            // The other half of the pairs are handled when d < dim_rope/2
            continue;
        }
    }
}

// Precompute frequency tables for more efficient RoPE application
void precompute_rope_freqs(float* freqs_cos, float* freqs_sin, int T, int dim, float base, cudaStream_t stream) {
    int threads = 256;
    int blocks = (T * dim / 2 + threads - 1) / threads;
    precompute_rope_freqs_kernel<<<blocks, threads, 0, stream>>>(freqs_cos, freqs_sin, T, dim, base);
    cudaCheck(cudaGetLastError());
}

// Apply RoPE to query and key tensors
void apply_rope(
    floatX* q_out, floatX* k_out, 
    const floatX* q_in, const floatX* k_in,
    const float* freqs_cos, const float* freqs_sin, 
    int B, int NH, int T, int HS, int dim_rope, cudaStream_t stream
) {
    int threads = 256;
    int blocks = min(65535, (B * NH * T * HS + threads - 1) / threads);
    
    // Apply RoPE to query vectors
    apply_rope_kernel<<<blocks, threads, 0, stream>>>(
        q_out, q_in, freqs_cos, freqs_sin, B, NH, T, HS, dim_rope
    );
    cudaCheck(cudaGetLastError());
    
    // Apply RoPE to key vectors
    apply_rope_kernel<<<blocks, threads, 0, stream>>>(
        k_out, k_in, freqs_cos, freqs_sin, B, NH, T, HS, dim_rope
    );
    cudaCheck(cudaGetLastError());
}

#endif // LLMC_ROPE_CUH
