/*
LLaMA3 model implementation for llm.c

This header defines structures and functions specific to the LLaMA3 architecture.
Key differences from GPT-2:
1. RoPE for positional encoding
2. Grouped Query Attention (GQA)
3. SwiGLU activation in the feed-forward network
*/

#ifndef LLMC_LLAMA3_CUH
#define LLMC_LLAMA3_CUH

#include "cuda_common.h"
#include "cuda_utils.cuh"
#include "rope.cuh"

// LLaMA3 model configuration
typedef struct {
    int max_seq_len;      // Maximum sequence length, e.g., 2048
    int vocab_size;       // Vocabulary size
    int padded_vocab_size; // Padded to e.g. %128==0
    int num_layers;       // Number of transformer layers
    int num_heads;        // Number of attention heads
    int num_kv_heads;     // Number of key/value heads (for GQA)
    int channels;         // Embedding dimensions
    int ffn_dim_mult;     // Multiplier for feed-forward hidden dimension
    float norm_eps;       // Epsilon for RMSNorm
    
    // RoPE parameters
    float rope_base;      // Base for rotary embeddings (default: 10000.0)
    int rope_dim;         // Dimension for rotary embeddings
} Llama3Config;

// SwiGLU activation function for feed-forward networks
__device__ inline float swiglu(float x, float gate) {
    return x * (1.0f / (1.0f + __expf(-gate)));
}

// Grouped Query Attention implementation
__global__ void group_query_attention_kernel(
    floatX* output,
    const floatX* q,     // [B, NH, T, HS]
    const floatX* k,     // [B, NKV, T, HS]
    const floatX* v,     // [B, NKV, T, HS]
    const int B, const int T, const int NH, const int NKV, const int HS
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Each head repeats KV heads by NH/NKV
    int rep = NH / NKV;
    
    for (int i = idx; i < B * NH * T; i += stride) {
        int b = i / (NH * T);
        int h = (i / T) % NH;
        int t = i % T;
        
        // Map query head to corresponding kv head
        int kv_head = h / rep;
        
        // Compute attention for this position
        for (int j = 0; j < HS; j++) {
            float sum = 0.0f;
            float normalizer = 0.0f;
            
            // Attend to all previous positions (causal attention)
            for (int s = 0; s <= t; s++) {
                // Compute dot product between q[b,h,t] and k[b,kv_head,s]
                float dot = 0.0f;
                for (int d = 0; d < HS; d++) {
                    dot += (float)q[b * NH * T * HS + h * T * HS + t * HS + d] * 
                           (float)k[b * NKV * T * HS + kv_head * T * HS + s * HS + d];
                }
                
                // Scale and compute softmax
                float score = __expf(dot / sqrtf(HS));
                normalizer += score;
                
                // Weighted sum of values
                sum += score * (float)v[b * NKV * T * HS + kv_head * T * HS + s * HS + j];
            }
            
            // Normalize and store result
            if (normalizer > 0) {
                output[b * NH * T * HS + h * T * HS + t * HS + j] = (floatX)(sum / normalizer);
            } else {
                output[b * NH * T * HS + h * T * HS + t * HS + j] = 0.0f;
            }
        }
    }
}

// Wrapper function for grouped query attention
void grouped_query_attention(
    floatX* output, 
    const floatX* q, const floatX* k, const floatX* v,
    int B, int T, int NH, int NKV, int HS,
    cudaStream_t stream
) {
    int block_size = 256;
    int num_blocks = CEIL_DIV(B * NH * T, block_size);
    
    // Use flash attention if available, otherwise fall back to custom kernel
    // (This is a simplified placeholder - in production, you would use cuDNN or a flash attention implementation)
    group_query_attention_kernel<<<num_blocks, block_size, 0, stream>>>(output, q, k, v, B, T, NH, NKV, HS);
    cudaCheck(cudaGetLastError());
}

// SwiGLU feed-forward network kernel
__global__ void swiglu_ffn_kernel(
    floatX* output,
    const floatX* input,
    const floatX* w1, const floatX* w2, const floatX* w3,
    int B, int T, int C, int hidden_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int N = B * T * hidden_dim;
    
    for (int i = idx; i < N; i += stride) {
        int bt = i / hidden_dim;
        int d = i % hidden_dim;
        int b = bt / T;
        int t = bt % T;
        
        // Compute gate activation (w1) and linear projection (w3)
        float gate_val = 0.0f;
        float proj_val = 0.0f;
        
        for (int j = 0; j < C; j++) {
            float x_j = (float)input[b * T * C + t * C + j];
            gate_val += x_j * (float)w1[j * hidden_dim + d];
            proj_val += x_j * (float)w3[j * hidden_dim + d];
        }
        
        // Apply SwiGLU: proj_val * sigmoid(gate_val)
        float activated = swiglu(proj_val, gate_val);
        
        // Store intermediate result
        output[i] = (floatX)activated;
    }
}

// Second part of SwiGLU feed-forward network kernel
__global__ void swiglu_ffn_output_kernel(
    floatX* output,
    const floatX* activated,
    const floatX* w2,
    int B, int T, int C, int hidden_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int N = B * T * C;
    
    for (int i = idx; i < N; i += stride) {
        int bt = i / C;
        int c = i % C;
        int b = bt / T;
        int t = bt % T;
        
        float result = 0.0f;
        
        // Matrix multiply with w2
        for (int j = 0; j < hidden_dim; j++) {
            result += (float)activated[b * T * hidden_dim + t * hidden_dim + j] * 
                     (float)w2[j * C + c];
        }
        
        output[i] = (floatX)result;
    }
}

// Wrapper for SwiGLU feed-forward network
void swiglu_ffn(
    floatX* output,
    const floatX* input,
    const floatX* w1, const floatX* w2, const floatX* w3,
    int B, int T, int C, int hidden_dim,
    cudaStream_t stream
) {
    int block_size = 256;
    
    // Allocate temporary buffer for activated values
    floatX* activated;
    cudaCheck(cudaMalloc(&activated, B * T * hidden_dim * sizeof(floatX)));
    
    // First part: compute SwiGLU activation
    int num_blocks1 = CEIL_DIV(B * T * hidden_dim, block_size);
    swiglu_ffn_kernel<<<num_blocks1, block_size, 0, stream>>>(
        activated, input, w1, w2, w3, B, T, C, hidden_dim
    );
    cudaCheck(cudaGetLastError());
    
    // Second part: compute output projection
    int num_blocks2 = CEIL_DIV(B * T * C, block_size);
    swiglu_ffn_output_kernel<<<num_blocks2, block_size, 0, stream>>>(
        output, activated, w2, B, T, C, hidden_dim
    );
    cudaCheck(cudaGetLastError());
    
    // Free temporary buffer
    cudaCheck(cudaFree(activated));
}

#endif // LLMC_LLAMA3_CUH
