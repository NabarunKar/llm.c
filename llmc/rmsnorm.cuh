/*
RMSNorm CUDA kernel

RMSNorm is a simplification of LayerNorm where we only normalize by the
root mean square without subtracting the mean (i.e., no centering).
This reduces computation and has been shown to work as well as LayerNorm in practice.

Reference: https://arxiv.org/abs/1910.07467
*/

#include <assert.h>
// llmc internal imports
#include "cuda_common.h"
#include "cuda_utils.cuh"

// ----------------------------------------------------------------------------
// CUDA kernels

__global__ void rmsnorm_forward_kernel3(floatX* __restrict__ out, float* __restrict__ rstd,
                                    const floatX*  __restrict__ inp, const floatX*  __restrict__ weight,
                                    int N, int C) {
    int lane_id = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    int num_warps = blockDim.x / WARP_SIZE;

    int idx = blockIdx.x * num_warps + warp_id;
    if(idx >= N) { return; } // guard

    // the row of input that this group of threads is responsible for
    const floatX* x = inp + idx * C;

    // calculate the mean of squares
    float sum_squares = 0.0f;
    for (int i = lane_id; i < C; i += WARP_SIZE) {
        float xi = (float)x[i];
        sum_squares += xi * xi;
    }
    sum_squares = warpReduceSum(sum_squares);
    float rms = sqrtf(sum_squares / C + 1e-5f);
    float s = 1.0f / rms;
    if(lane_id == 0 && rstd != nullptr) {
        __stcs(rstd + idx, s);
    }

    // final normalization and scaling by weight
    floatX* o = out + idx * C;
    for (int c = lane_id; c < C; c += WARP_SIZE) {
        // load and store using the .cs "streaming" hint to the compiler,
        // indicating that this data will not be reused soon, and can be streamed through the caches
        // this allows the threads to get more cache-hits for the shared weight parameters
        float n = s * (float)__ldcs(x+c);
        __stcs(o+c, (floatX)(n * (float)weight[c]));
    }
}

__global__ void rmsnorm_forward_kernel6(floatX* __restrict__ out, float* __restrict__ rstd,
                                    const floatX*  __restrict__ inp, const floatX*  __restrict__ weight,
                                    int N, int C) {
    assert(blockDim.x == WARP_SIZE);

    // load weights into shared memory
    // do this before we allow any threads to exit!
    extern __shared__ char* params[];
    // load128/store128 sometimes generated multiple instructions when the types here were floatX*, so
    // let's keep everything as x128
    x128* s_weight = reinterpret_cast<x128*>(params);
    x128* s_in = reinterpret_cast<x128*>(params) + (C / x128::size);

    int sidx = (threadIdx.x + WARP_SIZE * threadIdx.y) * x128::size;
    for(int i = sidx; i < C; i += blockDim.y * WARP_SIZE * x128::size) {
        s_weight[i/x128::size] = load128(weight + i);
    }
    __syncthreads();

    int idx = blockIdx.x * blockDim.y + threadIdx.y;
    if(idx >= N) { return; } // guard

    // adjust pointers to current token
    inp += idx * C;
    out += idx * C;

    const float eps = 1e-5f;
    float sum_squares = 0.0f;
    for(int c = threadIdx.x * x128::size; c < C; c += WARP_SIZE * x128::size) {
        const x128 in_data = load128cs(inp + c);
        for(int k = 0; k < x128::size; ++k) {
            float xi = (float)in_data[k];
            sum_squares += xi * xi;
        }
        s_in[c / x128::size] = in_data;
    }

    sum_squares = warpReduceSum(sum_squares);
    float rms = sqrtf(sum_squares / C + eps);
    float s = 1.0f / rms;

    for(int c = threadIdx.x * x128::size; c < C; c += WARP_SIZE * x128::size) {
        const x128 in_data = s_in[c / x128::size];
        const x128 w = s_weight[c / x128::size];
        x128 out_data;
        for(int k = 0; k < x128::size; ++k) {
            float n = s * (float)in_data[k]; // normalized output
            float o = n * (float)w[k]; // scale it
            out_data[k] = (floatX)o;
        }

        store128cs(out + c, out_data);
    }
    // cache the rstd for the backward pass later
    if(threadIdx.x == 0 && rstd != nullptr) {
        __stcs(rstd + idx, s);
    }
}

__global__ void fused_residual_rmsnorm_forward_kernel5(floatX* residual, floatX* normed, float* rstd,
                                               const floatX* inp1, const floatX* inp2,
                                               const floatX* weight,
                                               int N, int C) {
    assert(blockDim.x == WARP_SIZE);

    // load weights into shared memory
    // do this before we allow any threads to exit!
    extern __shared__ char* params[];
    // load128/store128 sometimes generated multiple instructions when the types here were floatX*, so
    // let's keep everything as x128
    x128* s_weight = reinterpret_cast<x128*>(params);
    x128* s_res = reinterpret_cast<x128*>(params) + (C / x128::size);

    int sidx = (threadIdx.x + WARP_SIZE * threadIdx.y) * x128::size;
    for(int i = sidx; i < C; i += blockDim.y * WARP_SIZE * x128::size) {
        s_weight[i/x128::size] = load128(weight + i);
    }
    __syncthreads();

    int idx = blockIdx.x * blockDim.y + threadIdx.y;
    if(idx > N) return;

    // adjust pointers to current token
    residual += C * idx;
    normed += C * idx;
    inp1 += C * idx;
    inp2 += C * idx;

    const float eps = 1e-5f;
    float sum_squares = 0.0f;
    for(int c = threadIdx.x * x128::size; c < C; c += WARP_SIZE * x128::size) {
        const x128 in1 = load128cs(inp1 + c);
        const x128 in2 = load128cs(inp2 + c);
        x128 out;
        for(int k = 0; k < x128::size; ++k) {
            out[k] = (float)in1[k] + (float)in2[k];
            sum_squares += (float)out[k] * (float)out[k];
        }
        store128cs(residual + c, out);
        s_res[c / x128::size] = out;
    }

    sum_squares = warpReduceSum(sum_squares);
    float rms = sqrtf(sum_squares / C + eps);
    float s = 1.0f / rms;

    for(int c = threadIdx.x * x128::size; c < C; c += WARP_SIZE * x128::size) {
        const x128 res = s_res[c / x128::size];
        const x128 w = s_weight[c / x128::size];
        x128 out;
        for(int k = 0; k < x128::size; ++k) {
            float n = s * (float)res[k]; // normalized output
            float o = n * (float)w[k]; // scale it
            out[k] = o;
        }

        store128cs(normed + c, out);
    }
    // cache the rstd for the backward pass later
    if(threadIdx.x == 0) {
        rstd[idx] = s;
    }
}

__global__ void rmsnorm_backward_kernel10(floatX* dinp, floatX* dweight, float* scratch,
                                const floatX* dout, const floatX* inp, const floatX* weight,
                                const float* rstd,
                                int B, int T, int C) {
    int BLOCK_SIZE = blockDim.x;
    int warpsInBlock = BLOCK_SIZE / WARP_SIZE; //number of warps in block
    extern __shared__ float shared[];

    int warpId = threadIdx.x / WARP_SIZE; // warp index within a block
    int baseIdx = blockIdx.x * warpsInBlock + warpId;
    int warpThreadIdx = threadIdx.x % WARP_SIZE; // Thread index within the warp
    int warpsInGrid = gridDim.x * warpsInBlock;
    int C_per_iteration = WARP_SIZE * x128::size;
    int iterations_C = CEIL_DIV(C, C_per_iteration);

    // shared memory is used only for weight gradients
    size_t rounded_C = CEIL_DIV(C, (32 * x128::size)) * (32 * x128::size);
    float* dweight_shared = shared;
    float* dweight_tmp_shared = shared + rounded_C + f128::size * BLOCK_SIZE - 2 * WARP_SIZE * f128::size;

    // init shared memory to zero
    for(int i = threadIdx.x * f128::size; i < rounded_C; i += BLOCK_SIZE * f128::size) {
        store128(dweight_shared + i, f128::zeros());
    }
    __syncthreads();

    for (int bt = baseIdx; bt < B * T; bt += warpsInGrid) {
        const floatX* dout_bt = dout + bt * C;
        const floatX* inp_bt = inp + bt * C;
        floatX* dinp_bt = dinp + bt * C;

        // compute dnorm_norm_mean term
        float dnorm_norm_mean = 0.0f;
        for (int i = warpThreadIdx * x128::size; i < C; i += WARP_SIZE * x128::size) {
            x128 dout128_i   = load128(dout_bt + i);
            x128 inp128_i    = load128(inp_bt  + i);
            x128 weight128_i = load128(weight  + i);
            for (int k = 0; k < x128::size; k++) {
                float dnorm_i = (float)weight128_i[k] * (float)dout128_i[k];
                float xi = (float)inp128_i[k];
                dnorm_norm_mean += dnorm_i * xi;
            }
        }

        const float rstd_bt = rstd[bt];
        dnorm_norm_mean = warpReduceSum(dnorm_norm_mean) / C * rstd_bt;

        for (int c = 0; c < iterations_C; c++) {
            int global_index = (warpThreadIdx * x128::size) + (c * C_per_iteration);

            x128 dout128   = x128::zeros();
            x128 inp128    = x128::zeros();
            x128 dinp128   = x128::zeros();
            x128 weight128 = x128::zeros();

            if(global_index < C) {
                dout128 = load128cs(dout_bt + global_index);
                inp128 = load128cs(inp_bt + global_index);
                dinp128 = load128(dinp_bt + global_index);
                weight128 = load128(weight + global_index);
            }

            for(int o = 0; o < x128::size / f128::size; ++o) {
                f128 dweight_f;
                for(int i = 0; i < f128::size; ++i) {
                    int x = o * f128::size + i;
                    float dout_i = (float)dout128[x];
                    float inp_i = (float)inp128[x];
                    float norm_bti = inp_i * rstd_bt;
                    dweight_f[i] = norm_bti * dout_i;

                    float dval = 0.0f;
                    dval += (float)weight128[x] * (float)dout128[x]; // term 1
                    dval -= norm_bti * dnorm_norm_mean; // term 2
                    dval *= rstd_bt; // final scale
                    dinp128[x] = (floatX)((float)dinp128[x] + dval);
                }

                if (warpId != 0) {
                    store128(dweight_tmp_shared + threadIdx.x * f128::size, dweight_f);
                }
                __syncthreads();
                if (warpId == 0) {
                    for (int j = 1; j < warpsInBlock; j++) {
                        f128 dweight_tmp = load128(dweight_tmp_shared + f128::size * (threadIdx.x + j * WARP_SIZE));
                        for(int i = 0; i < f128::size; ++i) {
                            dweight_f[i] += dweight_tmp[i];
                        }
                    }
                }
                __syncthreads();
                if (warpId == 0) {
                    f128 dw_old = load128(dweight_shared + global_index + f128::size * o);
                    for(int i = 0; i < f128::size; ++i) {
                        dweight_f[i] += dw_old[i];
                    }
                    store128(dweight_shared + global_index + f128::size * o, dweight_f);
                }
            }
            if(global_index < C) {
                // cache in L2 as this is read by the next kernel, but bypass L1 to minimise thrashing
                store128cg(dinp_bt + global_index, dinp128);
            }
        }
    }
    __syncthreads();
    // Each block writes its partial sum to global memory
    // The last block to finish becomes responsible for summing up all the partial sums
    unsigned int* scratchFlag = (unsigned int*)(scratch);
    // Increment scratch pointer by a full cacheline so that everything remains cacheline aligned
    scratch += 32;
    float* scratch_dweight = scratch;
    for(int i = threadIdx.x * f128::size; i < C; i += BLOCK_SIZE * f128::size) {
        // Write to global memory in the same "shared memory banking friendly" order
        store128(scratch_dweight + i + C*blockIdx.x, load128(dweight_shared + i));
    }
    __syncthreads();
    // that portion of shared memory is no longer used, so we can repurpose it for the scratch flag.
    unsigned int *tmp_flag = (unsigned int*)(shared + rounded_C);
    if (threadIdx.x == 0) {
        *tmp_flag = atomicInc(scratchFlag, gridDim.x);
    }
    __syncthreads();
    if (*tmp_flag == gridDim.x-1) {
        // Reduction of the partial sums by the final block
        for(int i = threadIdx.x * f128::size; i < C; i += BLOCK_SIZE * f128::size) {
            f128 dweight_accum = f128::zeros();

            for (int read_block_idx = 0; read_block_idx < gridDim.x; read_block_idx++) {
                f128 dweight_block = load128(scratch_dweight + i + C*read_block_idx);
                for(int k = 0; k < f128::size; ++k) {
                    dweight_accum[k] += dweight_block[k];
                }
            }
            store128(dweight_shared + i, dweight_accum);
        }
        __syncthreads();

        // convert from float/FP32 to floatX/BF16 for the final write
        for (int c = warpId; c < iterations_C; c += warpsInBlock) {
            int global_index = (warpThreadIdx * x128::size) + (c * C_per_iteration);
            if(global_index < C) {
                x128 dweight_out;
                for(int o = 0; o < x128::size / f128::size; ++o) {
                    f128 dweight_f = load128(dweight_shared + global_index + f128::size * o);
                    for(int i = 0; i < f128::size; ++i) {
                        int x = o * f128::size + i;
                        dweight_out[x] = (floatX)dweight_f[i];
                    }
                }
                store128(dweight + global_index, dweight_out);
            }
        }
    }
}

// ----------------------------------------------------------------------------
// kernel launchers

void rmsnorm_forward(floatX* out, float* rstd,
                      const floatX* inp, const floatX* weight,
                      int B, int T, int C, cudaStream_t stream) {
    int N = B * T;
    int block_size = 256; // can be tuned
    assert(block_size % 32 == 0);
    int grid_size = ceil_div(N * 32, block_size);
    rmsnorm_forward_kernel3<<<grid_size, block_size, 0, stream>>>(out, rstd, inp, weight, N, C);
    cudaCheck(cudaGetLastError());
}

void fused_residual_rmsnorm_forward(floatX* residual, floatX* normed, float* rstd,
                                 const floatX* inp1, const floatX* inp2, 
                                 const floatX* weight,
                                 int N, int C, cudaStream_t stream) {
    int block_y = 4; // can be tuned
    const int grid_size = ceil_div(N, block_y);
    size_t smem = (1 + block_y) * C * sizeof(float);

    auto status = cudaFuncSetAttribute(fused_residual_rmsnorm_forward_kernel5,
                                      cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    cudaGetLastError();
    if (status == cudaSuccess) {
        fused_residual_rmsnorm_forward_kernel5<<<grid_size, dim3(32, block_y), smem, stream>>>(
            residual, normed, rstd, inp1, inp2, weight, N, C);
    } else {
        // Fall back to non-fused version as a backup
        // First do the residual addition
        residual_forward_kernel<<<ceil_div(N * C / x128::size, 256), 256, 0, stream>>>(residual, inp1, inp2);
        // Then do the normalization
        rmsnorm_forward(normed, rstd, residual, weight, N / T, T, C, stream);
    }
    cudaCheck(cudaGetLastError());
}

void rmsnorm_backward(floatX* dinp, floatX* dweight, float* scratch,
                       const floatX* dout, const floatX* inp, const floatX* weight, const float* rstd,
                       int B, int T, int C, cudaStream_t stream) {
    const int block_size = 512;
    const int num_blocks = MAX_1024_THREADS_BLOCKS; // Use full GPU
    size_t shared_mem_size = ((CEIL_DIV(C, (32 * x128::size)) * (32 * x128::size))
                            + block_size * f128::size) * sizeof(float);

    // Zero out the scratch space for atomic flag
    cudaCheck(cudaMemsetAsync(scratch, 0, sizeof(unsigned int), stream));

    rmsnorm_backward_kernel10<<<num_blocks, block_size, shared_mem_size, stream>>>(
        dinp, dweight, scratch, dout, inp, weight, rstd, B, T, C);

    cudaCheck(cudaGetLastError());
}
