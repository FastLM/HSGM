/**
 * HSGM CUDA Kernels Implementation
 * 
 * High-performance CUDA kernels for HSGM operations:
 * - Pairwise similarity computation
 * - Adaptive thresholding
 * - Sparse graph construction
 * - Cross-segment attention
 * - Summary node aggregation
 * - Top-K retrieval
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <stdio.h>
#include <math.h>

#define TILE_SIZE 16
#define WARP_SIZE 32
#define MAX_SUMMARIES 128
#define MAX_EDGES 1024
#define EMBEDDING_DIM 768
#define HEAD_DIM 64

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// ============================================================================
// Kernel 1: Pairwise Similarity Computation
// ============================================================================

/**
 * Compute pairwise cosine similarity between tokens in a segment
 * 
 * @param embeddings: Input token embeddings [M, k, D]
 * @param similarities: Output similarity matrix [M, k, k]
 * @param M: Number of segments
 * @param k: Segment size (number of tokens)
 * @param D: Embedding dimension
 */
__global__ void compute_pairwise_similarity(
    const float* __restrict__ embeddings,
    float* __restrict__ similarities,
    const int M,
    const int k,
    const int D
) {
    __shared__ float tile_i[TILE_SIZE][EMBEDDING_DIM];
    __shared__ float tile_j[TILE_SIZE][EMBEDDING_DIM];
    
    int segment_id = blockIdx.z;
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    if (segment_id >= M || row >= k || col >= k) return;
    
    // Load embeddings into shared memory (coalesced)
    if (threadIdx.x == 0 && row < k) {
        for (int d = 0; d < D; d++) {
            tile_i[threadIdx.y][d] = 
                embeddings[segment_id * k * D + row * D + d];
        }
    }
    
    if (threadIdx.y == 0 && col < k) {
        for (int d = 0; d < D; d++) {
            tile_j[threadIdx.x][d] = 
                embeddings[segment_id * k * D + col * D + d];
        }
    }
    __syncthreads();
    
    // Compute dot product and norms
    float dot = 0.0f;
    float norm_i = 0.0f;
    float norm_j = 0.0f;
    
    #pragma unroll 8
    for (int d = 0; d < D; d++) {
        float vi = tile_i[threadIdx.y][d];
        float vj = tile_j[threadIdx.x][d];
        dot += vi * vj;
        norm_i += vi * vi;
        norm_j += vj * vj;
    }
    
    // Cosine similarity with epsilon for numerical stability
    float similarity = dot / (sqrtf(norm_i) * sqrtf(norm_j) + 1e-8f);
    
    // Write result
    similarities[segment_id * k * k + row * k + col] = similarity;
}

// Optimized version using vectorized loads (float4)
__global__ void compute_pairwise_similarity_vectorized(
    const float* __restrict__ embeddings,
    float* __restrict__ similarities,
    const int M,
    const int k,
    const int D
) {
    __shared__ float tile_i[TILE_SIZE][EMBEDDING_DIM];
    __shared__ float tile_j[TILE_SIZE][EMBEDDING_DIM];
    
    int segment_id = blockIdx.z;
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    if (segment_id >= M || row >= k || col >= k) return;
    
    // Load embeddings using float4 for better bandwidth utilization
    const float4* emb_vec = (const float4*)embeddings;
    
    if (threadIdx.x == 0 && row < k) {
        int base_idx = segment_id * k * D + row * D;
        for (int d = 0; d < D / 4; d++) {
            float4 vec = emb_vec[base_idx / 4 + d];
            tile_i[threadIdx.y][d * 4] = vec.x;
            tile_i[threadIdx.y][d * 4 + 1] = vec.y;
            tile_i[threadIdx.y][d * 4 + 2] = vec.z;
            tile_i[threadIdx.y][d * 4 + 3] = vec.w;
        }
    }
    
    if (threadIdx.y == 0 && col < k) {
        int base_idx = segment_id * k * D + col * D;
        for (int d = 0; d < D / 4; d++) {
            float4 vec = emb_vec[base_idx / 4 + d];
            tile_j[threadIdx.x][d * 4] = vec.x;
            tile_j[threadIdx.x][d * 4 + 1] = vec.y;
            tile_j[threadIdx.x][d * 4 + 2] = vec.z;
            tile_j[threadIdx.x][d * 4 + 3] = vec.w;
        }
    }
    __syncthreads();
    
    // Compute similarity with vectorized operations
    float dot = 0.0f;
    float norm_i = 0.0f;
    float norm_j = 0.0f;
    
    #pragma unroll 8
    for (int d = 0; d < D; d += 4) {
        float4 vi = *((float4*)&tile_i[threadIdx.y][d]);
        float4 vj = *((float4*)&tile_j[threadIdx.x][d]);
        
        dot += vi.x * vj.x + vi.y * vj.y + vi.z * vj.z + vi.w * vj.w;
        norm_i += vi.x * vi.x + vi.y * vi.y + vi.z * vi.z + vi.w * vi.w;
        norm_j += vj.x * vj.x + vj.y * vj.y + vj.z * vj.z + vj.w * vj.w;
    }
    
    float similarity = dot / (sqrtf(norm_i) * sqrtf(norm_j) + 1e-8f);
    similarities[segment_id * k * k + row * k + col] = similarity;
}

// ============================================================================
// Kernel 2: Adaptive Thresholding
// ============================================================================

/**
 * Compute mean and standard deviation for adaptive thresholding
 * Uses parallel reduction for efficient statistics computation
 */
__global__ void compute_threshold_statistics(
    const float* __restrict__ similarities,
    float* __restrict__ means,
    float* __restrict__ stds,
    const int M,
    const int k
) {
    __shared__ float shared_sum[256];
    __shared__ float shared_sum_sq[256];
    
    int segment_id = blockIdx.x;
    int tid = threadIdx.x;
    
    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;
    int N = k * k;
    
    // Each thread processes multiple elements
    for (int idx = tid; idx < N; idx += blockDim.x) {
        float val = similarities[segment_id * N + idx];
        local_sum += val;
        local_sum_sq += val * val;
    }
    
    shared_sum[tid] = local_sum;
    shared_sum_sq[tid] = local_sum_sq;
    __syncthreads();
    
    // Parallel reduction using sequential addressing
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
            shared_sum_sq[tid] += shared_sum_sq[tid + stride];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        float mean = shared_sum[0] / N;
        float variance = (shared_sum_sq[0] / N) - (mean * mean);
        float std = sqrtf(fmaxf(variance, 0.0f));  // Ensure non-negative
        
        means[segment_id] = mean;
        stds[segment_id] = std;
    }
}

// ============================================================================
// Kernel 3: Sparse Edge Creation
// ============================================================================

// Atomic max for floats
__device__ float atomicMax_float(float* address, float val) {
    int* address_as_int = (int*)address;
    int old = *address_as_int, assumed;
    
    do {
        assumed = old;
        old = atomicCAS(address_as_int, assumed,
                       __float_as_int(fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    
    return __int_as_float(old);
}

/**
 * Create sparse edges based on adaptive threshold
 * Uses COO (Coordinate) format for sparse storage
 */
__global__ void create_sparse_edges(
    const float* __restrict__ similarities,
    const float* __restrict__ means,
    const float* __restrict__ stds,
    int* __restrict__ edge_indices,      // [M, max_edges, 2]
    float* __restrict__ edge_weights,    // [M, max_edges]
    int* __restrict__ edge_counts,       // [M]
    const float alpha,
    const float beta,
    const int M,
    const int k,
    const int max_edges
) {
    int segment_id = blockIdx.y;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N = k * k;
    
    if (segment_id >= M || idx >= N) return;
    
    int row = idx / k;
    int col = idx % k;
    
    // Only process upper triangular (avoid duplicates)
    if (row >= col) return;
    
    float similarity = similarities[segment_id * N + idx];
    float threshold = alpha * means[segment_id] + beta * stds[segment_id];
    
    // Create edge if similarity exceeds threshold
    if (similarity >= threshold) {
        int edge_idx = atomicAdd(&edge_counts[segment_id], 1);
        
        if (edge_idx < max_edges) {
            int base = segment_id * max_edges;
            edge_indices[base * 2 + edge_idx * 2] = row;
            edge_indices[base * 2 + edge_idx * 2 + 1] = col;
            edge_weights[base + edge_idx] = similarity;
        }
    }
}

// ============================================================================
// Kernel 4: Warp-Level Reduction Utilities
// ============================================================================

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

// ============================================================================
// Kernel 5: Cross-Segment Multi-Head Attention
// ============================================================================

/**
 * Compute multi-head attention between current segment and previous summaries
 * 
 * @param query: Query vectors [M, seq_len, D]
 * @param key: Key vectors from summaries [M, num_summaries, D]
 * @param value: Value vectors from summaries [M, num_summaries, D]
 * @param output: Attention output [M, seq_len, D]
 */
__global__ void cross_segment_attention(
    const float* __restrict__ query,
    const float* __restrict__ key,
    const float* __restrict__ value,
    float* __restrict__ output,
    const int M,
    const int seq_len,
    const int num_summaries,
    const int D,
    const int num_heads
) {
    __shared__ float shared_kv[MAX_SUMMARIES][HEAD_DIM];
    
    int segment_id = blockIdx.y;
    int head_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    int token_id = blockIdx.x * WARP_SIZE + lane_id;
    
    if (segment_id >= M || token_id >= seq_len || head_id >= num_heads) return;
    
    int head_dim = D / num_heads;
    int head_offset = head_id * head_dim;
    
    // Load query vector for this token and head
    float q[HEAD_DIM];
    #pragma unroll
    for (int d = 0; d < head_dim; d++) {
        q[d] = query[segment_id * seq_len * D + token_id * D + head_offset + d];
    }
    
    // Load keys into shared memory cooperatively
    if (lane_id < num_summaries) {
        for (int d = 0; d < head_dim; d++) {
            shared_kv[lane_id][d] = 
                key[segment_id * num_summaries * D + lane_id * D + head_offset + d];
        }
    }
    __syncthreads();
    
    // Compute attention scores
    float scores[MAX_SUMMARIES];
    float max_score = -INFINITY;
    
    for (int i = 0; i < num_summaries; i++) {
        float score = 0.0f;
        #pragma unroll
        for (int d = 0; d < head_dim; d++) {
            score += q[d] * shared_kv[i][d];
        }
        score /= sqrtf((float)head_dim);
        scores[i] = score;
        max_score = fmaxf(max_score, score);
    }
    
    // Softmax normalization
    float sum_exp = 0.0f;
    for (int i = 0; i < num_summaries; i++) {
        scores[i] = expf(scores[i] - max_score);
        sum_exp += scores[i];
    }
    
    // Weighted sum of values
    float out[HEAD_DIM];
    #pragma unroll
    for (int d = 0; d < head_dim; d++) {
        out[d] = 0.0f;
    }
    
    for (int i = 0; i < num_summaries; i++) {
        float weight = scores[i] / (sum_exp + 1e-8f);
        for (int d = 0; d < head_dim; d++) {
            out[d] += weight * value[segment_id * num_summaries * D + 
                                    i * D + head_offset + d];
        }
    }
    
    // Write output
    for (int d = 0; d < head_dim; d++) {
        output[segment_id * seq_len * D + token_id * D + head_offset + d] = out[d];
    }
}

// ============================================================================
// Kernel 6: Summary Node Aggregation
// ============================================================================

/**
 * Aggregate local graph nodes into summary node using mean and max pooling
 */
__global__ void aggregate_summary_nodes(
    const float* __restrict__ node_embeddings,
    float* __restrict__ summary_nodes,
    const int M,
    const int k,
    const int D
) {
    __shared__ float shared_mean[EMBEDDING_DIM];
    __shared__ float shared_max[EMBEDDING_DIM];
    
    int segment_id = blockIdx.x;
    int tid = threadIdx.x;
    
    // Initialize shared memory
    if (tid < D) {
        shared_mean[tid] = 0.0f;
        shared_max[tid] = -INFINITY;
    }
    __syncthreads();
    
    // Each thread processes multiple nodes
    for (int node = tid; node < k; node += blockDim.x) {
        for (int d = 0; d < D; d++) {
            float val = node_embeddings[segment_id * k * D + node * D + d];
            atomicAdd(&shared_mean[d], val / k);
            atomicMax_float(&shared_max[d], val);
        }
    }
    __syncthreads();
    
    // Concatenate mean and max pooling
    if (tid < D) {
        summary_nodes[segment_id * 2 * D + tid] = shared_mean[tid];
        summary_nodes[segment_id * 2 * D + D + tid] = shared_max[tid];
    }
}

// ============================================================================
// Kernel 7: Top-K Retrieval
// ============================================================================

/**
 * Find top-K most similar summary nodes for a query
 * Uses bitonic sort for efficient top-K selection
 */
__global__ void top_k_retrieval(
    const float* __restrict__ query,
    const float* __restrict__ summary_nodes,
    int* __restrict__ top_k_indices,
    float* __restrict__ top_k_scores,
    const int M,
    const int D_summary,
    const int K
) {
    __shared__ float shared_scores[MAX_SUMMARIES];
    __shared__ int shared_indices[MAX_SUMMARIES];
    
    int tid = threadIdx.x;
    
    // Compute similarity scores
    for (int i = tid; i < M; i += blockDim.x) {
        float score = 0.0f;
        float query_norm = 0.0f;
        float node_norm = 0.0f;
        
        for (int d = 0; d < D_summary; d++) {
            float q = query[d];
            float n = summary_nodes[i * D_summary + d];
            score += q * n;
            query_norm += q * q;
            node_norm += n * n;
        }
        
        score /= (sqrtf(query_norm) * sqrtf(node_norm) + 1e-8f);
        shared_scores[i] = score;
        shared_indices[i] = i;
    }
    
    // Pad with negative infinity
    for (int i = M + tid; i < MAX_SUMMARIES; i += blockDim.x) {
        shared_scores[i] = -INFINITY;
        shared_indices[i] = -1;
    }
    __syncthreads();
    
    // Bitonic sort (descending order)
    for (int k = 2; k <= MAX_SUMMARIES; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int i = tid; i < MAX_SUMMARIES; i += blockDim.x) {
                int ixj = i ^ j;
                if (ixj > i) {
                    bool swap_needed = ((i & k) == 0 && shared_scores[i] < shared_scores[ixj]) ||
                                     ((i & k) != 0 && shared_scores[i] > shared_scores[ixj]);
                    
                    if (swap_needed) {
                        float temp_score = shared_scores[i];
                        int temp_idx = shared_indices[i];
                        shared_scores[i] = shared_scores[ixj];
                        shared_indices[i] = shared_indices[ixj];
                        shared_scores[ixj] = temp_score;
                        shared_indices[ixj] = temp_idx;
                    }
                }
            }
            __syncthreads();
        }
    }
    
    // Copy top-K results
    if (tid < K) {
        top_k_indices[tid] = shared_indices[tid];
        top_k_scores[tid] = shared_scores[tid];
    }
}

// ============================================================================
// Host Functions
// ============================================================================

extern "C" {

void launch_pairwise_similarity(
    const float* embeddings,
    float* similarities,
    int M, int k, int D,
    cudaStream_t stream
) {
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((k + TILE_SIZE - 1) / TILE_SIZE,
              (k + TILE_SIZE - 1) / TILE_SIZE,
              M);
    
    compute_pairwise_similarity_vectorized<<<grid, block, 0, stream>>>(
        embeddings, similarities, M, k, D
    );
    
    CUDA_CHECK(cudaGetLastError());
}

void launch_adaptive_thresholding(
    const float* similarities,
    float* means,
    float* stds,
    int M, int k,
    cudaStream_t stream
) {
    dim3 block(256);
    dim3 grid(M);
    
    compute_threshold_statistics<<<grid, block, 0, stream>>>(
        similarities, means, stds, M, k
    );
    
    CUDA_CHECK(cudaGetLastError());
}

void launch_create_sparse_edges(
    const float* similarities,
    const float* means,
    const float* stds,
    int* edge_indices,
    float* edge_weights,
    int* edge_counts,
    float alpha, float beta,
    int M, int k, int max_edges,
    cudaStream_t stream
) {
    dim3 block(256);
    dim3 grid((k * k + 255) / 256, M);
    
    create_sparse_edges<<<grid, block, 0, stream>>>(
        similarities, means, stds,
        edge_indices, edge_weights, edge_counts,
        alpha, beta, M, k, max_edges
    );
    
    CUDA_CHECK(cudaGetLastError());
}

void launch_cross_attention(
    const float* query,
    const float* key,
    const float* value,
    float* output,
    int M, int seq_len, int num_summaries, int D, int num_heads,
    cudaStream_t stream
) {
    dim3 block(num_heads * WARP_SIZE);
    dim3 grid((seq_len + WARP_SIZE - 1) / WARP_SIZE, M);
    
    cross_segment_attention<<<grid, block, 0, stream>>>(
        query, key, value, output,
        M, seq_len, num_summaries, D, num_heads
    );
    
    CUDA_CHECK(cudaGetLastError());
}

void launch_summary_aggregation(
    const float* node_embeddings,
    float* summary_nodes,
    int M, int k, int D,
    cudaStream_t stream
) {
    dim3 block(256);
    dim3 grid(M);
    
    aggregate_summary_nodes<<<grid, block, 0, stream>>>(
        node_embeddings, summary_nodes, M, k, D
    );
    
    CUDA_CHECK(cudaGetLastError());
}

void launch_top_k_retrieval(
    const float* query,
    const float* summary_nodes,
    int* top_k_indices,
    float* top_k_scores,
    int M, int D_summary, int K,
    cudaStream_t stream
) {
    dim3 block(256);
    dim3 grid(1);
    
    top_k_retrieval<<<grid, block, 0, stream>>>(
        query, summary_nodes,
        top_k_indices, top_k_scores,
        M, D_summary, K
    );
    
    CUDA_CHECK(cudaGetLastError());
}

} // extern "C"
