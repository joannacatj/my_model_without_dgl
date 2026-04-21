#include "encoder_cuda_kernels.cuh"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

#include <cuda_runtime.h>
#include <cub/cub.cuh>

namespace neugn::encoder {
namespace {

constexpr int kThreads = 256;

inline int BlocksFor(int64_t n) {
  return static_cast<int>((n + kThreads - 1) / kThreads);
}

__global__ void ValueProjectionKernel(const int64_t* feat_id,
                                      int64_t num_nodes,
                                      int64_t fixed_input_dim,
                                      const float* weight,
                                      const float* bias,
                                      int64_t out_dim,
                                      float* out) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = num_nodes * out_dim;
  if (idx >= total) return;

  int64_t n = idx / out_dim;
  int64_t d = idx % out_dim;
  int64_t token = feat_id[n] % fixed_input_dim;
  float v = weight[d * fixed_input_dim + token];
  if (bias) v += bias[d];
  out[idx] = v;
}

__global__ void DegreeEmbeddingKernel(const int64_t* in_degree,
                                      int64_t num_nodes,
                                      int64_t max_degree,
                                      const float* emb_table,
                                      int64_t feat_dim,
                                      float* out) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = num_nodes * feat_dim;
  if (idx >= total) return;

  int64_t n = idx / feat_dim;
  int64_t d = idx % feat_dim;
  int64_t deg = in_degree[n];
  if (deg < 0) deg = 0;
  if (deg > max_degree) deg = max_degree;
  out[idx] = emb_table[deg * feat_dim + d];
}

__global__ void RWSEProjectionKernel(const float* rwse,
                                     int64_t num_nodes,
                                     int64_t rwse_dim,
                                     const float* weight,
                                     const float* bias,
                                     int64_t feat_dim,
                                     float* out) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = num_nodes * feat_dim;
  if (idx >= total) return;

  int64_t n = idx / feat_dim;
  int64_t d = idx % feat_dim;

  float acc = bias ? bias[d] : 0.0f;
  const float* rw = rwse + n * rwse_dim;
  const float* w = weight + d * rwse_dim;
  for (int64_t k = 0; k < rwse_dim; ++k) {
    acc += rw[k] * w[k];
  }
  out[idx] = acc;
}

__global__ void GINAggregateKernel(const int64_t* row_ptr,
                                   const int64_t* col_idx,
                                   int64_t num_nodes,
                                   int64_t feat_dim,
                                   const float* x,
                                   float eps,
                                   float* out) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = num_nodes * feat_dim;
  if (idx >= total) return;

  int64_t n = idx / feat_dim;
  int64_t d = idx % feat_dim;

  float acc = (1.0f + eps) * x[n * feat_dim + d];
  int64_t begin = row_ptr[n];
  int64_t end = row_ptr[n + 1];
  for (int64_t e = begin; e < end; ++e) {
    int64_t src = col_idx[e];
    if (src >= 0 && src < num_nodes) {
      acc += x[src * feat_dim + d];
    }
  }
  out[idx] = acc;
}

__global__ void GraphConvAggregateKernel(const int64_t* row_ptr,
                                         const int64_t* col_idx,
                                         int64_t num_nodes,
                                         int64_t feat_dim,
                                         const float* x,
                                         float* out) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = num_nodes * feat_dim;
  if (idx >= total) return;

  int64_t n = idx / feat_dim;
  int64_t d = idx % feat_dim;

  int64_t deg_n = (row_ptr[n + 1] - row_ptr[n]) + 1;  // +I
  float acc = x[n * feat_dim + d] / static_cast<float>(deg_n);

  int64_t begin = row_ptr[n];
  int64_t end = row_ptr[n + 1];
  for (int64_t e = begin; e < end; ++e) {
    int64_t src = col_idx[e];
    int64_t deg_src = (row_ptr[src + 1] - row_ptr[src]) + 1;
    float norm = rsqrtf(static_cast<float>(deg_n * deg_src));
    acc += x[src * feat_dim + d] * norm;
  }
  out[idx] = acc;
}

__global__ void GlobalMeanPoolSumKernel(const float* node_x,
                                        const int64_t* batch_idx,
                                        int64_t num_nodes,
                                        int64_t feat_dim,
                                        float* out_graph,
                                        float* out_count) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = num_nodes * feat_dim;
  if (idx >= total) return;

  int64_t n = idx / feat_dim;
  int64_t d = idx % feat_dim;
  int64_t b = batch_idx[n];

  atomicAdd(out_graph + b * feat_dim + d, node_x[idx]);
  if (d == 0) atomicAdd(out_count + b, 1.0f);
}

__global__ void GlobalMeanPoolFinalizeKernel(float* out_graph,
                                             const float* out_count,
                                             int64_t num_graphs,
                                             int64_t feat_dim) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = num_graphs * feat_dim;
  if (idx >= total) return;

  int64_t b = idx / feat_dim;
  float cnt = fmaxf(out_count[b], 1.0f);
  out_graph[idx] /= cnt;
}

struct StridedLoad {
  const float* data;
  int stride;
  int col;
  __host__ __device__ StridedLoad(const float* data, int stride, int col) : data(data), stride(stride), col(col) {}
  __host__ __device__ float operator()(const int& i) const { return data[i * stride + col]; }
};

__global__ void BatchNormInferReLUKernel(const float* x,
                                         int64_t rows,
                                         int64_t feat_dim,
                                         const float* gamma,
                                         const float* beta,
                                         const float* mean,
                                         const float* var,
                                         float eps,
                                         float* out) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = rows * feat_dim;
  if (idx >= total) return;

  int64_t d = idx % feat_dim;
  float y = (x[idx] - mean[d]) / sqrtf(var[d] + eps);
  y = y * gamma[d] + beta[d];
  out[idx] = y > 0.0f ? y : 0.0f;
}

__global__ void WriteColumnNormalizedKernel(const float* col_sum,
                                            const float* counts,
                                            int num_graphs,
                                            int feat_dim,
                                            int col,
                                            float* out_graph) {
  int g = blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= num_graphs) return;
  float cnt = fmaxf(counts[g], 1.0f);
  out_graph[static_cast<int64_t>(g) * feat_dim + col] = col_sum[g] / cnt;
}

__global__ void ScatterCSRCountsKernel(const int64_t* unique_keys,
                                       const int* run_counts,
                                       int num_runs,
                                       int64_t num_nodes,
                                       int64_t* row_ptr) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_runs) return;
  int64_t key = unique_keys[i];
  if (key >= 0 && key < num_nodes) {
    row_ptr[key + 1] = static_cast<int64_t>(run_counts[i]);
  }
}

}  // namespace

void BuildCSRFromCOO(int64_t num_nodes,
                     const int64_t* coo_src,
                     const int64_t* coo_dst,
                     int64_t num_edges,
                     std::vector<int64_t>* row_ptr,
                     std::vector<int64_t>* col_idx) {
  if (!row_ptr || !col_idx) {
    throw std::invalid_argument("row_ptr/col_idx must not be null");
  }
  row_ptr->assign(num_nodes + 1, 0);
  col_idx->assign(num_edges, 0);

  for (int64_t e = 0; e < num_edges; ++e) {
    int64_t dst = coo_dst[e];
    if (dst < 0 || dst >= num_nodes) throw std::out_of_range("dst out of range");
    (*row_ptr)[dst + 1]++;
  }
  for (int64_t i = 1; i <= num_nodes; ++i) {
    (*row_ptr)[i] += (*row_ptr)[i - 1];
  }

  std::vector<int64_t> cursor = *row_ptr;
  for (int64_t e = 0; e < num_edges; ++e) {
    const int64_t dst = coo_dst[e];
    const int64_t src = coo_src[e];
    (*col_idx)[cursor[dst]++] = src;
  }
}

void BuildCSRFromCOOCUDA(int64_t num_nodes,
                         const int64_t* d_coo_src,
                         const int64_t* d_coo_dst,
                         int64_t num_edges,
                         int64_t* d_row_ptr,
                         int64_t* d_col_idx) {
  // 1) Sort by dst key to make destination-centered CSR col_idx.
  int64_t* d_keys_in = nullptr;
  int64_t* d_vals_in = nullptr;
  int64_t* d_keys_out = nullptr;
  int64_t* d_vals_out = nullptr;
  cudaMalloc(&d_keys_in, sizeof(int64_t) * num_edges);
  cudaMalloc(&d_vals_in, sizeof(int64_t) * num_edges);
  cudaMalloc(&d_keys_out, sizeof(int64_t) * num_edges);
  cudaMalloc(&d_vals_out, sizeof(int64_t) * num_edges);
  cudaMemcpy(d_keys_in, d_coo_dst, sizeof(int64_t) * num_edges, cudaMemcpyDeviceToDevice);
  cudaMemcpy(d_vals_in, d_coo_src, sizeof(int64_t) * num_edges, cudaMemcpyDeviceToDevice);

  void* sort_temp = nullptr;
  size_t sort_temp_bytes = 0;
  cub::DeviceRadixSort::SortPairs(
      sort_temp, sort_temp_bytes, d_keys_in, d_keys_out, d_vals_in, d_vals_out, num_edges);
  cudaMalloc(&sort_temp, sort_temp_bytes);
  cub::DeviceRadixSort::SortPairs(
      sort_temp, sort_temp_bytes, d_keys_in, d_keys_out, d_vals_in, d_vals_out, num_edges);

  // col_idx is sorted src by dst segments.
  cudaMemcpy(d_col_idx, d_vals_out, sizeof(int64_t) * num_edges, cudaMemcpyDeviceToDevice);

  // 2) Run-length encode sorted keys -> unique dst ids and counts.
  int64_t* d_unique = nullptr;
  int* d_counts = nullptr;
  int* d_num_runs = nullptr;
  cudaMalloc(&d_unique, sizeof(int64_t) * num_edges);
  cudaMalloc(&d_counts, sizeof(int) * num_edges);
  cudaMalloc(&d_num_runs, sizeof(int));

  void* rle_temp = nullptr;
  size_t rle_temp_bytes = 0;
  cub::DeviceRunLengthEncode::Encode(
      rle_temp, rle_temp_bytes, d_keys_out, d_unique, d_counts, d_num_runs, num_edges);
  cudaMalloc(&rle_temp, rle_temp_bytes);
  cub::DeviceRunLengthEncode::Encode(
      rle_temp, rle_temp_bytes, d_keys_out, d_unique, d_counts, d_num_runs, num_edges);

  int h_num_runs = 0;
  cudaMemcpy(&h_num_runs, d_num_runs, sizeof(int), cudaMemcpyDeviceToHost);

  // 3) row_ptr: scatter counts at [key+1], then inclusive-scan.
  cudaMemset(d_row_ptr, 0, sizeof(int64_t) * (num_nodes + 1));
  ScatterCSRCountsKernel<<<BlocksFor(h_num_runs), kThreads>>>(
      d_unique, d_counts, h_num_runs, num_nodes, d_row_ptr);

  void* scan_temp = nullptr;
  size_t scan_temp_bytes = 0;
  cub::DeviceScan::InclusiveSum(scan_temp, scan_temp_bytes, d_row_ptr, d_row_ptr, num_nodes + 1);
  cudaMalloc(&scan_temp, scan_temp_bytes);
  cub::DeviceScan::InclusiveSum(scan_temp, scan_temp_bytes, d_row_ptr, d_row_ptr, num_nodes + 1);

  cudaFree(scan_temp);
  cudaFree(rle_temp);
  cudaFree(sort_temp);
  cudaFree(d_num_runs);
  cudaFree(d_counts);
  cudaFree(d_unique);
  cudaFree(d_vals_out);
  cudaFree(d_keys_out);
  cudaFree(d_vals_in);
  cudaFree(d_keys_in);
}

void ValueProjectionForwardCUDA(const int64_t* feat_id,
                                int64_t num_nodes,
                                int64_t fixed_input_dim,
                                const float* weight,
                                const float* bias,
                                int64_t out_dim,
                                float* out) {
  ValueProjectionKernel<<<BlocksFor(num_nodes * out_dim), kThreads>>>(
      feat_id, num_nodes, fixed_input_dim, weight, bias, out_dim, out);
}

void DegreeEmbeddingForwardCUDA(const int64_t* in_degree,
                                int64_t num_nodes,
                                int64_t max_degree,
                                const float* emb_table,
                                int64_t feat_dim,
                                float* out) {
  DegreeEmbeddingKernel<<<BlocksFor(num_nodes * feat_dim), kThreads>>>(
      in_degree, num_nodes, max_degree, emb_table, feat_dim, out);
}

void RWSEProjectionForwardCUDA(const float* rwse,
                               int64_t num_nodes,
                               int64_t rwse_dim,
                               const float* weight,
                               const float* bias,
                               int64_t feat_dim,
                               float* out) {
  RWSEProjectionKernel<<<BlocksFor(num_nodes * feat_dim), kThreads>>>(
      rwse, num_nodes, rwse_dim, weight, bias, feat_dim, out);
}

void PureGINAggregateCUDA(const int64_t* csr_row_ptr,
                          const int64_t* csr_col_idx,
                          int64_t num_nodes,
                          int64_t feat_dim,
                          const float* x,
                          float eps,
                          float* out) {
  GINAggregateKernel<<<BlocksFor(num_nodes * feat_dim), kThreads>>>(
      csr_row_ptr, csr_col_idx, num_nodes, feat_dim, x, eps, out);
}

void PureGraphConvAggregateCUDA(const int64_t* csr_row_ptr,
                                const int64_t* csr_col_idx,
                                int64_t num_nodes,
                                int64_t feat_dim,
                                const float* x,
                                float* out) {
  GraphConvAggregateKernel<<<BlocksFor(num_nodes * feat_dim), kThreads>>>(
      csr_row_ptr, csr_col_idx, num_nodes, feat_dim, x, out);
}

void GlobalMeanPoolCUDA(const float* node_x,
                        const int64_t* batch_idx,
                        int64_t num_nodes,
                        int64_t num_graphs,
                        int64_t feat_dim,
                        float* out_graph,
                        float* out_count) {
  // Fast path with CUB segmented reduce (requires grouped batch indices).
  std::vector<int64_t> batch_host(num_nodes);
  cudaMemcpy(batch_host.data(), batch_idx, sizeof(int64_t) * num_nodes, cudaMemcpyDeviceToHost);

  std::vector<int> seg_offsets(static_cast<size_t>(num_graphs) + 1, 0);
  bool grouped = true;
  int cursor = 0;
  for (int g = 0; g < num_graphs; ++g) {
    seg_offsets[g] = cursor;
    while (cursor < num_nodes && batch_host[cursor] == g) ++cursor;
    if (cursor < num_nodes && batch_host[cursor] < g) grouped = false;
  }
  seg_offsets[num_graphs] = static_cast<int>(num_nodes);

  if (grouped) {
    int* d_offsets = nullptr;
    cudaMalloc(&d_offsets, sizeof(int) * (num_graphs + 1));
    cudaMemcpy(d_offsets, seg_offsets.data(), sizeof(int) * (num_graphs + 1), cudaMemcpyHostToDevice);

    float* d_counts = nullptr;
    cudaMalloc(&d_counts, sizeof(float) * num_graphs);

    // Reduce counts using all-ones iterator.
    using CountIter = cub::ConstantInputIterator<float>;
    CountIter ones(1.0f);
    void* temp_storage = nullptr;
    size_t temp_bytes = 0;
    cub::DeviceSegmentedReduce::Sum(
        temp_storage, temp_bytes, ones, d_counts, num_graphs, d_offsets, d_offsets + 1);
    cudaMalloc(&temp_storage, temp_bytes);
    cub::DeviceSegmentedReduce::Sum(
        temp_storage, temp_bytes, ones, d_counts, num_graphs, d_offsets, d_offsets + 1);
    cudaMemcpy(out_count, d_counts, sizeof(float) * num_graphs, cudaMemcpyDeviceToDevice);

    // Reduce each feature column by segments.
    for (int d = 0; d < feat_dim; ++d) {
      using BaseIter = cub::CountingInputIterator<int>;
      using InputIter = cub::TransformInputIterator<float, StridedLoad, BaseIter>;
      StridedLoad load(node_x, static_cast<int>(feat_dim), d);
      InputIter in(BaseIter(0), load);

      // out layout: [B, feat_dim], write one column with stride in a temporary buffer then scatter.
      float* d_col_sum = nullptr;
      cudaMalloc(&d_col_sum, sizeof(float) * num_graphs);

      cudaFree(temp_storage);
      temp_storage = nullptr;
      temp_bytes = 0;
      cub::DeviceSegmentedReduce::Sum(
          temp_storage, temp_bytes, in, d_col_sum, num_graphs, d_offsets, d_offsets + 1);
      cudaMalloc(&temp_storage, temp_bytes);
      cub::DeviceSegmentedReduce::Sum(
          temp_storage, temp_bytes, in, d_col_sum, num_graphs, d_offsets, d_offsets + 1);

      WriteColumnNormalizedKernel<<<BlocksFor(num_graphs), kThreads>>>(
          d_col_sum, d_counts, num_graphs, feat_dim, d, out_graph);
      cudaFree(d_col_sum);
    }

    cudaFree(temp_storage);
    cudaFree(d_counts);
    cudaFree(d_offsets);
    return;
  }

  // Fallback path when batch_idx is not grouped.
  cudaMemset(out_graph, 0, sizeof(float) * num_graphs * feat_dim);
  cudaMemset(out_count, 0, sizeof(float) * num_graphs);
  GlobalMeanPoolSumKernel<<<BlocksFor(num_nodes * feat_dim), kThreads>>>(
      node_x, batch_idx, num_nodes, feat_dim, out_graph, out_count);
  GlobalMeanPoolFinalizeKernel<<<BlocksFor(num_graphs * feat_dim), kThreads>>>(
      out_graph, out_count, num_graphs, feat_dim);
}

void BatchNormInferenceReLUCUDA(const float* x,
                                int64_t rows,
                                int64_t feat_dim,
                                BNInferenceParam bn,
                                float* out) {
  BatchNormInferReLUKernel<<<BlocksFor(rows * feat_dim), kThreads>>>(
      x, rows, feat_dim, bn.gamma, bn.beta, bn.running_mean, bn.running_var, bn.eps, out);
}

}  // namespace neugn::encoder
