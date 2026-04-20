#include "encoder_cuda_kernels.cuh"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

#include <cuda_runtime.h>

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
    acc += x[src * feat_dim + d];
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
