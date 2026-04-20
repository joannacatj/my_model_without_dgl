#include "graph_encoder_cuda.h"

#include <cuda_runtime.h>

#include <stdexcept>

namespace neugn::encoder {
namespace {

__global__ void AddKernel(const float* a, const float* b, int64_t total, float* out) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < total) out[idx] = a[idx] + b[idx];
}

__global__ void LinearKernel(const float* x,
                             int64_t rows,
                             int64_t in_dim,
                             const float* w,
                             const float* bias,
                             int64_t out_dim,
                             float* out) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = rows * out_dim;
  if (idx >= total) return;

  int64_t r = idx / out_dim;
  int64_t d = idx % out_dim;
  float acc = bias ? bias[d] : 0.0f;
  for (int64_t k = 0; k < in_dim; ++k) {
    acc += x[r * in_dim + k] * w[d * in_dim + k];
  }
  out[idx] = acc;
}

inline int BlocksFor(int64_t n) { return static_cast<int>((n + 255) / 256); }

}  // namespace

GraphEncoderCUDA::GraphEncoderCUDA(EncoderWeights weights) : w_(std::move(weights)) {}

void GraphEncoderCUDA::Forward(const EncoderIO& io) const {
  if (!io.feat_id || !io.graph_feature_out || !io.all_node_feature_out) {
    throw std::invalid_argument("GraphEncoderCUDA::Forward got null required pointer");
  }

  const int64_t feat_dim = w_.value_projection.out_dim;
  const int64_t total = io.num_nodes * feat_dim;

  float* h_attr = nullptr;
  float* h_deg = nullptr;
  float* h = nullptr;
  cudaMalloc(&h_attr, sizeof(float) * total);
  cudaMalloc(&h_deg, sizeof(float) * total);
  cudaMalloc(&h, sizeof(float) * total);

  ValueProjectionForwardCUDA(io.feat_id,
                             io.num_nodes,
                             w_.value_projection.in_dim,
                             w_.value_projection.weight,
                             w_.value_projection.bias,
                             feat_dim,
                             h_attr);

  // in-degree from CSR row length (destination-centered CSR)
  // io.coo_src/io.coo_dst are device pointers -> copy to host before BuildCSRFromCOO.
  std::vector<int64_t> coo_src_host(io.num_edges);
  std::vector<int64_t> coo_dst_host(io.num_edges);
  cudaMemcpy(coo_src_host.data(), io.coo_src, sizeof(int64_t) * io.num_edges, cudaMemcpyDeviceToHost);
  cudaMemcpy(coo_dst_host.data(), io.coo_dst, sizeof(int64_t) * io.num_edges, cudaMemcpyDeviceToHost);

  std::vector<int64_t> row_ptr_host;
  std::vector<int64_t> col_idx_host;
  BuildCSRFromCOO(
      io.num_nodes, coo_src_host.data(), coo_dst_host.data(), io.num_edges, &row_ptr_host, &col_idx_host);

  int64_t* d_row_ptr = nullptr;
  int64_t* d_col_idx = nullptr;
  int64_t* d_in_deg = nullptr;
  cudaMalloc(&d_row_ptr, sizeof(int64_t) * row_ptr_host.size());
  cudaMalloc(&d_col_idx, sizeof(int64_t) * col_idx_host.size());
  cudaMalloc(&d_in_deg, sizeof(int64_t) * io.num_nodes);
  cudaMemcpy(d_row_ptr, row_ptr_host.data(), sizeof(int64_t) * row_ptr_host.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(d_col_idx, col_idx_host.data(), sizeof(int64_t) * col_idx_host.size(), cudaMemcpyHostToDevice);

  // small helper on host for in-degree (aligned with Python clamp to [0,1000]).
  std::vector<int64_t> in_deg_host(io.num_nodes, 0);
  for (int64_t i = 0; i < io.num_nodes; ++i) {
    in_deg_host[i] = row_ptr_host[i + 1] - row_ptr_host[i];
  }
  cudaMemcpy(d_in_deg, in_deg_host.data(), sizeof(int64_t) * io.num_nodes, cudaMemcpyHostToDevice);

  DegreeEmbeddingForwardCUDA(d_in_deg,
                             io.num_nodes,
                             w_.degree_max,
                             w_.degree_embedding_table,
                             feat_dim,
                             h_deg);

  AddKernel<<<BlocksFor(total), 256>>>(h_attr, h_deg, total, h);

  if (w_.enable_rwse && io.rwse) {
    float* h_rwse = nullptr;
    cudaMalloc(&h_rwse, sizeof(float) * total);
    RWSEProjectionForwardCUDA(io.rwse,
                              io.num_nodes,
                              w_.rwse_projection.in_dim,
                              w_.rwse_projection.weight,
                              w_.rwse_projection.bias,
                              feat_dim,
                              h_rwse);
    AddKernel<<<BlocksFor(total), 256>>>(h, h_rwse, total, h);
    cudaFree(h_rwse);
  }

  float* curr = h;
  float* next = nullptr;
  cudaMalloc(&next, sizeof(float) * io.num_nodes * feat_dim);

  for (const auto& layer : w_.conv_layers) {
    if (layer.use_gin) {
      PureGINAggregateCUDA(d_row_ptr, d_col_idx, io.num_nodes, feat_dim, curr, layer.gin_eps, next);

      // MLP: lin1 -> BN/ReLU -> lin2
      float* t1 = nullptr;
      float* t2 = nullptr;
      cudaMalloc(&t1, sizeof(float) * io.num_nodes * layer.gin_mlp.lin1.out_dim);
      cudaMalloc(&t2, sizeof(float) * io.num_nodes * layer.gin_mlp.lin1.out_dim);

      LinearKernel<<<BlocksFor(io.num_nodes * layer.gin_mlp.lin1.out_dim), 256>>>(
          next, io.num_nodes, layer.gin_mlp.lin1.in_dim,
          layer.gin_mlp.lin1.weight, layer.gin_mlp.lin1.bias,
          layer.gin_mlp.lin1.out_dim, t1);
      BatchNormInferenceReLUCUDA(t1, io.num_nodes, layer.gin_mlp.lin1.out_dim, layer.gin_mlp.bn1, t2);
      LinearKernel<<<BlocksFor(io.num_nodes * layer.gin_mlp.lin2.out_dim), 256>>>(
          t2, io.num_nodes, layer.gin_mlp.lin2.in_dim,
          layer.gin_mlp.lin2.weight, layer.gin_mlp.lin2.bias,
          layer.gin_mlp.lin2.out_dim, next);

      cudaFree(t1);
      cudaFree(t2);
    } else {
      PureGraphConvAggregateCUDA(d_row_ptr, d_col_idx, io.num_nodes, feat_dim, curr, next);
      LinearKernel<<<BlocksFor(io.num_nodes * layer.gcn_linear.out_dim), 256>>>(
          next, io.num_nodes, layer.gcn_linear.in_dim,
          layer.gcn_linear.weight, layer.gcn_linear.bias,
          layer.gcn_linear.out_dim, next);
    }

    // each layer: BN(inference running stats) + ReLU, align with Python order.
    BatchNormInferenceReLUCUDA(next, io.num_nodes, feat_dim, layer.post_bn, next);
    std::swap(curr, next);
  }

  cudaMemcpy(io.all_node_feature_out, curr, sizeof(float) * io.num_nodes * feat_dim, cudaMemcpyDeviceToDevice);

  float* graph_count = nullptr;
  cudaMalloc(&graph_count, sizeof(float) * io.num_graphs);
  GlobalMeanPoolCUDA(curr,
                     io.batch_idx,
                     io.num_nodes,
                     io.num_graphs,
                     feat_dim,
                     io.graph_feature_out,
                     graph_count);

  cudaFree(graph_count);
  cudaFree(next);
  cudaFree(h);
  cudaFree(h_attr);
  cudaFree(h_deg);
  cudaFree(d_row_ptr);
  cudaFree(d_col_idx);
  cudaFree(d_in_deg);
}

}  // namespace neugn::encoder
