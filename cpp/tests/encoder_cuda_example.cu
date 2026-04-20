// CUDA 示例：直接调用 cpp/src/encoder/graph_encoder_cuda.{h,cu} 得到输出并打印。
//
// 编译示例（按实际 CUDA 路径调整）:
//   nvcc -std=c++17 \
//     cpp/tests/encoder_cuda_example.cu \
//     cpp/src/encoder/encoder_cuda_kernels.cu \
//     cpp/src/encoder/graph_encoder_cuda.cu \
//     -Icpp/src/encoder -o /tmp/encoder_cuda_example
//
// 运行:
//   /tmp/encoder_cuda_example

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

#include "graph_encoder_cuda.h"

namespace {

template <typename T>
T* CopyHostToDevice(const std::vector<T>& host) {
  T* dev = nullptr;
  cudaMalloc(&dev, sizeof(T) * host.size());
  cudaMemcpy(dev, host.data(), sizeof(T) * host.size(), cudaMemcpyHostToDevice);
  return dev;
}

void PrintTensor(const char* name, const std::vector<float>& x, int rows, int cols) {
  std::cout << name << " shape=(" << rows << "," << cols << ")\n";
  std::cout << name << " values=";
  for (float v : x) {
    std::cout << " " << std::round(v * 1e6f) / 1e6f;
  }
  std::cout << "\n";
}

}  // namespace

int main() {
  using namespace neugn::encoder;

  // 与 Python 示例一致
  const int64_t num_nodes = 4;
  const int64_t num_edges = 6;
  const int64_t feat_dim = 4;
  const int64_t fixed_input_dim = 8;

  std::vector<int64_t> coo_src{0, 1, 2, 2, 3, 0};
  std::vector<int64_t> coo_dst{1, 2, 0, 3, 1, 3};
  std::vector<int64_t> feat_id{1, 4, 2, 0};
  std::vector<int64_t> batch_idx{0, 0, 0, 0};

  std::vector<float> value_w{
      0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
      0.2, 0.1, 0.4, 0.3, 0.6, 0.5, 0.8, 0.7,
      0.3, 0.4, 0.1, 0.2, 0.7, 0.8, 0.5, 0.6,
      0.4, 0.3, 0.2, 0.1, 0.8, 0.7, 0.6, 0.5,
  };
  std::vector<float> value_b{0.01f, -0.02f, 0.03f, -0.04f};

  std::vector<float> deg_emb(1001 * feat_dim, 0.0f);
  auto set_deg = [&](int d, std::initializer_list<float> v) {
    int i = 0;
    for (float x : v) deg_emb[d * feat_dim + i++] = x;
  };
  set_deg(0, {0.0f, 0.0f, 0.0f, 0.0f});
  set_deg(1, {0.1f, 0.0f, 0.0f, 0.1f});
  set_deg(2, {0.2f, 0.1f, 0.1f, 0.2f});
  set_deg(3, {0.3f, 0.2f, 0.2f, 0.3f});

  std::vector<float> gcn_w{
      1, 0, 0, 0,
      0, 1, 0, 0,
      0, 0, 1, 0,
      0, 0, 0, 1,
  };
  std::vector<float> gcn_b{0, 0, 0, 0};

  std::vector<float> bn_gamma{1, 1, 1, 1};
  std::vector<float> bn_beta{0, 0, 0, 0};
  std::vector<float> bn_mean{0, 0, 0, 0};
  std::vector<float> bn_var{1, 1, 1, 1};

  // device buffers
  int64_t* d_src = CopyHostToDevice(coo_src);
  int64_t* d_dst = CopyHostToDevice(coo_dst);
  int64_t* d_feat_id = CopyHostToDevice(feat_id);
  int64_t* d_batch = CopyHostToDevice(batch_idx);

  float* d_value_w = CopyHostToDevice(value_w);
  float* d_value_b = CopyHostToDevice(value_b);
  float* d_deg_emb = CopyHostToDevice(deg_emb);

  float* d_gcn_w = CopyHostToDevice(gcn_w);
  float* d_gcn_b = CopyHostToDevice(gcn_b);
  float* d_bn_gamma = CopyHostToDevice(bn_gamma);
  float* d_bn_beta = CopyHostToDevice(bn_beta);
  float* d_bn_mean = CopyHostToDevice(bn_mean);
  float* d_bn_var = CopyHostToDevice(bn_var);

  float* d_graph_feature = nullptr;
  float* d_all_nodes = nullptr;
  cudaMalloc(&d_graph_feature, sizeof(float) * feat_dim);
  cudaMalloc(&d_all_nodes, sizeof(float) * num_nodes * feat_dim);

  EncoderWeights weights;
  weights.value_projection = {.weight = d_value_w,
                              .bias = d_value_b,
                              .in_dim = fixed_input_dim,
                              .out_dim = feat_dim};
  weights.degree_embedding_table = d_deg_emb;
  weights.degree_max = 1000;
  weights.enable_rwse = false;

  ConvLayerParam layer;
  layer.use_gin = false;
  layer.gcn_linear = {.weight = d_gcn_w, .bias = d_gcn_b, .in_dim = feat_dim, .out_dim = feat_dim};
  layer.post_bn = {.gamma = d_bn_gamma,
                   .beta = d_bn_beta,
                   .running_mean = d_bn_mean,
                   .running_var = d_bn_var,
                   .eps = 1e-5f};
  weights.conv_layers.push_back(layer);

  GraphEncoderCUDA encoder(weights);

  EncoderIO io;
  io.coo_src = d_src;
  io.coo_dst = d_dst;
  io.num_nodes = num_nodes;
  io.num_edges = num_edges;
  io.batch_idx = d_batch;
  io.num_graphs = 1;
  io.feat_id = d_feat_id;
  io.rwse = nullptr;
  io.graph_feature_out = d_graph_feature;
  io.all_node_feature_out = d_all_nodes;

  encoder.Forward(io);
  cudaDeviceSynchronize();

  std::vector<float> h_graph(feat_dim);
  std::vector<float> h_nodes(num_nodes * feat_dim);
  cudaMemcpy(h_graph.data(), d_graph_feature, sizeof(float) * feat_dim, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_nodes.data(), d_all_nodes, sizeof(float) * num_nodes * feat_dim, cudaMemcpyDeviceToHost);

  std::cout << "=== CUDA Encoder Output ===\n";
  PrintTensor("graph_feature", h_graph, 1, feat_dim);
  PrintTensor("all_node_features", h_nodes, num_nodes, feat_dim);

  cudaFree(d_src);
  cudaFree(d_dst);
  cudaFree(d_feat_id);
  cudaFree(d_batch);
  cudaFree(d_value_w);
  cudaFree(d_value_b);
  cudaFree(d_deg_emb);
  cudaFree(d_gcn_w);
  cudaFree(d_gcn_b);
  cudaFree(d_bn_gamma);
  cudaFree(d_bn_beta);
  cudaFree(d_bn_mean);
  cudaFree(d_bn_var);
  cudaFree(d_graph_feature);
  cudaFree(d_all_nodes);

  return 0;
}
