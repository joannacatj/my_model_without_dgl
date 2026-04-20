// CUDA 验证：使用导出的参数（bin + manifest + export_config）运行 GraphEncoderCUDA。
//
// 先执行导出:
//   python tools/export_for_cpp.py --checkpoint checkpoints/wikics/gin_checkpoint.pth --config-path . --output-dir checkpoints/wikics/export_cpp
//
// 编译:
//   nvcc -std=c++17 \
//     cpp/tests/encoder_cuda_example.cu \
//     cpp/src/encoder/encoder_cuda_kernels.cu \
//     cpp/src/encoder/graph_encoder_cuda.cu \
//     -Icpp/src/encoder -o /tmp/encoder_cuda_example
//
// 运行:
//   /tmp/encoder_cuda_example \
//     --manifest checkpoints/wikics/export_cpp/weights_manifest.json \
//     --weights checkpoints/wikics/export_cpp/graphdecoder_weights.bin \
//     --config checkpoints/wikics/export_cpp/export_config.json

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "graph_encoder_cuda.h"

namespace {

struct TensorMeta {
  std::string name;
  std::vector<int64_t> shape;
  std::string dtype;
  int64_t offset = 0;
  int64_t nbytes = 0;
};

template <typename T>
T* CopyHostToDevice(const std::vector<T>& host) {
  T* dev = nullptr;
  cudaMalloc(&dev, sizeof(T) * host.size());
  cudaMemcpy(dev, host.data(), sizeof(T) * host.size(), cudaMemcpyHostToDevice);
  return dev;
}

std::string ReadText(const std::string& path) {
  std::ifstream ifs(path);
  if (!ifs) throw std::runtime_error("failed to open: " + path);
  std::stringstream ss;
  ss << ifs.rdbuf();
  return ss.str();
}

std::vector<uint8_t> ReadBinary(const std::string& path) {
  std::ifstream ifs(path, std::ios::binary);
  if (!ifs) throw std::runtime_error("failed to open: " + path);
  ifs.seekg(0, std::ios::end);
  size_t n = static_cast<size_t>(ifs.tellg());
  ifs.seekg(0, std::ios::beg);
  std::vector<uint8_t> buf(n);
  ifs.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(n));
  return buf;
}

std::vector<int64_t> ParseShape(const std::string& s) {
  std::vector<int64_t> out;
  std::regex num_re(R"((\d+))");
  for (auto it = std::sregex_iterator(s.begin(), s.end(), num_re); it != std::sregex_iterator(); ++it) {
    out.push_back(std::stoll((*it)[1]));
  }
  return out;
}

TensorMeta FindTensorMeta(const std::string& manifest, const std::string& name) {
  std::string escaped = std::regex_replace(name, std::regex(R"([.^$|()\[\]{}*+?\\])"), R"(\$&)");
  std::regex item_re(
      "\\{[^\\{\\}]*\"name\"\\s*:\\s*\"" + escaped +
      "\"[^\\{\\}]*\"shape\"\\s*:\\s*\\[([^\\]]*)\\][^\\{\\}]*\"dtype\"\\s*:\\s*\"([^\"]*)\"[^\\{\\}]*\"offset\"\\s*:\\s*(\\d+)[^\\{\\}]*\"nbytes\"\\s*:\\s*(\\d+)");
  std::smatch m;
  if (!std::regex_search(manifest, m, item_re)) {
    throw std::runtime_error("tensor not found in manifest: " + name);
  }
  TensorMeta meta;
  meta.name = name;
  meta.shape = ParseShape(m[1].str());
  meta.dtype = m[2].str();
  meta.offset = std::stoll(m[3].str());
  meta.nbytes = std::stoll(m[4].str());
  return meta;
}

int64_t ParseConfigInt(const std::string& cfg, const std::string& key, int64_t def = 0) {
  std::regex re("\"" + key + "\"\\s*:\\s*(\\d+)");
  std::smatch m;
  if (std::regex_search(cfg, m, re)) return std::stoll(m[1].str());
  return def;
}

int64_t InferEncoderLayers(const std::string& manifest) {
  std::regex re(R"(encoder\.batch_norms\.(\d+)\.weight)");
  int64_t max_idx = -1;
  for (auto it = std::sregex_iterator(manifest.begin(), manifest.end(), re); it != std::sregex_iterator(); ++it) {
    int64_t idx = std::stoll((*it)[1].str());
    if (idx > max_idx) max_idx = idx;
  }
  return max_idx + 1;
}

float* LoadFloatTensor(const std::vector<uint8_t>& bin, const TensorMeta& meta) {
  if (meta.offset < 0 || meta.nbytes < 0 || static_cast<size_t>(meta.offset + meta.nbytes) > bin.size()) {
    throw std::runtime_error("invalid offset/nbytes for tensor: " + meta.name);
  }
  float* d = nullptr;
  cudaMalloc(&d, meta.nbytes);
  cudaMemcpy(d, bin.data() + meta.offset, meta.nbytes, cudaMemcpyHostToDevice);
  return d;
}

void PrintTensor(const char* name, const std::vector<float>& x, int rows, int cols) {
  std::cout << name << " shape=(" << rows << "," << cols << ")\n";
  std::cout << name << " values=";
  for (float v : x) std::cout << " " << v;
  std::cout << "\n";
}

}  // namespace

int main(int argc, char** argv) {
  using namespace neugn::encoder;

  std::string manifest_path = "checkpoints/wikics/export_cpp/weights_manifest.json";
  std::string weights_path = "checkpoints/wikics/export_cpp/graphdecoder_weights.bin";
  std::string config_path = "checkpoints/wikics/export_cpp/export_config.json";

  for (int i = 1; i + 1 < argc; i += 2) {
    std::string k = argv[i], v = argv[i + 1];
    if (k == "--manifest") manifest_path = v;
    if (k == "--weights") weights_path = v;
    if (k == "--config") config_path = v;
  }

  const std::string manifest = ReadText(manifest_path);
  const std::string cfg = ReadText(config_path);
  const auto bin = ReadBinary(weights_path);

  const int64_t dim = ParseConfigInt(cfg, "dim", 512);
  const int64_t n_layers = InferEncoderLayers(manifest);
  const int64_t rwse_dim = ParseConfigInt(cfg, "rwse_dim", 0);
  if (n_layers <= 0) {
    throw std::runtime_error("failed to infer encoder layer count from manifest");
  }

  // 与 python 验证脚本一致的小图
  const int64_t num_nodes = 4;
  const int64_t num_edges = 6;
  std::vector<int64_t> coo_src{0, 1, 2, 2, 3, 0};
  std::vector<int64_t> coo_dst{1, 2, 0, 3, 1, 3};
  std::vector<int64_t> feat_id{1, 4, 2, 0};
  std::vector<int64_t> batch_idx{0, 0, 0, 0};

  int64_t* d_src = CopyHostToDevice(coo_src);
  int64_t* d_dst = CopyHostToDevice(coo_dst);
  int64_t* d_feat_id = CopyHostToDevice(feat_id);
  int64_t* d_batch = CopyHostToDevice(batch_idx);
  float* d_rwse = nullptr;
  if (rwse_dim > 0) {
    std::vector<float> rwse(static_cast<size_t>(num_nodes * rwse_dim));
    for (int64_t i = 0; i < num_nodes; ++i) {
      for (int64_t j = 0; j < rwse_dim; ++j) {
        rwse[static_cast<size_t>(i * rwse_dim + j)] = 0.01f * static_cast<float>((i + 1) * (j + 1));
      }
    }
    d_rwse = CopyHostToDevice(rwse);
  }

  EncoderWeights weights;
  auto vp_w = FindTensorMeta(manifest, "encoder.value_projection.weight");
  auto vp_b = FindTensorMeta(manifest, "encoder.value_projection.bias");
  auto deg_w = FindTensorMeta(manifest, "encoder.degree_embedding.weight");
  weights.value_projection = {.weight = LoadFloatTensor(bin, vp_w),
                              .bias = LoadFloatTensor(bin, vp_b),
                              .in_dim = vp_w.shape[1],
                              .out_dim = vp_w.shape[0]};
  weights.degree_embedding_table = LoadFloatTensor(bin, deg_w);
  weights.degree_max = 1000;

  // optional rwse
  try {
    auto rwse_w = FindTensorMeta(manifest, "encoder.rwse_projection.weight");
    auto rwse_b = FindTensorMeta(manifest, "encoder.rwse_projection.bias");
    weights.enable_rwse = true;
    weights.rwse_projection = {.weight = LoadFloatTensor(bin, rwse_w),
                               .bias = LoadFloatTensor(bin, rwse_b),
                               .in_dim = rwse_w.shape[1],
                               .out_dim = rwse_w.shape[0]};
  } catch (...) {
    weights.enable_rwse = false;
  }

  // conv layers (supports gin/gcn by presence of keys)
  for (int64_t i = 0; i < n_layers; ++i) {
    ConvLayerParam layer;
    try {
      auto w1 = FindTensorMeta(manifest, "encoder.convs." + std::to_string(i) + ".mlp.0.weight");
      auto b1 = FindTensorMeta(manifest, "encoder.convs." + std::to_string(i) + ".mlp.0.bias");
      auto bn1_w = FindTensorMeta(manifest, "encoder.convs." + std::to_string(i) + ".mlp.1.weight");
      auto bn1_b = FindTensorMeta(manifest, "encoder.convs." + std::to_string(i) + ".mlp.1.bias");
      auto bn1_m = FindTensorMeta(manifest, "encoder.convs." + std::to_string(i) + ".mlp.1.running_mean");
      auto bn1_v = FindTensorMeta(manifest, "encoder.convs." + std::to_string(i) + ".mlp.1.running_var");
      auto w2 = FindTensorMeta(manifest, "encoder.convs." + std::to_string(i) + ".mlp.3.weight");
      auto b2 = FindTensorMeta(manifest, "encoder.convs." + std::to_string(i) + ".mlp.3.bias");

      layer.use_gin = true;
      layer.gin_mlp.lin1 = {.weight = LoadFloatTensor(bin, w1), .bias = LoadFloatTensor(bin, b1), .in_dim = w1.shape[1], .out_dim = w1.shape[0]};
      layer.gin_mlp.bn1 = {.gamma = LoadFloatTensor(bin, bn1_w), .beta = LoadFloatTensor(bin, bn1_b), .running_mean = LoadFloatTensor(bin, bn1_m), .running_var = LoadFloatTensor(bin, bn1_v), .eps = 1e-5f};
      layer.gin_mlp.lin2 = {.weight = LoadFloatTensor(bin, w2), .bias = LoadFloatTensor(bin, b2), .in_dim = w2.shape[1], .out_dim = w2.shape[0]};

      try {
        auto eps = FindTensorMeta(manifest, "encoder.convs." + std::to_string(i) + ".eps");
        float eps_v = 0.0f;
        std::memcpy(&eps_v, bin.data() + eps.offset, sizeof(float));
        layer.gin_eps = eps_v;
      } catch (...) {
        layer.gin_eps = 0.0f;
      }
    } catch (...) {
      auto lw = FindTensorMeta(manifest, "encoder.convs." + std::to_string(i) + ".linear.weight");
      auto lb = FindTensorMeta(manifest, "encoder.convs." + std::to_string(i) + ".linear.bias");
      layer.use_gin = false;
      layer.gcn_linear = {.weight = LoadFloatTensor(bin, lw), .bias = LoadFloatTensor(bin, lb), .in_dim = lw.shape[1], .out_dim = lw.shape[0]};
    }

    auto bnw = FindTensorMeta(manifest, "encoder.batch_norms." + std::to_string(i) + ".weight");
    auto bnb = FindTensorMeta(manifest, "encoder.batch_norms." + std::to_string(i) + ".bias");
    auto bnm = FindTensorMeta(manifest, "encoder.batch_norms." + std::to_string(i) + ".running_mean");
    auto bnv = FindTensorMeta(manifest, "encoder.batch_norms." + std::to_string(i) + ".running_var");
    layer.post_bn = {.gamma = LoadFloatTensor(bin, bnw), .beta = LoadFloatTensor(bin, bnb), .running_mean = LoadFloatTensor(bin, bnm), .running_var = LoadFloatTensor(bin, bnv), .eps = 1e-5f};

    weights.conv_layers.push_back(layer);
  }

  float* d_graph_feature = nullptr;
  float* d_all_nodes = nullptr;
  cudaMalloc(&d_graph_feature, sizeof(float) * dim);
  cudaMalloc(&d_all_nodes, sizeof(float) * num_nodes * dim);

  GraphEncoderCUDA encoder(weights);
  EncoderIO io;
  io.coo_src = d_src;
  io.coo_dst = d_dst;
  io.num_nodes = num_nodes;
  io.num_edges = num_edges;
  io.batch_idx = d_batch;
  io.num_graphs = 1;
  io.feat_id = d_feat_id;
  io.rwse = d_rwse;
  io.graph_feature_out = d_graph_feature;
  io.all_node_feature_out = d_all_nodes;

  encoder.Forward(io);
  cudaDeviceSynchronize();

  std::vector<float> h_graph(dim);
  std::vector<float> h_nodes(num_nodes * dim);
  cudaMemcpy(h_graph.data(), d_graph_feature, sizeof(float) * dim, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_nodes.data(), d_all_nodes, sizeof(float) * num_nodes * dim, cudaMemcpyDeviceToHost);

  std::cout << "=== CUDA Trained NeuGN Encoder Output ===\n";
  PrintTensor("graph_feature", h_graph, 1, static_cast<int>(dim));
  PrintTensor("all_node_features", h_nodes, static_cast<int>(num_nodes), static_cast<int>(dim));

  if (d_rwse) cudaFree(d_rwse);
  cudaFree(d_src);
  cudaFree(d_dst);
  cudaFree(d_feat_id);
  cudaFree(d_batch);
  cudaFree(d_graph_feature);
  cudaFree(d_all_nodes);

  return 0;
}
