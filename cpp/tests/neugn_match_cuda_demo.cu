// NeuGN C++/CUDA end-to-end matching demo (demo.py style, GPU model calls).
//
// Build:
// nvcc -std=c++17 \
//   cpp/tests/neugn_match_cuda_demo.cu \
//   cpp/src/encoder/encoder_cuda_kernels.cu \
//   cpp/src/encoder/graph_encoder_cuda.cu \
//   cpp/src/decoder/graph_transformer_decoder_cuda.cu \
//   cpp/src/preprocess/preprocess_utils.cpp \
//   -Icpp/src/encoder -Icpp/src/decoder -Icpp/src/preprocess \
//   -o /tmp/neugn_match_cuda_demo

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "graph_encoder_cuda.h"
#include "graph_transformer_decoder_cuda.h"
#include "preprocess_utils.h"

namespace {
using neugn::decoder::DecoderIO;
using neugn::decoder::DecoderLayerParam;
using neugn::decoder::DecoderWeights;
using neugn::decoder::GraphTransformerDecoderCUDA;
using neugn::encoder::ConvLayerParam;
using neugn::encoder::EncoderIO;
using neugn::encoder::EncoderWeights;
using neugn::encoder::GraphEncoderCUDA;
using neugn::preprocess::ComputeSubgraphSPD;
using neugn::preprocess::Graph2PathV2Pure;
using neugn::preprocess::SimpleGraphCPU;

struct TensorMeta {
  std::vector<int64_t> shape;
  int64_t offset = 0;
  int64_t nbytes = 0;
};

std::string ReadText(const std::string& p) {
  std::ifstream f(p);
  if (!f) throw std::runtime_error("open " + p);
  std::stringstream ss;
  ss << f.rdbuf();
  return ss.str();
}
std::vector<uint8_t> ReadBin(const std::string& p) {
  std::ifstream f(p, std::ios::binary);
  if (!f) throw std::runtime_error("open " + p);
  f.seekg(0, std::ios::end);
  size_t n = static_cast<size_t>(f.tellg());
  f.seekg(0);
  std::vector<uint8_t> b(n);
  f.read(reinterpret_cast<char*>(b.data()), static_cast<std::streamsize>(n));
  return b;
}
std::vector<int64_t> ParseShape(const std::string& s) {
  std::vector<int64_t> o;
  std::regex re(R"((\d+))");
  for (auto it = std::sregex_iterator(s.begin(), s.end(), re); it != std::sregex_iterator(); ++it) {
    o.push_back(std::stoll((*it)[1]));
  }
  return o;
}
TensorMeta Find(const std::string& m, const std::string& name) {
  std::string esc = std::regex_replace(name, std::regex(R"([.^$|()\[\]{}*+?\\])"), R"(\$&)");
  std::regex re("\\{[^\\{\\}]*\"name\"\\s*:\\s*\"" + esc +
                "\"[^\\{\\}]*\"shape\"\\s*:\\s*\\[([^\\]]*)\\][^\\{\\}]*\"offset\"\\s*:\\s*(\\d+)[^\\{\\}]*\"nbytes\"\\s*:\\s*(\\d+)");
  std::smatch sm;
  if (!std::regex_search(m, sm, re)) throw std::runtime_error("tensor not found: " + name);
  return {ParseShape(sm[1]), std::stoll(sm[2]), std::stoll(sm[3])};
}
int64_t Cfg(const std::string& c, const std::string& k, int64_t d) {
  std::regex re("\"" + k + "\"\\s*:\\s*(\\d+)");
  std::smatch m;
  return std::regex_search(c, m, re) ? std::stoll(m[1]) : d;
}
float* Load(const std::vector<uint8_t>& b, const TensorMeta& t) {
  float* d = nullptr;
  cudaMalloc(&d, t.nbytes);
  cudaMemcpy(d, b.data() + t.offset, t.nbytes, cudaMemcpyHostToDevice);
  return d;
}
int64_t InferEncLayers(const std::string& m) {
  std::regex re(R"(encoder\.batch_norms\.(\d+)\.weight)");
  int64_t mx = -1;
  for (auto it = std::sregex_iterator(m.begin(), m.end(), re); it != std::sregex_iterator(); ++it) {
    mx = std::max(mx, static_cast<int64_t>(std::stoll((*it)[1])));
  }
  return mx + 1;
}
int64_t InferDecLayers(const std::string& m) {
  std::regex re(R"(decoder\.layers\.(\d+)\.attention\.wq\.weight)");
  int64_t mx = -1;
  for (auto it = std::sregex_iterator(m.begin(), m.end(), re); it != std::sregex_iterator(); ++it) {
    mx = std::max(mx, static_cast<int64_t>(std::stoll((*it)[1])));
  }
  return mx + 1;
}

__global__ void GatherInputFeaturesKernel(const float* data_feat,
                                          const int64_t* map_nodes,
                                          int64_t seq,
                                          int64_t feat_dim,
                                          float* out) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = seq * feat_dim;
  if (idx >= total) return;
  int64_t t = idx / feat_dim;
  int64_t d = idx % feat_dim;
  int64_t node = map_nodes[t];
  if (node < 0) {
    out[idx] = 0.0f;
  } else {
    out[idx] = data_feat[node * feat_dim + d];
  }
}

__global__ void GatherSeqSPDKernel(const int64_t* full_spd,
                                   const int64_t* query_nodes,
                                   int64_t seq,
                                   int64_t qn,
                                   int64_t* out) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = seq * seq;
  if (idx >= total) return;
  int64_t i = idx / seq;
  int64_t j = idx % seq;
  int64_t qi = query_nodes[i], qj = query_nodes[j];
  out[idx] = full_spd[qi * qn + qj];
}

__global__ void CandidateCosineKernel(const float* pred,
                                      const float* data_feat,
                                      const int64_t* cand,
                                      int64_t K,
                                      int64_t feat_dim,
                                      float* out) {
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= K) return;
  int64_t node = cand[i];
  float dot = 0.0f, pn = 0.0f, cn = 0.0f;
  for (int64_t d = 0; d < feat_dim; ++d) {
    float pv = pred[d];
    float cv = data_feat[node * feat_dim + d];
    dot += pv * cv;
    pn += pv * pv;
    cn += cv * cv;
  }
  float denom = sqrtf(fmaxf(pn, 1e-12f)) * sqrtf(fmaxf(cn, 1e-12f));
  out[i] = dot / fmaxf(denom, 1e-12f);
}

struct NeuGNGPUWrapper {
  GraphEncoderCUDA* encoder = nullptr;
  GraphTransformerDecoderCUDA* decoder = nullptr;
  int64_t feat_dim = 0;
  int64_t sub_node_id_size = 64;
  int64_t qn = 0;

  float* d_data_node_feat = nullptr;  // [N, feat]
  float* d_query_feat = nullptr;      // [1,1,dim]
  int64_t* d_query_spd = nullptr;     // [Q,Q]
  std::vector<int64_t> path_nodes;
  std::unordered_map<int64_t, std::vector<int64_t>> node_pos;

  void SetDataNodeFeatures(float* d_feat) { d_data_node_feat = d_feat; }

  void SetQuery(const SimpleGraphCPU& qg,
                const std::vector<int64_t>& q_feat_id,
                const std::vector<int64_t>& q_src,
                const std::vector<int64_t>& q_dst,
                int64_t rwse_dim) {
    qn = qg.num_nodes;
    // query encoder forward
    int64_t *d_src = nullptr, *d_dst = nullptr, *d_feat = nullptr, *d_batch = nullptr;
    cudaMalloc(&d_src, sizeof(int64_t) * q_src.size());
    cudaMalloc(&d_dst, sizeof(int64_t) * q_dst.size());
    cudaMalloc(&d_feat, sizeof(int64_t) * q_feat_id.size());
    std::vector<int64_t> batch(qn, 0);
    cudaMalloc(&d_batch, sizeof(int64_t) * qn);
    cudaMemcpy(d_src, q_src.data(), sizeof(int64_t) * q_src.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dst, q_dst.data(), sizeof(int64_t) * q_dst.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_feat, q_feat_id.data(), sizeof(int64_t) * q_feat_id.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_batch, batch.data(), sizeof(int64_t) * qn, cudaMemcpyHostToDevice);

    float* d_graph = nullptr;
    float* d_all = nullptr;
    cudaMalloc(&d_graph, sizeof(float) * feat_dim);
    cudaMalloc(&d_all, sizeof(float) * qn * feat_dim);
    EncoderIO io{d_src, d_dst, qn, static_cast<int64_t>(q_src.size()), d_batch, 1, d_feat, nullptr, d_graph, d_all};
    encoder->Forward(io);
    cudaDeviceSynchronize();

    if (d_query_feat) cudaFree(d_query_feat);
    cudaMalloc(&d_query_feat, sizeof(float) * feat_dim);
    cudaMemcpy(d_query_feat, d_graph, sizeof(float) * feat_dim, cudaMemcpyDeviceToDevice);
    cudaFree(d_graph);
    cudaFree(d_all);
    cudaFree(d_src); cudaFree(d_dst); cudaFree(d_feat); cudaFree(d_batch);

    // query path + spd
    path_nodes.clear();
    node_pos.clear();
    auto p = Graph2PathV2Pure(qg, true, 0);
    if (!p.empty()) {
      for (auto& e : p) path_nodes.push_back(e.first);
      path_nodes.push_back(p.back().second);
    } else {
      for (int64_t i = 0; i < qn; ++i) path_nodes.push_back(i);
    }
    for (size_t i = 0; i < path_nodes.size(); ++i) node_pos[path_nodes[i]].push_back(static_cast<int64_t>(i));

    auto h_spd = ComputeSubgraphSPD(qg.edges, qn, 20);
    if (d_query_spd) cudaFree(d_query_spd);
    cudaMalloc(&d_query_spd, sizeof(int64_t) * qn * qn);
    cudaMemcpy(d_query_spd, h_spd.data(), sizeof(int64_t) * qn * qn, cudaMemcpyHostToDevice);
  }

  std::unordered_map<int64_t, float> PredictScores(
      int64_t target_q_node,
      const std::unordered_map<int64_t, int64_t>& partial,
      const std::vector<int64_t>& candidates) {
    std::unordered_map<int64_t, float> out;
    if (candidates.empty()) return out;
    auto it = node_pos.find(target_q_node);
    if (it == node_pos.end()) {
      for (auto c : candidates) out[c] = 0.0f;
      return out;
    }
    int64_t target_idx = it->second.front();
    int64_t seq = target_idx;

    float* d_input = nullptr;
    int64_t* d_sub = nullptr;
    int64_t* d_tml = nullptr;
    int64_t* d_spd = nullptr;
    float* d_out = nullptr;
    std::vector<int64_t> h_map_nodes(static_cast<size_t>(seq), -1);
    std::vector<int64_t> h_q_prefix(static_cast<size_t>(seq), 0);
    std::vector<int64_t> h_sub(static_cast<size_t>(seq), 0);
    for (int64_t i = 0; i < seq; ++i) {
      int64_t qn_i = path_nodes[static_cast<size_t>(i)];
      h_q_prefix[static_cast<size_t>(i)] = qn_i;
      auto m = partial.find(qn_i);
      h_map_nodes[static_cast<size_t>(i)] = (m == partial.end()) ? -1 : m->second;
      h_sub[static_cast<size_t>(i)] = i % sub_node_id_size;
    }

    int64_t *d_map = nullptr, *d_qprefix = nullptr;
    cudaMalloc(&d_map, sizeof(int64_t) * std::max<int64_t>(seq, 1));
    cudaMalloc(&d_qprefix, sizeof(int64_t) * std::max<int64_t>(seq, 1));
    if (seq > 0) {
      cudaMemcpy(d_map, h_map_nodes.data(), sizeof(int64_t) * seq, cudaMemcpyHostToDevice);
      cudaMemcpy(d_qprefix, h_q_prefix.data(), sizeof(int64_t) * seq, cudaMemcpyHostToDevice);
    }

    cudaMalloc(&d_input, sizeof(float) * std::max<int64_t>(seq, 1) * feat_dim);
    if (seq > 0) {
      GatherInputFeaturesKernel<<<(seq * feat_dim + 255) / 256, 256>>>(d_data_node_feat, d_map, seq, feat_dim, d_input);
    }

    cudaMalloc(&d_sub, sizeof(int64_t) * std::max<int64_t>(seq, 1));
    if (seq > 0) cudaMemcpy(d_sub, h_sub.data(), sizeof(int64_t) * seq, cudaMemcpyHostToDevice);
    cudaMalloc(&d_tml, sizeof(int64_t));
    cudaMemcpy(d_tml, &seq, sizeof(int64_t), cudaMemcpyHostToDevice);
    if (seq > 0) {
      cudaMalloc(&d_spd, sizeof(int64_t) * seq * seq);
      GatherSeqSPDKernel<<<(seq * seq + 255) / 256, 256>>>(d_query_spd, d_qprefix, seq, qn, d_spd);
    }
    cudaMalloc(&d_out, sizeof(float) * (seq + 1) * feat_dim);

    DecoderIO io{d_query_feat, d_input, d_sub, d_tml, d_spd, 1, seq, feat_dim, d_out};
    decoder->Forward(io);
    cudaDeviceSynchronize();

    float* d_pred = d_out + static_cast<int64_t>(seq) * feat_dim;
    int64_t *d_cand = nullptr;
    float* d_scores = nullptr;
    cudaMalloc(&d_cand, sizeof(int64_t) * candidates.size());
    cudaMalloc(&d_scores, sizeof(float) * candidates.size());
    cudaMemcpy(d_cand, candidates.data(), sizeof(int64_t) * candidates.size(), cudaMemcpyHostToDevice);
    CandidateCosineKernel<<<(candidates.size() + 255) / 256, 256>>>(
        d_pred, d_data_node_feat, d_cand, static_cast<int64_t>(candidates.size()), feat_dim, d_scores);
    std::vector<float> h_scores(candidates.size(), 0.0f);
    cudaMemcpy(h_scores.data(), d_scores, sizeof(float) * candidates.size(), cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < candidates.size(); ++i) out[candidates[i]] = h_scores[i];

    cudaFree(d_scores); cudaFree(d_cand);
    cudaFree(d_out); cudaFree(d_tml); cudaFree(d_sub); if (d_spd) cudaFree(d_spd);
    cudaFree(d_input); cudaFree(d_map); cudaFree(d_qprefix);
    return out;
  }
};

struct Matcher {
  std::vector<std::vector<int64_t>> data_adj;
  std::vector<int64_t> data_feat;
  std::vector<std::vector<int64_t>> query_adj;
  std::vector<int64_t> query_feat;
  NeuGNGPUWrapper* wrapper = nullptr;
  bool use_neugn = false;
  int64_t steps = 0;
  std::unordered_map<int64_t, int64_t> mapping;
  bool found = false;

  std::vector<int64_t> GetOrder() {
    std::vector<int64_t> nodes(query_adj.size());
    for (int64_t i = 0; i < static_cast<int64_t>(nodes.size()); ++i) nodes[i] = i;
    std::sort(nodes.begin(), nodes.end(), [&](int64_t a, int64_t b) {
      return query_adj[a].size() > query_adj[b].size();
    });
    return nodes;
  }

  std::vector<int64_t> Candidates(int64_t qn) {
    std::unordered_set<int64_t> used;
    for (auto& kv : mapping) used.insert(kv.second);
    std::vector<int64_t> c;
    for (int64_t n = 0; n < static_cast<int64_t>(data_adj.size()); ++n) {
      if (used.count(n)) continue;
      if (data_feat[n] != query_feat[qn]) continue;
      if (data_adj[n].size() < query_adj[qn].size()) continue;
      c.push_back(n);
    }
    return c;
  }

  void DFS(const std::vector<int64_t>& order, int64_t idx) {
    if (found) return;
    steps++;
    if (idx == static_cast<int64_t>(order.size())) {
      found = true;
      return;
    }
    int64_t q = order[idx];
    auto cands = Candidates(q);
    if (cands.empty()) return;
    if (use_neugn && cands.size() > 1) {
      auto scores = wrapper->PredictScores(q, mapping, cands);
      std::sort(cands.begin(), cands.end(), [&](int64_t a, int64_t b) { return scores[a] > scores[b]; });
    } else {
      std::sort(cands.begin(), cands.end());
    }
    for (auto v : cands) {
      mapping[q] = v;
      DFS(order, idx + 1);
      if (found) return;
      mapping.erase(q);
    }
  }
};

}  // namespace

int main(int argc, char** argv) {
  std::string manifest = "checkpoints/wikics/export_cpp/weights_manifest.json";
  std::string weights = "checkpoints/wikics/export_cpp/graphdecoder_weights.bin";
  std::string config = "checkpoints/wikics/export_cpp/export_config.json";
  for (int i = 1; i + 1 < argc; i += 2) {
    std::string k = argv[i], v = argv[i + 1];
    if (k == "--manifest") manifest = v;
    if (k == "--weights") weights = v;
    if (k == "--config") config = v;
  }
  auto m = ReadText(manifest);
  auto b = ReadBin(weights);
  auto c = ReadText(config);

  // build encoder weights
  EncoderWeights ew;
  auto vpw = Find(m, "encoder.value_projection.weight");
  auto vpb = Find(m, "encoder.value_projection.bias");
  auto deg = Find(m, "encoder.degree_embedding.weight");
  ew.value_projection = {.weight = Load(b, vpw), .bias = Load(b, vpb), .in_dim = vpw.shape[1], .out_dim = vpw.shape[0]};
  ew.degree_embedding_table = Load(b, deg);
  ew.degree_max = 1000;
  int64_t enc_layers = InferEncLayers(m);
  for (int64_t i = 0; i < enc_layers; ++i) {
    ConvLayerParam L;
    auto w1 = Find(m, "encoder.convs." + std::to_string(i) + ".mlp.0.weight");
    auto b1 = Find(m, "encoder.convs." + std::to_string(i) + ".mlp.0.bias");
    auto bnw = Find(m, "encoder.convs." + std::to_string(i) + ".mlp.1.weight");
    auto bnb = Find(m, "encoder.convs." + std::to_string(i) + ".mlp.1.bias");
    auto bnm = Find(m, "encoder.convs." + std::to_string(i) + ".mlp.1.running_mean");
    auto bnv = Find(m, "encoder.convs." + std::to_string(i) + ".mlp.1.running_var");
    auto w2 = Find(m, "encoder.convs." + std::to_string(i) + ".mlp.3.weight");
    auto b2 = Find(m, "encoder.convs." + std::to_string(i) + ".mlp.3.bias");
    L.use_gin = true;
    L.gin_mlp.lin1 = {.weight = Load(b, w1), .bias = Load(b, b1), .in_dim = w1.shape[1], .out_dim = w1.shape[0]};
    L.gin_mlp.bn1 = {.gamma = Load(b, bnw), .beta = Load(b, bnb), .running_mean = Load(b, bnm), .running_var = Load(b, bnv), .eps = 1e-5f};
    L.gin_mlp.lin2 = {.weight = Load(b, w2), .bias = Load(b, b2), .in_dim = w2.shape[1], .out_dim = w2.shape[0]};
    auto pbw = Find(m, "encoder.batch_norms." + std::to_string(i) + ".weight");
    auto pbb = Find(m, "encoder.batch_norms." + std::to_string(i) + ".bias");
    auto pbm = Find(m, "encoder.batch_norms." + std::to_string(i) + ".running_mean");
    auto pbv = Find(m, "encoder.batch_norms." + std::to_string(i) + ".running_var");
    L.post_bn = {.gamma = Load(b, pbw), .beta = Load(b, pbb), .running_mean = Load(b, pbm), .running_var = Load(b, pbv), .eps = 1e-5f};
    ew.conv_layers.push_back(L);
  }
  GraphEncoderCUDA encoder(ew);

  // build decoder weights
  DecoderWeights dw;
  dw.dim = Cfg(c, "dim", 512);
  dw.n_heads = Cfg(c, "n_heads", 8);
  dw.n_kv_heads = dw.n_heads;
  dw.max_spd = Cfg(c, "max_spd", 20);
  dw.sub_node_id_size = Cfg(c, "sub_node_id_size", 64);
  dw.pos_size = 1024;
  dw.n_layers = InferDecLayers(m);
  auto inw = Find(m, "decoder.input_projection.weight");
  auto inb = Find(m, "decoder.input_projection.bias");
  dw.input_projection = {.weight = Load(b, inw), .bias = Load(b, inb), .in_dim = inw.shape[1], .out_dim = inw.shape[0]};
  dw.sos_token = Load(b, Find(m, "decoder.sos_token"));
  dw.node_embeddings = Load(b, Find(m, "decoder.node_embeddings.ne"));
  dw.pos_embeddings = Load(b, Find(m, "decoder.pos_embeddings.pe"));
  dw.type_embeddings = Load(b, Find(m, "decoder.type_embeddings.weight"));
  dw.spd_bias_embedding = Load(b, Find(m, "decoder.spd_bias_embedding.weight"));
  dw.final_norm = {.weight = Load(b, Find(m, "decoder.norm.weight")), .eps = 1e-5f};
  auto o1w = Find(m, "decoder.output.0.weight"), o1b = Find(m, "decoder.output.0.bias");
  auto o2w = Find(m, "decoder.output.2.weight"), o2b = Find(m, "decoder.output.2.bias");
  dw.out1 = {.weight = Load(b, o1w), .bias = Load(b, o1b), .in_dim = o1w.shape[1], .out_dim = o1w.shape[0]};
  dw.out2 = {.weight = Load(b, o2w), .bias = Load(b, o2b), .in_dim = o2w.shape[1], .out_dim = o2w.shape[0]};
  for (int64_t i = 0; i < dw.n_layers; ++i) {
    DecoderLayerParam L;
    auto q = Find(m, "decoder.layers." + std::to_string(i) + ".attention.wq.weight");
    auto k = Find(m, "decoder.layers." + std::to_string(i) + ".attention.wk.weight");
    auto v = Find(m, "decoder.layers." + std::to_string(i) + ".attention.wv.weight");
    auto o = Find(m, "decoder.layers." + std::to_string(i) + ".attention.wo.weight");
    L.wq = {Load(b, q), nullptr, q.shape[1], q.shape[0]};
    L.wk = {Load(b, k), nullptr, k.shape[1], k.shape[0]};
    L.wv = {Load(b, v), nullptr, v.shape[1], v.shape[0]};
    L.wo = {Load(b, o), nullptr, o.shape[1], o.shape[0]};
    auto w1 = Find(m, "decoder.layers." + std::to_string(i) + ".feed_forward.w1.weight");
    auto w2 = Find(m, "decoder.layers." + std::to_string(i) + ".feed_forward.w2.weight");
    auto w3 = Find(m, "decoder.layers." + std::to_string(i) + ".feed_forward.w3.weight");
    L.w1 = {Load(b, w1), nullptr, w1.shape[1], w1.shape[0]};
    L.w2 = {Load(b, w2), nullptr, w2.shape[1], w2.shape[0]};
    L.w3 = {Load(b, w3), nullptr, w3.shape[1], w3.shape[0]};
    L.attention_norm = {Load(b, Find(m, "decoder.layers." + std::to_string(i) + ".attention_norm.weight")), 1e-5f};
    L.ffn_norm = {Load(b, Find(m, "decoder.layers." + std::to_string(i) + ".ffn_norm.weight")), 1e-5f};
    dw.layers.push_back(L);
  }
  GraphTransformerDecoderCUDA decoder(dw);

  // data graph
  std::vector<int64_t> data_src{0, 1, 2, 3, 4, 5, 6, 0, 2, 4, 1, 6};
  std::vector<int64_t> data_dst{1, 2, 3, 4, 5, 6, 7, 7, 5, 7, 4, 0};
  std::vector<int64_t> data_feat{1, 2, 3, 2, 1, 3, 2, 1};
  int64_t data_n = 8;
  int64_t data_e = static_cast<int64_t>(data_src.size());
  std::vector<int64_t> data_batch(data_n, 0);
  int64_t *d_dsrc = nullptr, *d_ddst = nullptr, *d_dfeat = nullptr, *d_dbatch = nullptr;
  cudaMalloc(&d_dsrc, sizeof(int64_t) * data_e);
  cudaMalloc(&d_ddst, sizeof(int64_t) * data_e);
  cudaMalloc(&d_dfeat, sizeof(int64_t) * data_n);
  cudaMalloc(&d_dbatch, sizeof(int64_t) * data_n);
  cudaMemcpy(d_dsrc, data_src.data(), sizeof(int64_t) * data_e, cudaMemcpyHostToDevice);
  cudaMemcpy(d_ddst, data_dst.data(), sizeof(int64_t) * data_e, cudaMemcpyHostToDevice);
  cudaMemcpy(d_dfeat, data_feat.data(), sizeof(int64_t) * data_n, cudaMemcpyHostToDevice);
  cudaMemcpy(d_dbatch, data_batch.data(), sizeof(int64_t) * data_n, cudaMemcpyHostToDevice);
  float *d_data_graph = nullptr, *d_data_nodes = nullptr;
  cudaMalloc(&d_data_graph, sizeof(float) * dw.dim);
  cudaMalloc(&d_data_nodes, sizeof(float) * data_n * dw.dim);
  EncoderIO dio{d_dsrc, d_ddst, data_n, data_e, d_dbatch, 1, d_dfeat, nullptr, d_data_graph, d_data_nodes};
  encoder.Forward(dio);
  cudaDeviceSynchronize();

  // query graph (subgraph)
  SimpleGraphCPU qg;
  qg.num_nodes = 4;
  qg.edges = {{0, 1}, {1, 2}, {2, 3}, {1, 3}};
  std::vector<int64_t> q_feat{1, 2, 3, 2};
  std::vector<int64_t> q_src{0, 1, 2, 1};
  std::vector<int64_t> q_dst{1, 2, 3, 3};

  NeuGNGPUWrapper wrapper;
  wrapper.encoder = &encoder;
  wrapper.decoder = &decoder;
  wrapper.feat_dim = dw.out2.out_dim;
  wrapper.sub_node_id_size = dw.sub_node_id_size;
  wrapper.SetDataNodeFeatures(d_data_nodes);
  wrapper.SetQuery(qg, q_feat, q_src, q_dst, 0);

  // build adjacency for matcher
  auto make_adj = [](int64_t n, const std::vector<int64_t>& s, const std::vector<int64_t>& d) {
    std::vector<std::vector<int64_t>> adj(static_cast<size_t>(n));
    for (size_t i = 0; i < s.size(); ++i) {
      adj[s[i]].push_back(d[i]);
      adj[d[i]].push_back(s[i]);
    }
    return adj;
  };

  Matcher base;
  base.data_adj = make_adj(data_n, data_src, data_dst);
  base.query_adj = make_adj(qg.num_nodes, q_src, q_dst);
  base.data_feat = data_feat;
  base.query_feat = q_feat;
  base.wrapper = &wrapper;
  base.use_neugn = false;

  Matcher neugn = base;
  neugn.use_neugn = true;

  auto run = [](Matcher& m) {
    auto order = m.GetOrder();
    auto t0 = std::chrono::high_resolution_clock::now();
    m.DFS(order, 0);
    cudaDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return ms;
  };

  double t_base = run(base);
  double t_neugn = run(neugn);

  std::cout << "=== NeuGN C++/CUDA Match Demo ===\n";
  std::cout << "baseline_found=" << (base.found ? 1 : 0) << " baseline_steps=" << base.steps
            << " baseline_time_ms=" << t_base << "\n";
  std::cout << "neugn_found=" << (neugn.found ? 1 : 0) << " neugn_steps=" << neugn.steps
            << " neugn_time_ms=" << t_neugn << "\n";
  std::cout << "step_improvement=" << (base.steps - neugn.steps) << "\n";

  cudaFree(d_dsrc); cudaFree(d_ddst); cudaFree(d_dfeat); cudaFree(d_dbatch);
  cudaFree(d_data_graph); cudaFree(d_data_nodes);
  return 0;
}
