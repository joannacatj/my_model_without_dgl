// demo.py-like NeuGN C++/CUDA matcher demo.
// - Loads real .graph file and value2id mapping.
// - Generates random query subgraphs from data graph.
// - Runs baseline vs NeuGN rerank and reports steps/time stats.

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <queue>
#include <random>
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
using neugn::preprocess::SimpleNodeSubgraph;

struct TensorMeta {
  std::vector<int64_t> shape;
  int64_t offset = 0;
  int64_t nbytes = 0;
};

struct DataGraph {
  int64_t num_nodes = 0;
  std::vector<int64_t> feat_id;
  std::vector<int64_t> src;
  std::vector<int64_t> dst;
  std::vector<std::vector<int64_t>> adj;
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

int64_t ParseCfgInt(const std::string& c, const std::string& key, int64_t d) {
  std::regex re("\"" + key + "\"\\s*:\\s*(\\d+)");
  std::smatch m;
  return std::regex_search(c, m, re) ? std::stoll(m[1]) : d;
}

int64_t InferEncoderLayers(const std::string& m) {
  std::regex re(R"(encoder\.batch_norms\.(\d+)\.weight)");
  int64_t mx = -1;
  for (auto it = std::sregex_iterator(m.begin(), m.end(), re); it != std::sregex_iterator(); ++it) {
    mx = std::max(mx, static_cast<int64_t>(std::stoll((*it)[1])));
  }
  return mx + 1;
}

int64_t InferDecoderLayers(const std::string& m) {
  std::regex re(R"(decoder\.layers\.(\d+)\.attention\.wq\.weight)");
  int64_t mx = -1;
  for (auto it = std::sregex_iterator(m.begin(), m.end(), re); it != std::sregex_iterator(); ++it) {
    mx = std::max(mx, static_cast<int64_t>(std::stoll((*it)[1])));
  }
  return mx + 1;
}

float* Load(const std::vector<uint8_t>& b, const TensorMeta& t) {
  float* d = nullptr;
  cudaMalloc(&d, t.nbytes);
  cudaMemcpy(d, b.data() + t.offset, t.nbytes, cudaMemcpyHostToDevice);
  return d;
}

std::string InferDatasetName(const std::string& graph_path) {
  std::vector<std::string> names = {"lastfm", "hamster", "nell", "wikics", "dblp", "youtube"};
  for (const auto& n : names) {
    if (graph_path.find(n) != std::string::npos) return n;
  }
  return "wikics";
}

std::unordered_map<std::string, int64_t> LoadValue2Id(const std::string& config_path, const std::string& name) {
  std::unordered_map<std::string, int64_t> out;
  std::string fp = config_path + "/" + name + "_value2id_mapping.csv";
  std::ifstream f(fp);
  if (!f) return out;
  std::string line;
  std::getline(f, line);  // header
  while (std::getline(f, line)) {
    if (line.empty()) continue;
    auto p = line.find(',');
    if (p == std::string::npos) continue;
    out[line.substr(0, p)] = std::stoll(line.substr(p + 1));
  }
  return out;
}

DataGraph LoadGraphLikeDemoPy(const std::string& graph_path, const std::string& dataset, const std::string& config_path) {
  std::string fp = graph_path + "/" + dataset + ".graph";
  std::ifstream f(fp);
  if (!f) throw std::runtime_error("graph file not found: " + fp);

  std::vector<int64_t> node_values;
  std::vector<int64_t> src_raw, dst_raw;
  std::string line;
  while (std::getline(f, line)) {
    if (line.empty()) continue;
    std::stringstream ss(line);
    char t;
    ss >> t;
    if (t == 'v') {
      int64_t nid, val;
      ss >> nid >> val;
      if (nid >= static_cast<int64_t>(node_values.size())) node_values.resize(static_cast<size_t>(nid + 1), 0);
      node_values[static_cast<size_t>(nid)] = val;
    } else if (t == 'e') {
      int64_t u, v;
      ss >> u >> v;
      src_raw.push_back(u);
      dst_raw.push_back(v);
    }
  }

  auto value2id = LoadValue2Id(config_path, dataset);
  DataGraph g;
  g.num_nodes = static_cast<int64_t>(node_values.size());
  g.feat_id.resize(node_values.size(), 0);
  for (size_t i = 0; i < node_values.size(); ++i) {
    auto k = std::to_string(node_values[i]);
    auto it = value2id.find(k);
    g.feat_id[i] = (it == value2id.end()) ? 0 : it->second;
  }

  g.src.reserve(src_raw.size() * 2);
  g.dst.reserve(dst_raw.size() * 2);
  for (size_t i = 0; i < src_raw.size(); ++i) {
    g.src.push_back(src_raw[i]);
    g.dst.push_back(dst_raw[i]);
    g.src.push_back(dst_raw[i]);
    g.dst.push_back(src_raw[i]);
  }

  g.adj.assign(static_cast<size_t>(g.num_nodes), {});
  for (size_t i = 0; i < g.src.size(); ++i) g.adj[static_cast<size_t>(g.src[i])].push_back(g.dst[i]);
  for (auto& nbrs : g.adj) {
    std::sort(nbrs.begin(), nbrs.end());
    nbrs.erase(std::unique(nbrs.begin(), nbrs.end()), nbrs.end());
  }
  return g;
}

SimpleGraphCPU RandomQueryFromData(const DataGraph& g, int64_t size, std::mt19937_64* rng) {
  if (g.num_nodes == 0 || size <= 0 || size > g.num_nodes) return {};
  std::uniform_int_distribution<int64_t> pick(0, g.num_nodes - 1);
  for (int retry = 0; retry < 100; ++retry) {
    int64_t start = pick(*rng);
    std::unordered_set<int64_t> sub{start};
    std::vector<int64_t> q{start};
    while (!q.empty() && static_cast<int64_t>(sub.size()) < size) {
      int64_t u = q.front();
      q.erase(q.begin());
      auto nbrs = g.adj[static_cast<size_t>(u)];
      std::shuffle(nbrs.begin(), nbrs.end(), *rng);
      for (auto v : nbrs) {
        if (!sub.count(v)) {
          sub.insert(v);
          q.push_back(v);
          if (static_cast<int64_t>(sub.size()) == size) break;
        }
      }
    }
    if (static_cast<int64_t>(sub.size()) == size) {
      std::vector<int64_t> nodes(sub.begin(), sub.end());
      std::sort(nodes.begin(), nodes.end());
      SimpleGraphCPU base;
      base.num_nodes = g.num_nodes;
      base.feat_id = g.feat_id;
      base.nid.resize(g.num_nodes);
      std::iota(base.nid.begin(), base.nid.end(), 0);
      base.edges.reserve(g.src.size());
      for (size_t i = 0; i < g.src.size(); ++i) base.edges.push_back({g.src[i], g.dst[i]});
      return SimpleNodeSubgraph(base, nodes);
    }
  }
  return {};
}

double Median(std::vector<double> x) {
  if (x.empty()) return 0.0;
  std::sort(x.begin(), x.end());
  size_t m = x.size() / 2;
  if (x.size() % 2 == 1) return x[m];
  return 0.5 * (x[m - 1] + x[m]);
}

__global__ void GatherInputKernel(const float* data_feat, const int64_t* nodes, int64_t seq, int64_t dim, float* out) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = seq * dim;
  if (idx >= total) return;
  int64_t t = idx / dim, d = idx % dim;
  int64_t n = nodes[t];
  out[idx] = (n < 0) ? 0.0f : data_feat[n * dim + d];
}

__global__ void GatherSPDKernel(const int64_t* full, const int64_t* prefix, int64_t seq, int64_t qn, int64_t* out) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = seq * seq;
  if (idx >= total) return;
  int64_t i = idx / seq, j = idx % seq;
  out[idx] = full[prefix[i] * qn + prefix[j]];
}

__global__ void CosineScoresKernel(const float* pred, const float* data_feat, const int64_t* cand, int64_t K, int64_t dim, float* out) {
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= K) return;
  int64_t n = cand[i];
  float dot = 0.f, pn = 0.f, dn = 0.f;
  for (int64_t d = 0; d < dim; ++d) {
    float p = pred[d], x = data_feat[n * dim + d];
    dot += p * x;
    pn += p * p;
    dn += x * x;
  }
  out[i] = dot / fmaxf(1e-12f, sqrtf(pn) * sqrtf(dn));
}

struct Wrapper {
  GraphEncoderCUDA* enc = nullptr;
  GraphTransformerDecoderCUDA* dec = nullptr;
  int64_t feat_dim = 0;
  int64_t sub_node_id_size = 64;
  float* d_data_node_feat = nullptr;      // [N, feat_dim]
  float* d_query_graph_feat = nullptr;    // [1, feat_dim]
  int64_t* d_query_spd = nullptr;         // [Q,Q]
  std::vector<int64_t> path_nodes;
  std::unordered_map<int64_t, std::vector<int64_t>> path_pos;
  int64_t qn = 0;

  void SetDataNodeFeat(float* d) { d_data_node_feat = d; }

  void SetQuery(const SimpleGraphCPU& q) {
    qn = q.num_nodes;
    std::vector<int64_t> src, dst, batch(qn, 0);
    src.reserve(q.edges.size());
    dst.reserve(q.edges.size());
    for (auto& e : q.edges) {
      src.push_back(e.first);
      dst.push_back(e.second);
    }

    int64_t *d_src = nullptr, *d_dst = nullptr, *d_feat = nullptr, *d_batch = nullptr;
    cudaMalloc(&d_src, sizeof(int64_t) * std::max<size_t>(src.size(), 1));
    cudaMalloc(&d_dst, sizeof(int64_t) * std::max<size_t>(dst.size(), 1));
    cudaMalloc(&d_feat, sizeof(int64_t) * qn);
    cudaMalloc(&d_batch, sizeof(int64_t) * qn);
    if (!src.empty()) cudaMemcpy(d_src, src.data(), sizeof(int64_t) * src.size(), cudaMemcpyHostToDevice);
    if (!dst.empty()) cudaMemcpy(d_dst, dst.data(), sizeof(int64_t) * dst.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_feat, q.feat_id.data(), sizeof(int64_t) * qn, cudaMemcpyHostToDevice);
    cudaMemcpy(d_batch, batch.data(), sizeof(int64_t) * qn, cudaMemcpyHostToDevice);

    float *d_g = nullptr, *d_all = nullptr;
    cudaMalloc(&d_g, sizeof(float) * feat_dim);
    cudaMalloc(&d_all, sizeof(float) * qn * feat_dim);
    EncoderIO io{d_src, d_dst, qn, static_cast<int64_t>(src.size()), d_batch, 1, d_feat, nullptr, d_g, d_all};
    enc->Forward(io);
    cudaDeviceSynchronize();

    if (d_query_graph_feat) cudaFree(d_query_graph_feat);
    cudaMalloc(&d_query_graph_feat, sizeof(float) * feat_dim);
    cudaMemcpy(d_query_graph_feat, d_g, sizeof(float) * feat_dim, cudaMemcpyDeviceToDevice);
    cudaFree(d_g); cudaFree(d_all);
    cudaFree(d_src); cudaFree(d_dst); cudaFree(d_feat); cudaFree(d_batch);

    auto h_spd = ComputeSubgraphSPD(q.edges, qn, 20);
    if (d_query_spd) cudaFree(d_query_spd);
    cudaMalloc(&d_query_spd, sizeof(int64_t) * qn * qn);
    cudaMemcpy(d_query_spd, h_spd.data(), sizeof(int64_t) * qn * qn, cudaMemcpyHostToDevice);

    path_nodes.clear();
    path_pos.clear();
    auto p = Graph2PathV2Pure(q, true, 0);
    if (!p.empty()) {
      for (auto& e : p) path_nodes.push_back(e.first);
      path_nodes.push_back(p.back().second);
    } else {
      for (int64_t i = 0; i < qn; ++i) path_nodes.push_back(i);
    }
    for (size_t i = 0; i < path_nodes.size(); ++i) path_pos[path_nodes[i]].push_back(static_cast<int64_t>(i));
  }

  std::unordered_map<int64_t, float> PredictScores(
      int64_t target_q_node,
      const std::unordered_map<int64_t, int64_t>& partial,
      const std::vector<int64_t>& cands) {
    std::unordered_map<int64_t, float> out;
    if (cands.empty()) return out;
    auto it = path_pos.find(target_q_node);
    if (it == path_pos.end()) {
      for (auto c : cands) out[c] = 0.0f;
      return out;
    }
    int64_t seq = it->second.front();
    std::vector<int64_t> h_map(seq, -1), h_prefix(seq), h_sub(seq);
    for (int64_t i = 0; i < seq; ++i) {
      int64_t qn_i = path_nodes[static_cast<size_t>(i)];
      h_prefix[i] = qn_i;
      auto m = partial.find(qn_i);
      h_map[i] = (m == partial.end()) ? -1 : m->second;
      h_sub[i] = i % sub_node_id_size;
    }

    int64_t *d_map = nullptr, *d_prefix = nullptr, *d_sub = nullptr, *d_tml = nullptr, *d_spd = nullptr, *d_cands = nullptr;
    float *d_input = nullptr, *d_out = nullptr, *d_scores = nullptr;
    auto fallback_zero_scores = [&]() {
      for (auto c : cands) out[c] = 0.0f;
      return out;
    };

    cudaError_t st = cudaSuccess;
    st = cudaMalloc(&d_map, sizeof(int64_t) * std::max<int64_t>(seq, 1));
    if (st != cudaSuccess) return fallback_zero_scores();
    st = cudaMalloc(&d_prefix, sizeof(int64_t) * std::max<int64_t>(seq, 1));
    if (st != cudaSuccess) { cudaFree(d_map); return fallback_zero_scores(); }
    st = cudaMalloc(&d_sub, sizeof(int64_t) * std::max<int64_t>(seq, 1));
    if (st != cudaSuccess) { cudaFree(d_map); cudaFree(d_prefix); return fallback_zero_scores(); }
    st = cudaMalloc(&d_tml, sizeof(int64_t));
    if (st != cudaSuccess) { cudaFree(d_map); cudaFree(d_prefix); cudaFree(d_sub); return fallback_zero_scores(); }
    if (seq > 0) {
      cudaMemcpy(d_map, h_map.data(), sizeof(int64_t) * seq, cudaMemcpyHostToDevice);
      cudaMemcpy(d_prefix, h_prefix.data(), sizeof(int64_t) * seq, cudaMemcpyHostToDevice);
      cudaMemcpy(d_sub, h_sub.data(), sizeof(int64_t) * seq, cudaMemcpyHostToDevice);
    }
    cudaMemcpy(d_tml, &seq, sizeof(int64_t), cudaMemcpyHostToDevice);

    st = cudaMalloc(&d_input, sizeof(float) * std::max<int64_t>(seq, 1) * feat_dim);
    if (st != cudaSuccess) {
      cudaFree(d_tml); cudaFree(d_sub); cudaFree(d_prefix); cudaFree(d_map);
      return fallback_zero_scores();
    }
    if (seq > 0) GatherInputKernel<<<(seq * feat_dim + 255) / 256, 256>>>(d_data_node_feat, d_map, seq, feat_dim, d_input);

    if (seq > 0) {
      st = cudaMalloc(&d_spd, sizeof(int64_t) * seq * seq);
      if (st != cudaSuccess) {
        cudaFree(d_input); cudaFree(d_tml); cudaFree(d_sub); cudaFree(d_prefix); cudaFree(d_map);
        return fallback_zero_scores();
      }
      GatherSPDKernel<<<(seq * seq + 255) / 256, 256>>>(d_query_spd, d_prefix, seq, qn, d_spd);
    }
    st = cudaMalloc(&d_out, sizeof(float) * (seq + 1) * feat_dim);
    if (st != cudaSuccess) {
      if (d_spd) cudaFree(d_spd);
      cudaFree(d_input); cudaFree(d_tml); cudaFree(d_sub); cudaFree(d_prefix); cudaFree(d_map);
      return fallback_zero_scores();
    }

    if (!d_query_graph_feat || !d_input || !d_sub || !d_tml || !d_out) {
      cudaFree(d_out); if (d_spd) cudaFree(d_spd);
      cudaFree(d_input); cudaFree(d_tml); cudaFree(d_sub); cudaFree(d_prefix); cudaFree(d_map);
      return fallback_zero_scores();
    }
    DecoderIO io{d_query_graph_feat, d_input, d_sub, d_tml, d_spd, 1, seq, feat_dim, d_out};
    dec->Forward(io);

    st = cudaMalloc(&d_cands, sizeof(int64_t) * cands.size());
    if (st != cudaSuccess) {
      cudaFree(d_out); if (d_spd) cudaFree(d_spd);
      cudaFree(d_input); cudaFree(d_tml); cudaFree(d_sub); cudaFree(d_prefix); cudaFree(d_map);
      return fallback_zero_scores();
    }
    st = cudaMalloc(&d_scores, sizeof(float) * cands.size());
    if (st != cudaSuccess) {
      cudaFree(d_cands);
      cudaFree(d_out); if (d_spd) cudaFree(d_spd);
      cudaFree(d_input); cudaFree(d_tml); cudaFree(d_sub); cudaFree(d_prefix); cudaFree(d_map);
      return fallback_zero_scores();
    }
    cudaMemcpy(d_cands, cands.data(), sizeof(int64_t) * cands.size(), cudaMemcpyHostToDevice);
    float* d_pred = d_out + seq * feat_dim;
    CosineScoresKernel<<<(cands.size() + 255) / 256, 256>>>(d_pred, d_data_node_feat, d_cands, static_cast<int64_t>(cands.size()), feat_dim, d_scores);
    cudaDeviceSynchronize();

    std::vector<float> h_scores(cands.size(), 0.f);
    cudaMemcpy(h_scores.data(), d_scores, sizeof(float) * cands.size(), cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < cands.size(); ++i) out[cands[i]] = h_scores[i];

    cudaFree(d_scores); cudaFree(d_cands); cudaFree(d_out); if (d_spd) cudaFree(d_spd);
    cudaFree(d_input); cudaFree(d_tml); cudaFree(d_sub); cudaFree(d_prefix); cudaFree(d_map);
    return out;
  }
};

struct Matcher {
  const DataGraph* data = nullptr;
  const SimpleGraphCPU* query = nullptr;
  Wrapper* wrapper = nullptr;
  bool use_neugn = false;
  std::string method_name = "VF3";
  std::unordered_map<int64_t, int64_t>* label_freq = nullptr;
  bool found = false;
  int64_t steps = 0;
  std::unordered_map<int64_t, int64_t> mapping;
  std::vector<std::vector<int64_t>> q_adj;

  void BuildQueryAdj() {
    q_adj.assign(static_cast<size_t>(query->num_nodes), {});
    for (auto& e : query->edges) {
      q_adj[e.first].push_back(e.second);
      q_adj[e.second].push_back(e.first);
    }
    for (auto& nbrs : q_adj) {
      std::sort(nbrs.begin(), nbrs.end());
      nbrs.erase(std::unique(nbrs.begin(), nbrs.end()), nbrs.end());
    }
  }

  std::vector<int64_t> QueryOrder() {
    std::vector<int64_t> nodes(query->num_nodes);
    std::iota(nodes.begin(), nodes.end(), 0);
    if (method_name == "GQL" && label_freq) {
      std::sort(nodes.begin(), nodes.end(), [&](int64_t a, int64_t b) {
        int64_t fa = 1e9, fb = 1e9;
        auto ia = label_freq->find(query->feat_id[a]);
        auto ib = label_freq->find(query->feat_id[b]);
        if (ia != label_freq->end()) fa = ia->second;
        if (ib != label_freq->end()) fb = ib->second;
        if (fa != fb) return fa < fb;
        return q_adj[a].size() > q_adj[b].size();
      });
    } else {
      std::sort(nodes.begin(), nodes.end(), [&](int64_t a, int64_t b) {
        return q_adj[a].size() > q_adj[b].size();
      });
    }
    return nodes;
  }

  std::vector<int64_t> Candidates(int64_t qn) {
    std::unordered_set<int64_t> used;
    for (auto& kv : mapping) used.insert(kv.second);
    std::vector<int64_t> c;
    for (int64_t n = 0; n < data->num_nodes; ++n) {
      if (used.count(n)) continue;
      if (data->feat_id[n] != query->feat_id[qn]) continue;
      if (data->adj[n].size() < q_adj[qn].size()) continue;
      c.push_back(n);
    }
    return c;
  }

  bool CheckJoinTopology(int64_t qn, int64_t dn) const {
    auto has_data_edge = [&](int64_t u, int64_t v) {
      const auto& nbrs = data->adj[static_cast<size_t>(u)];
      return std::binary_search(nbrs.begin(), nbrs.end(), v);
    };
    auto has_query_edge = [&](int64_t u, int64_t v) {
      const auto& nbrs = q_adj[static_cast<size_t>(u)];
      return std::binary_search(nbrs.begin(), nbrs.end(), v);
    };

    // Join-stage topology consistency:
    // for every already-matched query node q_m,
    // if (qn, q_m) is an edge in query, then (dn, d_m) must be an edge in data.
    for (const auto& kv : mapping) {
      const int64_t q_m = kv.first;
      const int64_t d_m = kv.second;
      if (has_query_edge(qn, q_m) && !has_data_edge(dn, d_m)) {
        return false;
      }
    }
    return true;
  }

  void DFS(const std::vector<int64_t>& order, int64_t idx) {
    if (found) return;
    steps++;
    if (idx == static_cast<int64_t>(order.size())) {
      found = true;
      return;
    }
    int64_t qn = order[idx];
    auto cands = Candidates(qn);
    if (cands.empty()) return;
    if (use_neugn && cands.size() > 1) {
      auto scores = wrapper->PredictScores(qn, mapping, cands);
      std::sort(cands.begin(), cands.end(), [&](int64_t a, int64_t b) { return scores[a] > scores[b]; });
    } else {
      std::sort(cands.begin(), cands.end());
    }
    for (auto v : cands) {
      if (!CheckJoinTopology(qn, v)) continue;
      mapping[qn] = v;
      DFS(order, idx + 1);
      if (found) return;
      mapping.erase(qn);
    }
  }

  int64_t Run() {
    steps = 0;
    found = false;
    mapping.clear();
    BuildQueryAdj();
    auto order = QueryOrder();
    DFS(order, 0);
    return steps;
  }
};

}  // namespace

int main(int argc, char** argv) {
  std::string manifest = "checkpoints/wikics/export_cpp/weights_manifest.json";
  std::string weights = "checkpoints/wikics/export_cpp/graphdecoder_weights.bin";
  std::string config = "checkpoints/wikics/export_cpp/export_config.json";
  std::string graph_path = "./datasets/wikics";
  std::string config_path = ".";
  int64_t num_queries = 5;
  uint64_t seed = 0;

  for (int i = 1; i + 1 < argc; i += 2) {
    std::string k = argv[i], v = argv[i + 1];
    if (k == "--manifest") manifest = v;
    else if (k == "--weights") weights = v;
    else if (k == "--config") config = v;
    else if (k == "--graph-path") graph_path = v;
    else if (k == "--config-path") config_path = v;
    else if (k == "--num-queries") num_queries = std::stoll(v);
    else if (k == "--seed") seed = static_cast<uint64_t>(std::stoll(v));
  }

  auto m = ReadText(manifest);
  auto b = ReadBin(weights);
  auto c = ReadText(config);

  std::string dataset = InferDatasetName(graph_path);
  std::cout << "Dataset: " << dataset << "\n";
  DataGraph data = LoadGraphLikeDemoPy(graph_path, dataset, config_path);
  std::cout << "Data graph: nodes=" << data.num_nodes << " edges=" << data.src.size() << "\n";

  std::unordered_map<int64_t, int64_t> label_freq;
  for (auto f : data.feat_id) label_freq[f]++;

  // -------- Build encoder weights --------
  EncoderWeights ew;
  auto vpw = Find(m, "encoder.value_projection.weight");
  auto vpb = Find(m, "encoder.value_projection.bias");
  auto deg = Find(m, "encoder.degree_embedding.weight");
  ew.value_projection = {.weight = Load(b, vpw), .bias = Load(b, vpb), .in_dim = vpw.shape[1], .out_dim = vpw.shape[0]};
  ew.degree_embedding_table = Load(b, deg);
  ew.degree_max = 1000;
  int64_t enc_layers = InferEncoderLayers(m);
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

  // -------- Build decoder weights --------
  DecoderWeights dw;
  dw.dim = ParseCfgInt(c, "dim", 512);
  dw.n_heads = ParseCfgInt(c, "n_heads", 8);
  dw.n_kv_heads = dw.n_heads;
  dw.max_spd = ParseCfgInt(c, "max_spd", 20);
  dw.sub_node_id_size = ParseCfgInt(c, "sub_node_id_size", 64);
  dw.pos_size = 1024;
  dw.n_layers = InferDecoderLayers(m);
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

  // -------- Data graph encoder forward (all node features) --------
  int64_t *d_src = nullptr, *d_dst = nullptr, *d_feat = nullptr, *d_batch = nullptr;
  cudaMalloc(&d_src, sizeof(int64_t) * data.src.size());
  cudaMalloc(&d_dst, sizeof(int64_t) * data.dst.size());
  cudaMalloc(&d_feat, sizeof(int64_t) * data.num_nodes);
  std::vector<int64_t> batch(data.num_nodes, 0);
  cudaMalloc(&d_batch, sizeof(int64_t) * data.num_nodes);
  cudaMemcpy(d_src, data.src.data(), sizeof(int64_t) * data.src.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(d_dst, data.dst.data(), sizeof(int64_t) * data.dst.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(d_feat, data.feat_id.data(), sizeof(int64_t) * data.num_nodes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_batch, batch.data(), sizeof(int64_t) * data.num_nodes, cudaMemcpyHostToDevice);
  float *d_graph = nullptr, *d_data_node_feat = nullptr;
  cudaMalloc(&d_graph, sizeof(float) * dw.dim);
  cudaMalloc(&d_data_node_feat, sizeof(float) * data.num_nodes * dw.dim);
  EncoderIO dio{d_src, d_dst, data.num_nodes, static_cast<int64_t>(data.src.size()), d_batch, 1, d_feat, nullptr, d_graph, d_data_node_feat};
  encoder.Forward(dio);
  cudaDeviceSynchronize();

  Wrapper wrapper;
  wrapper.enc = &encoder;
  wrapper.dec = &decoder;
  wrapper.feat_dim = dw.out2.out_dim;
  wrapper.sub_node_id_size = dw.sub_node_id_size;
  wrapper.SetDataNodeFeat(d_data_node_feat);

  std::vector<std::string> baselines = {"VF3", "GQL"};
  std::vector<int64_t> query_sizes = {4, 8, 16};
  std::mt19937_64 rng(seed);

  std::cout << "\n==================== EXPERIMENT START ====================\n";
  std::cout << "Generating " << num_queries << " queries for sizes: 4,8,16\n";

  for (auto qsize : query_sizes) {
    std::cout << "\n>>> Query Size: " << qsize << " Nodes\n";
    std::unordered_map<std::string, std::vector<double>> avg_steps, avg_steps_neu, avg_times, avg_times_neu;
    std::unordered_map<std::string, std::array<int, 3>> stats;  // better,worse,same
    for (auto& bname : baselines) stats[bname] = {0, 0, 0};

    for (int64_t qi = 0; qi < num_queries; ++qi) {
      auto qg = RandomQueryFromData(data, qsize, &rng);
      if (qg.num_nodes == 0) {
        std::cout << "  [Q" << (qi + 1) << "] Failed to generate connected subgraph.\n";
        continue;
      }
      std::cout << "  [Q" << (qi + 1) << "] Generated " << qg.num_nodes << " nodes, " << qg.edges.size() << " edges.\n";
      wrapper.SetQuery(qg);

      for (auto& bname : baselines) {
        Matcher m0{&data, &qg, &wrapper, false, bname, &label_freq};
        auto t0 = std::chrono::high_resolution_clock::now();
        int64_t s0 = m0.Run();
        auto t1 = std::chrono::high_resolution_clock::now();
        double dt0 = std::chrono::duration<double>(t1 - t0).count();

        Matcher m1{&data, &qg, &wrapper, true, bname, &label_freq};
        auto t2 = std::chrono::high_resolution_clock::now();
        int64_t s1 = m1.Run();
        auto t3 = std::chrono::high_resolution_clock::now();
        double dt1 = std::chrono::duration<double>(t3 - t2).count();

        avg_steps[bname].push_back(static_cast<double>(s0));
        avg_steps_neu[bname].push_back(static_cast<double>(s1));
        avg_times[bname].push_back(dt0);
        avg_times_neu[bname].push_back(dt1);

        std::string status = "Same";
        if (s1 < s0) {
          stats[bname][0]++; status = "Better";
        } else if (s1 > s0) {
          stats[bname][1]++; status = "Worse";
        } else {
          stats[bname][2]++;
        }

        std::cout << "    " << bname << ": Steps: " << s0 << " -> " << s1 << " (" << status
                  << ") | Time: " << dt0 << "s -> " << dt1 << "s\n";
      }
    }

    std::cout << "----------------------------------------------------------------------\n";
    std::cout << "Size " << qsize << " Summary:\n";
    for (auto& bname : baselines) {
      double m_pure = Median(avg_steps[bname]);
      double m_neu = Median(avg_steps_neu[bname]);
      double imp = (m_pure > 0.0) ? (m_pure - m_neu) / m_pure * 100.0 : 0.0;
      double m_t0 = Median(avg_times[bname]);
      double m_t1 = Median(avg_times_neu[bname]);
      double imp_t = (m_t0 > 0.0) ? (m_t0 - m_t1) / m_t0 * 100.0 : 0.0;
      int total = static_cast<int>(avg_steps[bname].size());
      double p_b = total ? 100.0 * stats[bname][0] / total : 0.0;
      double p_w = total ? 100.0 * stats[bname][1] / total : 0.0;
      double p_s = total ? 100.0 * stats[bname][2] / total : 0.0;
      std::cout << "  " << bname << ":\n";
      std::cout << "          [Steps] Median Imp: " << imp << "% (Orig=" << m_pure << " -> NeuGN=" << m_neu << ")\n";
      std::cout << "                  Counts: Better: " << stats[bname][0] << "/" << total << " (" << p_b
                << "%) | Worse: " << stats[bname][1] << "/" << total << " (" << p_w
                << "%) | Same: " << stats[bname][2] << "/" << total << " (" << p_s << "%)\n";
      std::cout << "          [Time]  Median Imp: " << imp_t << "% (Orig=" << m_t0 << "s -> NeuGN=" << m_t1 << "s)\n";
    }
  }

  cudaFree(d_graph); cudaFree(d_data_node_feat); cudaFree(d_src); cudaFree(d_dst); cudaFree(d_feat); cudaFree(d_batch);
  return 0;
}
