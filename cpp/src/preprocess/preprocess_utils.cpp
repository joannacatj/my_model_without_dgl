#include "preprocess_utils.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>
#include <random>
#include <stdexcept>
#include <unordered_set>

namespace neugn::preprocess {
namespace {

std::vector<std::vector<int64_t>> BuildUndirectedAdj(
    const std::vector<std::pair<int64_t, int64_t>>& edges,
    int64_t num_nodes) {
  std::vector<std::vector<int64_t>> adj(static_cast<size_t>(num_nodes));
  for (const auto& [u, v] : edges) {
    if (u < 0 || v < 0 || u >= num_nodes || v >= num_nodes) continue;
    adj[static_cast<size_t>(u)].push_back(v);
    adj[static_cast<size_t>(v)].push_back(u);
  }
  return adj;
}

std::vector<std::vector<int64_t>> ConnectedComponents(
    const std::vector<std::vector<int64_t>>& adj) {
  const int64_t n = static_cast<int64_t>(adj.size());
  std::vector<int8_t> seen(static_cast<size_t>(n), 0);
  std::vector<std::vector<int64_t>> comps;
  for (int64_t s = 0; s < n; ++s) {
    if (seen[static_cast<size_t>(s)]) continue;
    std::queue<int64_t> q;
    q.push(s);
    seen[static_cast<size_t>(s)] = 1;
    std::vector<int64_t> comp;
    while (!q.empty()) {
      int64_t u = q.front();
      q.pop();
      comp.push_back(u);
      for (int64_t v : adj[static_cast<size_t>(u)]) {
        if (!seen[static_cast<size_t>(v)]) {
          seen[static_cast<size_t>(v)] = 1;
          q.push(v);
        }
      }
    }
    comps.push_back(std::move(comp));
  }
  return comps;
}

std::vector<int64_t> ShortestPathBFS(const std::vector<std::vector<int64_t>>& adj,
                                     int64_t src,
                                     int64_t dst) {
  if (src == dst) return {src};
  const int64_t n = static_cast<int64_t>(adj.size());
  std::vector<int64_t> parent(static_cast<size_t>(n), -1);
  std::queue<int64_t> q;
  q.push(src);
  parent[static_cast<size_t>(src)] = src;
  while (!q.empty()) {
    int64_t u = q.front();
    q.pop();
    for (int64_t v : adj[static_cast<size_t>(u)]) {
      if (parent[static_cast<size_t>(v)] != -1) continue;
      parent[static_cast<size_t>(v)] = u;
      if (v == dst) {
        std::vector<int64_t> path;
        int64_t cur = dst;
        while (cur != src) {
          path.push_back(cur);
          cur = parent[static_cast<size_t>(cur)];
        }
        path.push_back(src);
        std::reverse(path.begin(), path.end());
        return path;
      }
      q.push(v);
    }
  }
  return {};
}

struct MultiGraph {
  std::vector<std::pair<int64_t, int64_t>> edges;
  std::vector<std::vector<std::pair<int64_t, int64_t>>> adj;  // (neighbor, edge_id)
};

MultiGraph BuildMultiGraph(const std::vector<std::pair<int64_t, int64_t>>& edges,
                           int64_t num_nodes) {
  MultiGraph g;
  g.adj.assign(static_cast<size_t>(num_nodes), {});
  for (const auto& [u, v] : edges) {
    if (u < 0 || v < 0 || u >= num_nodes || v >= num_nodes) continue;
    const int64_t eid = static_cast<int64_t>(g.edges.size());
    g.edges.push_back({u, v});
    g.adj[static_cast<size_t>(u)].push_back({v, eid});
    g.adj[static_cast<size_t>(v)].push_back({u, eid});
  }
  for (auto& nbrs : g.adj) {
    std::sort(nbrs.begin(), nbrs.end(),
              [](const auto& a, const auto& b) {
                if (a.first != b.first) return a.first < b.first;
                return a.second < b.second;
              });
  }
  return g;
}

std::vector<std::pair<int64_t, int64_t>> EulerianCircuit(MultiGraph g, int64_t start) {
  if (start < 0 || start >= static_cast<int64_t>(g.adj.size())) return {};
  std::vector<int8_t> used(g.edges.size(), 0);
  std::vector<size_t> ptr(g.adj.size(), 0);

  std::vector<int64_t> stack;
  std::vector<int64_t> node_tour;
  stack.push_back(start);

  while (!stack.empty()) {
    int64_t u = stack.back();
    auto& nbrs = g.adj[static_cast<size_t>(u)];
    while (ptr[static_cast<size_t>(u)] < nbrs.size() &&
           used[static_cast<size_t>(nbrs[ptr[static_cast<size_t>(u)]].second)]) {
      ++ptr[static_cast<size_t>(u)];
    }
    if (ptr[static_cast<size_t>(u)] == nbrs.size()) {
      node_tour.push_back(u);
      stack.pop_back();
      continue;
    }
    const auto [v, eid] = nbrs[ptr[static_cast<size_t>(u)]];
    used[static_cast<size_t>(eid)] = 1;
    stack.push_back(v);
  }

  std::reverse(node_tour.begin(), node_tour.end());
  std::vector<std::pair<int64_t, int64_t>> path;
  for (size_t i = 1; i < node_tour.size(); ++i) {
    path.push_back({node_tour[i - 1], node_tour[i]});
  }
  return path;
}

std::vector<std::pair<int64_t, int64_t>> ConnectedGraph2Path(
    const std::vector<int64_t>& comp_nodes,
    const std::vector<std::pair<int64_t, int64_t>>& comp_edges,
    bool deterministic,
    std::mt19937_64* rng) {
  if (comp_nodes.size() <= 1) return {};
  MultiGraph mg = BuildMultiGraph(comp_edges, static_cast<int64_t>(comp_nodes.size()));

  // relabel to compact ids [0, n_comp) for path algorithms.
  std::vector<int64_t> local2global = comp_nodes;
  std::vector<int64_t> global2local(static_cast<size_t>(*std::max_element(comp_nodes.begin(), comp_nodes.end()) + 1), -1);
  for (size_t i = 0; i < comp_nodes.size(); ++i) {
    global2local[static_cast<size_t>(comp_nodes[i])] = static_cast<int64_t>(i);
  }

  std::vector<std::pair<int64_t, int64_t>> local_edges;
  local_edges.reserve(comp_edges.size());
  for (const auto& [u, v] : comp_edges) {
    local_edges.push_back({global2local[static_cast<size_t>(u)], global2local[static_cast<size_t>(v)]});
  }
  mg = BuildMultiGraph(local_edges, static_cast<int64_t>(comp_nodes.size()));

  // Eulerize: pair odd nodes deterministically by shortest path in unweighted graph.
  std::vector<int64_t> odd;
  for (int64_t i = 0; i < static_cast<int64_t>(mg.adj.size()); ++i) {
    if ((mg.adj[static_cast<size_t>(i)].size() & 1U) == 1U) odd.push_back(i);
  }
  std::sort(odd.begin(), odd.end());
  auto base_adj = BuildUndirectedAdj(local_edges, static_cast<int64_t>(comp_nodes.size()));

  while (!odd.empty()) {
    const int64_t u = odd.front();
    odd.erase(odd.begin());
    size_t best_pos = 0;
    int64_t best_len = std::numeric_limits<int64_t>::max();
    std::vector<int64_t> best_path;
    for (size_t i = 0; i < odd.size(); ++i) {
      auto p = ShortestPathBFS(base_adj, u, odd[i]);
      if (p.empty()) continue;
      const int64_t len = static_cast<int64_t>(p.size());
      if (len < best_len || (len == best_len && odd[i] < odd[best_pos])) {
        best_len = len;
        best_pos = i;
        best_path = std::move(p);
      }
    }
    if (best_path.empty()) break;
    const int64_t v = odd[best_pos];
    odd.erase(odd.begin() + static_cast<int64_t>(best_pos));

    for (size_t i = 1; i < best_path.size(); ++i) {
      local_edges.push_back({best_path[i - 1], best_path[i]});
    }
    mg = BuildMultiGraph(local_edges, static_cast<int64_t>(comp_nodes.size()));
  }

  int64_t start_local = 0;
  if (!deterministic && rng) {
    std::uniform_int_distribution<int64_t> d(0, static_cast<int64_t>(comp_nodes.size()) - 1);
    start_local = d(*rng);
  }
  auto raw_local_path = EulerianCircuit(mg, start_local);

  std::vector<std::pair<int64_t, int64_t>> raw_global_path;
  raw_global_path.reserve(raw_local_path.size());
  for (const auto& [u, v] : raw_local_path) {
    raw_global_path.push_back({local2global[static_cast<size_t>(u)], local2global[static_cast<size_t>(v)]});
  }

  // Match python behavior: shortest prefix that covers all unique undirected edges.
  std::unordered_set<uint64_t> all_unique;
  all_unique.reserve(raw_global_path.size() * 2 + 1);
  auto pack = [](int64_t a, int64_t b) -> uint64_t {
    if (a > b) std::swap(a, b);
    return (static_cast<uint64_t>(a) << 32) ^ static_cast<uint64_t>(b);
  };
  for (const auto& [u, v] : raw_global_path) all_unique.insert(pack(u, v));

  std::unordered_set<uint64_t> seen;
  size_t cut = raw_global_path.size();
  for (size_t i = 0; i < raw_global_path.size(); ++i) {
    seen.insert(pack(raw_global_path[i].first, raw_global_path[i].second));
    if (seen.size() == all_unique.size()) {
      cut = i + 1;
      break;
    }
  }
  raw_global_path.resize(cut);
  return raw_global_path;
}

}  // namespace

std::vector<float> ComputeRWSE(const std::vector<std::pair<int64_t, int64_t>>& edges,
                               int64_t num_nodes,
                               int64_t k_steps) {
  if (num_nodes < 0 || k_steps < 0) throw std::invalid_argument("num_nodes/k_steps must be non-negative");
  if (num_nodes == 0 || k_steps == 0) return {};

  const int64_t n = num_nodes;
  std::vector<float> adj(static_cast<size_t>(n * n), 0.0f);
  for (const auto& [u, v] : edges) {
    if (u < 0 || v < 0 || u >= n || v >= n) continue;
    adj[static_cast<size_t>(u * n + v)] = 1.0f;
    adj[static_cast<size_t>(v * n + u)] = 1.0f;
  }

  std::vector<float> deg(static_cast<size_t>(n), 0.0f);
  for (int64_t i = 0; i < n; ++i) {
    float s = 0.0f;
    for (int64_t j = 0; j < n; ++j) s += adj[static_cast<size_t>(i * n + j)];
    deg[static_cast<size_t>(i)] = s;
  }

  std::vector<float> P(static_cast<size_t>(n * n), 0.0f);
  for (int64_t i = 0; i < n; ++i) {
    if (deg[static_cast<size_t>(i)] <= 0.0f) continue;
    const float inv = 1.0f / deg[static_cast<size_t>(i)];
    for (int64_t j = 0; j < n; ++j) P[static_cast<size_t>(i * n + j)] = adj[static_cast<size_t>(i * n + j)] * inv;
  }

  std::vector<float> Pk = P;
  std::vector<float> rw(static_cast<size_t>(n * k_steps), 0.0f);
  for (int64_t i = 0; i < n; ++i) rw[static_cast<size_t>(i * k_steps)] = Pk[static_cast<size_t>(i * n + i)];

  std::vector<float> tmp(static_cast<size_t>(n * n), 0.0f);
  for (int64_t step = 1; step < k_steps; ++step) {
    std::fill(tmp.begin(), tmp.end(), 0.0f);
    for (int64_t i = 0; i < n; ++i) {
      for (int64_t k = 0; k < n; ++k) {
        const float a = Pk[static_cast<size_t>(i * n + k)];
        if (a == 0.0f) continue;
        for (int64_t j = 0; j < n; ++j) {
          tmp[static_cast<size_t>(i * n + j)] += a * P[static_cast<size_t>(k * n + j)];
        }
      }
    }
    Pk.swap(tmp);
    for (int64_t i = 0; i < n; ++i) {
      rw[static_cast<size_t>(i * k_steps + step)] = Pk[static_cast<size_t>(i * n + i)];
    }
  }
  return rw;
}

std::vector<int64_t> ComputeSubgraphSPD(const std::vector<std::pair<int64_t, int64_t>>& edges,
                                        int64_t num_nodes,
                                        int64_t max_spd,
                                        SPDMethod method) {
  if (num_nodes < 0) throw std::invalid_argument("num_nodes must be non-negative");
  if (num_nodes == 0) return {};

  const int64_t inf = max_spd + 1;
  std::vector<int64_t> out(static_cast<size_t>(num_nodes * num_nodes), inf);
  for (int64_t i = 0; i < num_nodes; ++i) out[static_cast<size_t>(i * num_nodes + i)] = 0;

  if (method == SPDMethod::kFloydWarshall) {
    for (const auto& [u, v] : edges) {
      if (u < 0 || v < 0 || u >= num_nodes || v >= num_nodes) continue;
      out[static_cast<size_t>(u * num_nodes + v)] = 1;
      out[static_cast<size_t>(v * num_nodes + u)] = 1;
    }
    for (int64_t k = 0; k < num_nodes; ++k) {
      for (int64_t i = 0; i < num_nodes; ++i) {
        for (int64_t j = 0; j < num_nodes; ++j) {
          int64_t cand = out[static_cast<size_t>(i * num_nodes + k)] + out[static_cast<size_t>(k * num_nodes + j)];
          if (cand < out[static_cast<size_t>(i * num_nodes + j)]) {
            out[static_cast<size_t>(i * num_nodes + j)] = std::min(cand, inf);
          }
        }
      }
    }
    for (auto& v : out) if (v > max_spd) v = inf;
    return out;
  }

  auto adj = BuildUndirectedAdj(edges, num_nodes);
  for (auto& nbrs : adj) std::sort(nbrs.begin(), nbrs.end());
  for (int64_t s = 0; s < num_nodes; ++s) {
    std::vector<int64_t> dist(static_cast<size_t>(num_nodes), -1);
    std::queue<int64_t> q;
    dist[static_cast<size_t>(s)] = 0;
    q.push(s);
    while (!q.empty()) {
      int64_t u = q.front();
      q.pop();
      if (dist[static_cast<size_t>(u)] >= max_spd) continue;
      for (int64_t v : adj[static_cast<size_t>(u)]) {
        if (dist[static_cast<size_t>(v)] != -1) continue;
        dist[static_cast<size_t>(v)] = dist[static_cast<size_t>(u)] + 1;
        q.push(v);
      }
    }
    for (int64_t t = 0; t < num_nodes; ++t) {
      int64_t d = dist[static_cast<size_t>(t)];
      out[static_cast<size_t>(s * num_nodes + t)] = (d < 0 || d > max_spd) ? inf : d;
    }
  }
  return out;
}

SimpleGraphCPU SimpleNodeSubgraph(const SimpleGraphCPU& graph, const std::vector<int64_t>& nodes) {
  SimpleGraphCPU out;
  out.num_nodes = static_cast<int64_t>(nodes.size());
  if (!graph.feat_id.empty()) out.feat_id.resize(nodes.size());
  out.nid.resize(nodes.size());

  std::vector<int64_t> node_map(static_cast<size_t>(graph.num_nodes), -1);
  for (size_t i = 0; i < nodes.size(); ++i) {
    const int64_t n = nodes[i];
    node_map[static_cast<size_t>(n)] = static_cast<int64_t>(i);
    if (!graph.feat_id.empty()) out.feat_id[i] = graph.feat_id[static_cast<size_t>(n)];
    if (!graph.nid.empty()) {
      out.nid[i] = graph.nid[static_cast<size_t>(n)];
    } else {
      out.nid[i] = n;
    }
  }

  for (const auto& [u, v] : graph.edges) {
    if (u < 0 || v < 0 || u >= graph.num_nodes || v >= graph.num_nodes) continue;
    const int64_t mu = node_map[static_cast<size_t>(u)];
    const int64_t mv = node_map[static_cast<size_t>(v)];
    if (mu >= 0 && mv >= 0) out.edges.push_back({mu, mv});
  }
  return out;
}

std::vector<std::pair<int64_t, int64_t>> Graph2PathV2Pure(const SimpleGraphCPU& graph,
                                                           bool deterministic,
                                                           uint64_t seed) {
  auto adj = BuildUndirectedAdj(graph.edges, graph.num_nodes);
  auto comps = ConnectedComponents(adj);
  if (comps.empty()) return {};

  std::mt19937_64 rng(seed);
  if (!deterministic) {
    std::shuffle(comps.begin(), comps.end(), rng);
  }

  std::vector<std::pair<int64_t, int64_t>> path;
  auto comp_edges = [&](const std::vector<int64_t>& nodes) {
    std::vector<int8_t> in_comp(static_cast<size_t>(graph.num_nodes), 0);
    for (int64_t n : nodes) in_comp[static_cast<size_t>(n)] = 1;
    std::vector<std::pair<int64_t, int64_t>> e;
    for (const auto& [u, v] : graph.edges) {
      if (u >= 0 && v >= 0 && u < graph.num_nodes && v < graph.num_nodes &&
          in_comp[static_cast<size_t>(u)] && in_comp[static_cast<size_t>(v)]) {
        e.push_back({u, v});
      }
    }
    return e;
  };

  auto first_edges = comp_edges(comps[0]);
  path = ConnectedGraph2Path(comps[0], first_edges, deterministic, &rng);

  int64_t prev_connect_node = path.empty() ? comps[0][0] : path.back().second;
  for (size_t i = 1; i < comps.size(); ++i) {
    auto e = comp_edges(comps[i]);
    auto spath = ConnectedGraph2Path(comps[i], e, deterministic, &rng);
    int64_t curr_connect_node = spath.empty() ? comps[i][0] : spath[0].first;
    path.push_back({prev_connect_node, curr_connect_node});
    path.insert(path.end(), spath.begin(), spath.end());
    if (!path.empty()) prev_connect_node = path.back().second;
  }
  return path;
}

}  // namespace neugn::preprocess
