#pragma once

#include <cstdint>
#include <utility>
#include <vector>

namespace neugn::preprocess {

struct SimpleGraphCPU {
  int64_t num_nodes = 0;
  std::vector<std::pair<int64_t, int64_t>> edges;  // (src, dst)
  std::vector<int64_t> feat_id;                    // optional
  std::vector<int64_t> nid;                        // optional
};

enum class SPDMethod {
  kBFS,
  kFloydWarshall,
};

// Equivalent to NeuGN/utils.py::compute_rwse.
// Returns row-major [num_nodes, k_steps].
std::vector<float> ComputeRWSE(const std::vector<std::pair<int64_t, int64_t>>& edges,
                               int64_t num_nodes,
                               int64_t k_steps);

// Equivalent to NeuGN/utils.py::compute_subgraph_spd.
// Returns row-major [num_nodes, num_nodes], clipped to max_spd+1.
std::vector<int64_t> ComputeSubgraphSPD(const std::vector<std::pair<int64_t, int64_t>>& edges,
                                        int64_t num_nodes,
                                        int64_t max_spd,
                                        SPDMethod method = SPDMethod::kBFS);

// Equivalent to NeuGN/utils.py::simple_node_subgraph.
SimpleGraphCPU SimpleNodeSubgraph(const SimpleGraphCPU& graph, const std::vector<int64_t>& nodes);

// Equivalent to NeuGN/utils.py::graph2path_v2_pure with deterministic option.
// Returns path edge list.
std::vector<std::pair<int64_t, int64_t>> Graph2PathV2Pure(const SimpleGraphCPU& graph,
                                                           bool deterministic = true,
                                                           uint64_t seed = 0);

}  // namespace neugn::preprocess
