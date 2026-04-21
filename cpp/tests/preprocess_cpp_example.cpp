#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

#include "../src/preprocess/preprocess_utils.h"

int main() {
  using namespace neugn::preprocess;

  SimpleGraphCPU g;
  g.num_nodes = 5;
  g.edges = {{0, 1}, {1, 2}, {2, 3}, {3, 4}, {1, 3}};
  g.feat_id = {10, 11, 12, 13, 14};
  g.nid = {100, 101, 102, 103, 104};

  auto rwse = ComputeRWSE(g.edges, g.num_nodes, 4);
  auto spd = ComputeSubgraphSPD(g.edges, g.num_nodes, 3, SPDMethod::kBFS);
  auto spd_fw = ComputeSubgraphSPD(g.edges, g.num_nodes, 3, SPDMethod::kFloydWarshall);

  auto sg = SimpleNodeSubgraph(g, {1, 3, 4});
  auto p1 = Graph2PathV2Pure(g, true, 123);
  auto p2 = Graph2PathV2Pure(g, true, 456);

  bool same_det = (p1 == p2);
  bool spd_same = (spd == spd_fw);

  std::cout << "rwse shape=(" << g.num_nodes << ",4)\n";
  std::cout << "rwse first-row=";
  for (int i = 0; i < 4; ++i) std::cout << " " << rwse[i];
  std::cout << "\n";

  std::cout << "spd[0,*]=";
  for (int64_t j = 0; j < g.num_nodes; ++j) {
    std::cout << " " << spd[static_cast<size_t>(j)];
  }
  std::cout << "\n";

  std::cout << "subgraph num_nodes=" << sg.num_nodes << " edges=" << sg.edges.size() << "\n";
  std::cout << "deterministic_path_same=" << (same_det ? "true" : "false") << "\n";
  std::cout << "spd_bfs_eq_floyd=" << (spd_same ? "true" : "false") << "\n";

  if (!same_det || !spd_same) return 1;
  return 0;
}
