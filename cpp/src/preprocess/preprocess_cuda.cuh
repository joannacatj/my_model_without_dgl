#pragma once

#include <cstdint>

#include <cuda_runtime.h>

namespace neugn::preprocess {

// GPU version of compute_rwse (dense/random-walk, result-first implementation).
// d_out: [num_nodes, k_steps] row-major.
void ComputeRWSECUDA(const int64_t* d_src,
                     const int64_t* d_dst,
                     int64_t num_edges,
                     int64_t num_nodes,
                     int64_t k_steps,
                     float* d_out,
                     cudaStream_t stream = 0);

// GPU version of compute_subgraph_spd (BFS all-pairs, clipped to max_spd+1).
// d_out_spd: [num_nodes, num_nodes] row-major.
void ComputeSubgraphSPDCUDA(const int64_t* d_src,
                            const int64_t* d_dst,
                            int64_t num_edges,
                            int64_t num_nodes,
                            int64_t max_spd,
                            int64_t* d_out_spd,
                            cudaStream_t stream = 0);

// GPU version of simple_node_subgraph.
// nodes[] keeps python-equivalent order for local reindex.
// outputs:
//   d_out_src/d_out_dst: mapped edge list (capacity >= num_edges)
//   d_out_edge_count: produced mapped edge count
//   d_out_feat/d_out_nid: optional, size=num_sub_nodes
void SimpleNodeSubgraphCUDA(const int64_t* d_src,
                            const int64_t* d_dst,
                            int64_t num_edges,
                            int64_t num_nodes,
                            const int64_t* d_nodes,
                            int64_t num_sub_nodes,
                            const int64_t* d_feat_id,
                            const int64_t* d_nid,
                            int64_t* d_out_src,
                            int64_t* d_out_dst,
                            int64_t* d_out_edge_count,
                            int64_t* d_out_feat,
                            int64_t* d_out_nid,
                            cudaStream_t stream = 0);

// Deterministic GPU path generation approximation for graph2path_v2_pure.
// Result-first baseline: single-thread graph walk covering all edges in components,
// with jump edges between components.
// d_path_src/d_path_dst capacity must be >= num_edges + num_nodes.
void Graph2PathV2PureDeterministicCUDA(const int64_t* d_src,
                                       const int64_t* d_dst,
                                       int64_t num_edges,
                                       int64_t num_nodes,
                                       int64_t* d_path_src,
                                       int64_t* d_path_dst,
                                       int64_t* d_path_len,
                                       cudaStream_t stream = 0);

}  // namespace neugn::preprocess
