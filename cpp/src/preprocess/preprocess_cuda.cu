#include "preprocess_cuda.cuh"

#include <algorithm>
#include <vector>

namespace neugn::preprocess {
namespace {

__global__ void RWSEKernel(const int64_t* src,
                           const int64_t* dst,
                           int64_t num_edges,
                           int64_t num_nodes,
                           int64_t k_steps,
                           float* out) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  if (num_nodes <= 0 || k_steps <= 0) return;

  extern __shared__ float smem[];
  float* adj = smem;                              // [N*N]
  float* P = adj + num_nodes * num_nodes;         // [N*N]
  float* Pk = P + num_nodes * num_nodes;          // [N*N]
  float* tmp = Pk + num_nodes * num_nodes;        // [N*N]
  float* deg = tmp + num_nodes * num_nodes;       // [N]

  const int64_t nn = num_nodes * num_nodes;
  for (int64_t i = 0; i < nn; ++i) adj[i] = 0.0f;
  for (int64_t i = 0; i < num_nodes; ++i) deg[i] = 0.0f;

  for (int64_t e = 0; e < num_edges; ++e) {
    int64_t u = src[e];
    int64_t v = dst[e];
    if (u < 0 || v < 0 || u >= num_nodes || v >= num_nodes) continue;
    adj[u * num_nodes + v] = 1.0f;
    adj[v * num_nodes + u] = 1.0f;
  }

  for (int64_t i = 0; i < num_nodes; ++i) {
    float s = 0.0f;
    for (int64_t j = 0; j < num_nodes; ++j) s += adj[i * num_nodes + j];
    deg[i] = s;
  }

  for (int64_t i = 0; i < num_nodes; ++i) {
    for (int64_t j = 0; j < num_nodes; ++j) {
      P[i * num_nodes + j] = (deg[i] > 0.0f) ? (adj[i * num_nodes + j] / deg[i]) : 0.0f;
      Pk[i * num_nodes + j] = P[i * num_nodes + j];
    }
  }

  for (int64_t i = 0; i < num_nodes; ++i) out[i * k_steps + 0] = Pk[i * num_nodes + i];

  for (int64_t step = 1; step < k_steps; ++step) {
    for (int64_t i = 0; i < nn; ++i) tmp[i] = 0.0f;
    for (int64_t i = 0; i < num_nodes; ++i) {
      for (int64_t k = 0; k < num_nodes; ++k) {
        float a = Pk[i * num_nodes + k];
        if (a == 0.0f) continue;
        for (int64_t j = 0; j < num_nodes; ++j) {
          tmp[i * num_nodes + j] += a * P[k * num_nodes + j];
        }
      }
    }
    for (int64_t i = 0; i < nn; ++i) Pk[i] = tmp[i];
    for (int64_t i = 0; i < num_nodes; ++i) out[i * k_steps + step] = Pk[i * num_nodes + i];
  }
}

__global__ void SPDKernel(const int64_t* src,
                          const int64_t* dst,
                          int64_t num_edges,
                          int64_t num_nodes,
                          int64_t max_spd,
                          int64_t* out) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  const int64_t inf = max_spd + 1;
  for (int64_t i = 0; i < num_nodes * num_nodes; ++i) out[i] = inf;
  for (int64_t i = 0; i < num_nodes; ++i) out[i * num_nodes + i] = 0;

  extern __shared__ int64_t smem[];
  int64_t* q = smem;  // [N]
  int64_t* dist = q + num_nodes;  // [N]

  for (int64_t s = 0; s < num_nodes; ++s) {
    for (int64_t i = 0; i < num_nodes; ++i) dist[i] = -1;
    int64_t head = 0, tail = 0;
    dist[s] = 0;
    q[tail++] = s;
    while (head < tail) {
      int64_t u = q[head++];
      if (dist[u] >= max_spd) continue;
      for (int64_t e = 0; e < num_edges; ++e) {
        int64_t a = src[e], b = dst[e];
        int64_t v = -1;
        if (a == u) v = b;
        else if (b == u) v = a;
        if (v < 0 || v >= num_nodes) continue;
        if (dist[v] == -1) {
          dist[v] = dist[u] + 1;
          q[tail++] = v;
        }
      }
    }
    for (int64_t t = 0; t < num_nodes; ++t) {
      int64_t d = dist[t];
      out[s * num_nodes + t] = (d < 0 || d > max_spd) ? inf : d;
    }
  }
}

__global__ void BuildNodeMapKernel(const int64_t* nodes,
                                   int64_t num_sub_nodes,
                                   int64_t* node_map,
                                   int64_t num_nodes,
                                   const int64_t* feat_id,
                                   const int64_t* nid,
                                   int64_t* out_feat,
                                   int64_t* out_nid) {
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < num_nodes) node_map[i] = -1;
  if (i >= num_sub_nodes) return;
  int64_t n = nodes[i];
  node_map[n] = i;
  if (out_feat && feat_id) out_feat[i] = feat_id[n];
  if (out_nid) out_nid[i] = nid ? nid[n] : n;
}

__global__ void FilterEdgesKernel(const int64_t* src,
                                  const int64_t* dst,
                                  int64_t num_edges,
                                  const int64_t* node_map,
                                  int64_t* out_src,
                                  int64_t* out_dst,
                                  int64_t* out_edge_count) {
  int64_t e = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (e >= num_edges) return;
  int64_t u = src[e], v = dst[e];
  int64_t mu = node_map[u], mv = node_map[v];
  if (mu >= 0 && mv >= 0) {
    int64_t idx = atomicAdd(out_edge_count, static_cast<int64_t>(1));
    out_src[idx] = mu;
    out_dst[idx] = mv;
  }
}

__global__ void Graph2PathDetKernel(const int64_t* src,
                                    const int64_t* dst,
                                    int64_t num_edges,
                                    int64_t num_nodes,
                                    int64_t* path_src,
                                    int64_t* path_dst,
                                    int64_t* path_len) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  *path_len = 0;
  if (num_nodes <= 0) return;

  extern __shared__ int64_t smem[];
  int64_t* comp = smem;                      // [N]
  int64_t* queue = comp + num_nodes;         // [N]
  int64_t* visited_edge = queue + num_nodes; // [E]

  for (int64_t i = 0; i < num_nodes; ++i) comp[i] = -1;
  for (int64_t e = 0; e < num_edges; ++e) visited_edge[e] = 0;

  int64_t comp_id = 0;
  for (int64_t s = 0; s < num_nodes; ++s) {
    if (comp[s] != -1) continue;
    int64_t head = 0, tail = 0;
    queue[tail++] = s;
    comp[s] = comp_id;
    while (head < tail) {
      int64_t u = queue[head++];
      for (int64_t e = 0; e < num_edges; ++e) {
        int64_t a = src[e], b = dst[e];
        int64_t v = -1;
        if (a == u) v = b;
        else if (b == u) v = a;
        if (v < 0 || v >= num_nodes) continue;
        if (comp[v] == -1) {
          comp[v] = comp_id;
          queue[tail++] = v;
        }
      }
    }
    ++comp_id;
  }

  int64_t prev_tail_node = -1;
  for (int64_t cid = 0; cid < comp_id; ++cid) {
    int64_t start = -1;
    for (int64_t n = 0; n < num_nodes; ++n) {
      if (comp[n] == cid) {
        start = n;
        break;
      }
    }
    if (start < 0) continue;

    if (prev_tail_node >= 0) {
      path_src[*path_len] = prev_tail_node;
      path_dst[*path_len] = start;
      (*path_len)++;
    }

    int64_t curr = start;
    bool progressed = true;
    while (progressed) {
      progressed = false;
      int64_t best_e = -1;
      int64_t best_v = -1;
      for (int64_t e = 0; e < num_edges; ++e) {
        if (visited_edge[e]) continue;
        int64_t a = src[e], b = dst[e];
        int64_t v = -1;
        if (a == curr) v = b;
        else if (b == curr) v = a;
        if (v < 0) continue;
        if (comp[v] != cid) continue;
        if (best_v == -1 || v < best_v) {
          best_v = v;
          best_e = e;
        }
      }
      if (best_e >= 0) {
        visited_edge[best_e] = 1;
        path_src[*path_len] = curr;
        path_dst[*path_len] = best_v;
        (*path_len)++;
        curr = best_v;
        progressed = true;
      }
    }

    prev_tail_node = curr;
  }
}

}  // namespace

void ComputeRWSECUDA(const int64_t* d_src,
                     const int64_t* d_dst,
                     int64_t num_edges,
                     int64_t num_nodes,
                     int64_t k_steps,
                     float* d_out,
                     cudaStream_t stream) {
  size_t bytes = static_cast<size_t>((4 * num_nodes * num_nodes + num_nodes) * sizeof(float));
  RWSEKernel<<<1, 1, bytes, stream>>>(d_src, d_dst, num_edges, num_nodes, k_steps, d_out);
}

void ComputeSubgraphSPDCUDA(const int64_t* d_src,
                            const int64_t* d_dst,
                            int64_t num_edges,
                            int64_t num_nodes,
                            int64_t max_spd,
                            int64_t* d_out_spd,
                            cudaStream_t stream) {
  size_t bytes = static_cast<size_t>((2 * num_nodes) * sizeof(int64_t));
  SPDKernel<<<1, 1, bytes, stream>>>(d_src, d_dst, num_edges, num_nodes, max_spd, d_out_spd);
}

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
                            cudaStream_t stream) {
  int64_t* d_node_map = nullptr;
  cudaMalloc(&d_node_map, sizeof(int64_t) * num_nodes);
  cudaMemsetAsync(d_out_edge_count, 0, sizeof(int64_t), stream);

  int threads = 256;
  int blocks_map = static_cast<int>((std::max(num_nodes, num_sub_nodes) + threads - 1) / threads);
  BuildNodeMapKernel<<<blocks_map, threads, 0, stream>>>(
      d_nodes, num_sub_nodes, d_node_map, num_nodes, d_feat_id, d_nid, d_out_feat, d_out_nid);

  int blocks_edge = static_cast<int>((num_edges + threads - 1) / threads);
  FilterEdgesKernel<<<blocks_edge, threads, 0, stream>>>(
      d_src, d_dst, num_edges, d_node_map, d_out_src, d_out_dst, d_out_edge_count);
  cudaFree(d_node_map);
}

void Graph2PathV2PureDeterministicCUDA(const int64_t* d_src,
                                       const int64_t* d_dst,
                                       int64_t num_edges,
                                       int64_t num_nodes,
                                       int64_t* d_path_src,
                                       int64_t* d_path_dst,
                                       int64_t* d_path_len,
                                       cudaStream_t stream) {
  size_t bytes = static_cast<size_t>((2 * num_nodes + num_edges) * sizeof(int64_t));
  Graph2PathDetKernel<<<1, 1, bytes, stream>>>(
      d_src, d_dst, num_edges, num_nodes, d_path_src, d_path_dst, d_path_len);
}

}  // namespace neugn::preprocess
