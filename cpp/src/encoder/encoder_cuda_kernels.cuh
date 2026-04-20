#pragma once

#include <cstdint>
#include <vector>

namespace neugn::encoder {

struct BNInferenceParam {
  const float* gamma = nullptr;
  const float* beta = nullptr;
  const float* running_mean = nullptr;
  const float* running_var = nullptr;
  float eps = 1e-5f;
};

// Host-side helper: COO(2, E) -> CSR(row_ptr[N+1], col[E]).
void BuildCSRFromCOO(int64_t num_nodes,
                     const int64_t* coo_src,
                     const int64_t* coo_dst,
                     int64_t num_edges,
                     std::vector<int64_t>* row_ptr,
                     std::vector<int64_t>* col_idx);

// Device-side COO -> CSR using CUB sort + run-length encode + scan.
// Caller owns d_row_ptr/d_col_idx allocations.
void BuildCSRFromCOOCUDA(int64_t num_nodes,
                         const int64_t* d_coo_src,
                         const int64_t* d_coo_dst,
                         int64_t num_edges,
                         int64_t* d_row_ptr,   // [num_nodes + 1]
                         int64_t* d_col_idx);  // [num_edges]

// value_projection(one_hot(feat_id)) equivalent.
void ValueProjectionForwardCUDA(const int64_t* feat_id,
                                int64_t num_nodes,
                                int64_t fixed_input_dim,
                                const float* weight,   // [out_dim, fixed_input_dim]
                                const float* bias,     // [out_dim] or nullptr
                                int64_t out_dim,
                                float* out);            // [N, out_dim]

void DegreeEmbeddingForwardCUDA(const int64_t* in_degree,
                                int64_t num_nodes,
                                int64_t max_degree,
                                const float* emb_table,  // [max_degree+1, feat_dim]
                                int64_t feat_dim,
                                float* out);             // [N, feat_dim]

void RWSEProjectionForwardCUDA(const float* rwse,        // [N, rwse_dim]
                               int64_t num_nodes,
                               int64_t rwse_dim,
                               const float* weight,      // [feat_dim, rwse_dim]
                               const float* bias,        // [feat_dim] or nullptr
                               int64_t feat_dim,
                               float* out);              // [N, feat_dim]

// out = MLP((1+eps)*x + Sum(neighbor)) ; this API computes pre-MLP aggregation term.
void PureGINAggregateCUDA(const int64_t* csr_row_ptr,
                          const int64_t* csr_col_idx,
                          int64_t num_nodes,
                          int64_t feat_dim,
                          const float* x,
                          float eps,
                          float* out);  // [N, feat_dim]

// out = D^-1/2 (A+I) D^-1/2 XW ; this API computes normalized aggregation before linear W.
void PureGraphConvAggregateCUDA(const int64_t* csr_row_ptr,
                                const int64_t* csr_col_idx,
                                int64_t num_nodes,
                                int64_t feat_dim,
                                const float* x,
                                float* out);  // [N, feat_dim]

void GlobalMeanPoolCUDA(const float* node_x,         // [N, feat_dim]
                        const int64_t* batch_idx,    // [N]
                        int64_t num_nodes,
                        int64_t num_graphs,
                        int64_t feat_dim,
                        float* out_graph,            // [B, feat_dim]
                        float* out_count);           // [B]

void BatchNormInferenceReLUCUDA(const float* x,
                                int64_t rows,
                                int64_t feat_dim,
                                BNInferenceParam bn,
                                float* out);

}  // namespace neugn::encoder
