#pragma once

#include <cstdint>
#include <vector>

#include "encoder_cuda_kernels.cuh"

namespace neugn::encoder {

struct LinearParam {
  const float* weight = nullptr;  // [out_dim, in_dim]
  const float* bias = nullptr;    // [out_dim]
  int64_t in_dim = 0;
  int64_t out_dim = 0;
};

struct GINMLPParam {
  LinearParam lin1;
  BNInferenceParam bn1;
  LinearParam lin2;
};

struct ConvLayerParam {
  // one of (GIN / GCN)
  bool use_gin = true;
  float gin_eps = 0.0f;
  GINMLPParam gin_mlp;
  LinearParam gcn_linear;
  BNInferenceParam post_bn;
};

struct EncoderWeights {
  LinearParam value_projection;
  const float* degree_embedding_table = nullptr;  // [1001, feat_dim]
  int64_t degree_max = 1000;

  bool enable_rwse = false;
  LinearParam rwse_projection;

  std::vector<ConvLayerParam> conv_layers;
};

struct EncoderIO {
  // graph
  const int64_t* coo_src = nullptr;
  const int64_t* coo_dst = nullptr;
  int64_t num_nodes = 0;
  int64_t num_edges = 0;
  const int64_t* batch_idx = nullptr;
  int64_t num_graphs = 1;

  // features
  const int64_t* feat_id = nullptr;
  const float* rwse = nullptr;

  // outputs (device pointers allocated by caller)
  float* graph_feature_out = nullptr;     // [B, dim]
  float* all_node_feature_out = nullptr;  // [N, dim]
};

class GraphEncoderCUDA {
 public:
  explicit GraphEncoderCUDA(EncoderWeights weights);

  // Runs: value_projection + degree_embedding (+ optional rwse_projection)
  // -> conv layers (BN inference + ReLU) -> global_mean_pool.
  // NOTE: all pointers must be CUDA device pointers.
  void Forward(const EncoderIO& io) const;

 private:
  EncoderWeights w_;
};

}  // namespace neugn::encoder
