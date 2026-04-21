#pragma once

#include <cstdint>
#include <vector>

namespace neugn::decoder {

struct LinearParam {
  const float* weight = nullptr;  // [out_dim, in_dim]
  const float* bias = nullptr;    // [out_dim] or nullptr
  int64_t in_dim = 0;
  int64_t out_dim = 0;
};

struct RMSNormParam {
  const float* weight = nullptr;  // [dim]
  float eps = 1e-5f;
};

struct DecoderLayerParam {
  LinearParam wq;
  LinearParam wk;
  LinearParam wv;
  LinearParam wo;

  LinearParam w1;
  LinearParam w2;
  LinearParam w3;

  RMSNormParam attention_norm;
  RMSNormParam ffn_norm;
};

struct DecoderWeights {
  LinearParam input_projection;

  const float* sos_token = nullptr;         // [1,1,dim]
  const float* node_embeddings = nullptr;   // [sub_node_id_size, dim]
  const float* pos_embeddings = nullptr;    // [1, pos_size, dim]
  const float* type_embeddings = nullptr;   // [2, dim]
  const float* spd_bias_embedding = nullptr; // [max_spd+2, n_heads]

  std::vector<DecoderLayerParam> layers;
  RMSNormParam final_norm;

  LinearParam out1;  // output.0
  LinearParam out2;  // output.2

  int64_t dim = 0;
  int64_t n_layers = 0;
  int64_t n_heads = 0;
  int64_t n_kv_heads = 0;
  int64_t max_spd = 20;
  int64_t sub_node_id_size = 64;
  int64_t pos_size = 1024;
};

struct DecoderIO {
  const float* graph_features = nullptr;   // [B,1,dim]
  const float* input_features = nullptr;   // [B,seq,feat_dim]
  const int64_t* subnode_ids = nullptr;    // [B,seq]
  const int64_t* token_mask_len = nullptr; // [B]
  const int64_t* spd_indices = nullptr;    // [B,seq,seq] optional

  int64_t batch = 1;
  int64_t seq = 0;
  int64_t feat_dim = 0;

  float* output = nullptr; // [B,seq+1,feat_dim]
};

class GraphTransformerDecoderCUDA {
 public:
  explicit GraphTransformerDecoderCUDA(DecoderWeights weights);
  void Forward(const DecoderIO& io) const;

 private:
  DecoderWeights w_;
};

}  // namespace neugn::decoder
