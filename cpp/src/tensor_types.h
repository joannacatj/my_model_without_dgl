#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace neugn {

enum class DType {
  kFloat16,
  kFloat32,
  kInt64,
};

enum class DeviceType {
  kCUDA,
  kCPU,
};

struct TensorView {
  void* data = nullptr;
  std::vector<int64_t> shape;
  DType dtype = DType::kFloat32;
  DeviceType device = DeviceType::kCUDA;
};

struct GraphBatchView {
  TensorView feat_id;               // [N], int64
  TensorView edge_index;            // [2, E], int64
  TensorView batch_index;           // [N], int64 (optional when B=1)
  TensorView rwse;                  // [N, rwse_dim], optional
};

struct EncoderOutput {
  TensorView graph_feature;         // [B, Dim]
  TensorView node_feature;          // [N, Dim]
};

struct DecoderInput {
  TensorView graph_features;        // [B, 1, Dim]
  TensorView input_features;        // [B, Seq, FeatDim]
  TensorView subnode_ids;           // [B, Seq], int64
  TensorView token_mask_len;        // [B], int64
  TensorView spd_indices;           // [B, Seq, Seq], int64 (optional)
};

struct DecoderOutput {
  TensorView sequence_output;       // [B, Seq+1, FeatDim]
};

using ScoreMap = std::unordered_map<int64_t, float>;

}  // namespace neugn
