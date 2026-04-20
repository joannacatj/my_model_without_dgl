#pragma once

#include <string>

#include "tensor_types.h"

namespace neugn {

struct GraphEncoderConfig {
  std::string encoder_name;  // "gin" / "gcn"
  int64_t encoder_layers = 0;
  int64_t graph_feature_dim = 0;
  int64_t hidden_dim = 0;
  int64_t fixed_input_dim = 0;
  int64_t rwse_dim = 0;
};

class GraphEncoder {
 public:
  explicit GraphEncoder(GraphEncoderConfig config);

  // Load encoded tensors from exported manifest/name-indexed storage.
  bool LoadWeights(const std::string& weights_bin, const std::string& manifest_json);

  // Align with NeuGN/mpnn_encoder.py::GNN.forward
  EncoderOutput Forward(const GraphBatchView& graph) const;

 private:
  GraphEncoderConfig config_;
};

}  // namespace neugn
