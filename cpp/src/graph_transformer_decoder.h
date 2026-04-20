#pragma once

#include <string>

#include "tensor_types.h"

namespace neugn {

struct GraphTransformerDecoderConfig {
  int64_t dim = 0;
  int64_t n_layers = 0;
  int64_t n_heads = 0;
  int64_t max_spd = 0;
  double norm_eps = 1e-5;
};

class GraphTransformerDecoder {
 public:
  explicit GraphTransformerDecoder(GraphTransformerDecoderConfig config);

  bool LoadWeights(const std::string& weights_bin, const std::string& manifest_json);

  // Align with NeuGN/model.py::Transformer.forward
  DecoderOutput Forward(const DecoderInput& input) const;

 private:
  GraphTransformerDecoderConfig config_;
};

}  // namespace neugn
