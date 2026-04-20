#include "graph_encoder.h"

namespace neugn {

GraphEncoder::GraphEncoder(GraphEncoderConfig config) : config_(std::move(config)) {}

bool GraphEncoder::LoadWeights(const std::string& /*weights_bin*/, const std::string& /*manifest_json*/) {
  // TODO: parse manifest and bind raw slices to encoder tensors.
  return true;
}

EncoderOutput GraphEncoder::Forward(const GraphBatchView& /*graph*/) const {
  // TODO: implement GIN/GCN encoder forward.
  return {};
}

}  // namespace neugn
