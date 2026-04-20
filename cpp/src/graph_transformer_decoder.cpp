#include "graph_transformer_decoder.h"

namespace neugn {

GraphTransformerDecoder::GraphTransformerDecoder(GraphTransformerDecoderConfig config)
    : config_(std::move(config)) {}

bool GraphTransformerDecoder::LoadWeights(const std::string& /*weights_bin*/, const std::string& /*manifest_json*/) {
  // TODO: parse manifest and bind raw slices to decoder tensors.
  return true;
}

DecoderOutput GraphTransformerDecoder::Forward(const DecoderInput& /*input*/) const {
  // TODO: implement Transformer decoder forward.
  return {};
}

}  // namespace neugn
