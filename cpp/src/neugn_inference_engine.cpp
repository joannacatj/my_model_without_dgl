#include "neugn_inference_engine.h"

namespace neugn {

NeuGNInferenceEngine::NeuGNInferenceEngine(EngineConfig config)
    : config_(std::move(config)), encoder_(config_.encoder), decoder_(config_.decoder) {}

bool NeuGNInferenceEngine::LoadWeights(const std::string& weights_bin, const std::string& manifest_json) {
  const bool ok_encoder = encoder_.LoadWeights(weights_bin, manifest_json);
  const bool ok_decoder = decoder_.LoadWeights(weights_bin, manifest_json);
  return ok_encoder && ok_decoder;
}

bool NeuGNInferenceEngine::SetQueryGraph(const GraphBatchView& /*query_graph*/) {
  // TODO: compute/query RWSE + SPD + query path, cache current_query_feat_.
  return true;
}

ScoreMap NeuGNInferenceEngine::PredictScores(
    int64_t /*target_query_node*/,
    const std::unordered_map<int64_t, int64_t>& /*partial_mapping*/,
    const std::vector<int64_t>& /*candidates*/) const {
  // TODO: build decoder inputs from cached query states and return cosine scores.
  return {};
}

}  // namespace neugn
