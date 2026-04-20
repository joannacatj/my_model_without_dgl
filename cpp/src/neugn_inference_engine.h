#pragma once

#include <unordered_map>
#include <vector>

#include "graph_encoder.h"
#include "graph_transformer_decoder.h"

namespace neugn {

struct EngineConfig {
  GraphEncoderConfig encoder;
  GraphTransformerDecoderConfig decoder;
};

class NeuGNInferenceEngine {
 public:
  explicit NeuGNInferenceEngine(EngineConfig config);

  bool LoadWeights(const std::string& weights_bin, const std::string& manifest_json);

  // Align with demo.py::NeuGNInferenceWrapper.set_query_graph
  bool SetQueryGraph(const GraphBatchView& query_graph);

  // Align with demo.py::NeuGNInferenceWrapper.predict_scores
  ScoreMap PredictScores(
      int64_t target_query_node,
      const std::unordered_map<int64_t, int64_t>& partial_mapping,
      const std::vector<int64_t>& candidates) const;

 private:
  EngineConfig config_;
  GraphEncoder encoder_;
  GraphTransformerDecoder decoder_;

  // Cached query-side states from SetQueryGraph().
  TensorView current_query_feat_;    // [1, 1, Dim]
  TensorView current_query_spd_;     // [Q, Q]
  std::vector<int64_t> current_query_path_;
  std::unordered_map<int64_t, std::vector<int64_t>> query_node_to_path_indices_;
};

}  // namespace neugn
