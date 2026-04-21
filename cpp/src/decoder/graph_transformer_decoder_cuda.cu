#include "graph_transformer_decoder_cuda.h"

#include <cuda_runtime.h>

#include <cmath>
#include <stdexcept>

namespace neugn::decoder {
namespace {

inline int BlocksFor(int64_t n) { return static_cast<int>((n + 255) / 256); }

__global__ void LinearKernel(const float* x, int64_t rows, int64_t in_dim,
                             const float* w, const float* b, int64_t out_dim, float* out) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = rows * out_dim;
  if (idx >= total) return;
  int64_t r = idx / out_dim;
  int64_t d = idx % out_dim;
  float acc = b ? b[d] : 0.0f;
  for (int64_t k = 0; k < in_dim; ++k) acc += x[r * in_dim + k] * w[d * in_dim + k];
  out[idx] = acc;
}

__global__ void AddKernel(const float* a, const float* b, int64_t total, float* out) {
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < total) out[i] = a[i] + b[i];
}

__global__ void AddInplaceKernel(float* a, const float* b, int64_t total) {
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < total) a[i] += b[i];
}

__global__ void RMSNormKernel(const float* x, const float* w, int64_t rows, int64_t dim, float eps, float* out) {
  int64_t r = blockIdx.x;
  int64_t d = threadIdx.x;
  if (r >= rows || d >= dim) return;

  __shared__ float mean_sq;
  if (d == 0) {
    float s = 0.0f;
    for (int64_t k = 0; k < dim; ++k) {
      float v = x[r * dim + k];
      s += v * v;
    }
    mean_sq = s / static_cast<float>(dim);
  }
  __syncthreads();

  float v = x[r * dim + d] * rsqrtf(mean_sq + eps);
  out[r * dim + d] = v * w[d];
}

__global__ void BuildTokenMaskKernel(const int64_t* token_mask_len, int64_t batch, int64_t full_len,
                                     int64_t graph_tokens, float* mask) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = batch * full_len * full_len;
  if (idx >= total) return;
  int64_t b = idx / (full_len * full_len);
  int64_t rem = idx % (full_len * full_len);
  int64_t i = rem / full_len;
  int64_t valid = token_mask_len[b] + 1 + graph_tokens;
  mask[idx] = (i < valid) ? 0.0f : -INFINITY;
}

__global__ void BuildTokenEmbKernel(const float* input_proj, const float* sos_token,
                                    const float* node_emb, const int64_t* subnode_ids,
                                    const float* type_emb, int64_t batch, int64_t seq, int64_t dim,
                                    int64_t subnode_vocab, float* out_tokens) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = batch * (seq + 1) * dim;
  if (idx >= total) return;
  int64_t b = idx / ((seq + 1) * dim);
  int64_t rem = idx % ((seq + 1) * dim);
  int64_t t = rem / dim;
  int64_t d = rem % dim;

  float v = 0.0f;
  if (t == 0) {
    v = sos_token[d];
  } else {
    v = input_proj[(b * seq + (t - 1)) * dim + d];
  }

  int64_t sub_id = 0;
  if (t > 0) sub_id = subnode_ids[b * seq + (t - 1)];
  if (sub_id < 0) sub_id = 0;
  if (sub_id >= subnode_vocab) sub_id %= subnode_vocab;
  v += node_emb[sub_id * dim + d];

  // token type = 0
  v += type_emb[d];
  out_tokens[idx] = v;
}

__global__ void BuildFullSeqKernel(const float* graph_features, const float* type_emb,
                                   const float* tokens, int64_t batch, int64_t seq,
                                   int64_t dim, float* full) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t full_len = seq + 2;
  int64_t total = batch * full_len * dim;
  if (idx >= total) return;
  int64_t b = idx / (full_len * dim);
  int64_t rem = idx % (full_len * dim);
  int64_t t = rem / dim;
  int64_t d = rem % dim;

  if (t == 0) {
    // graph type = 1
    full[idx] = graph_features[b * dim + d] + type_emb[dim + d];
  } else {
    full[idx] = tokens[(b * (seq + 1) + (t - 1)) * dim + d];
  }
}

__global__ void AddPosKernel(const float* pos, int64_t batch, int64_t full_len, int64_t dim, float* x) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = batch * full_len * dim;
  if (idx >= total) return;
  int64_t rem = idx % (full_len * dim);
  int64_t t = rem / dim;
  int64_t d = rem % dim;
  x[idx] += pos[t * dim + d];
}

__global__ void BuildGraphBiasKernel(const int64_t* spd, const float* emb,
                                     int64_t batch, int64_t seq, int64_t max_spd, int64_t heads,
                                     float* bias) {
  // bias: [B, heads, full, full], full=seq+2
  int64_t full = seq + 2;
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = batch * heads * full * full;
  if (idx >= total) return;
  int64_t b = idx / (heads * full * full);
  int64_t rem = idx % (heads * full * full);
  int64_t h = rem / (full * full);
  int64_t rem2 = rem % (full * full);
  int64_t i = rem2 / full;
  int64_t j = rem2 % full;

  int64_t d = max_spd + 1;
  if (i >= 2 && j >= 2 && spd) {
    d = spd[(b * seq + (i - 2)) * seq + (j - 2)];
    if (d < 0) d = 0;
    if (d > max_spd + 1) d = max_spd + 1;
  } else if (i == j) {
    d = 0;
  }
  bias[idx] = emb[d * heads + h];
}

__global__ void ReshapeQKVKernel(const float* x, int64_t batch, int64_t seq, int64_t heads, int64_t head_dim, float* out) {
  // in [B,seq,heads*head_dim] -> out [B,heads,seq,head_dim]
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = batch * heads * seq * head_dim;
  if (idx >= total) return;
  int64_t b = idx / (heads * seq * head_dim);
  int64_t rem = idx % (heads * seq * head_dim);
  int64_t h = rem / (seq * head_dim);
  int64_t rem2 = rem % (seq * head_dim);
  int64_t s = rem2 / head_dim;
  int64_t d = rem2 % head_dim;
  out[idx] = x[(b * seq + s) * (heads * head_dim) + h * head_dim + d];
}

__global__ void RepeatKVKernel(const float* kv, int64_t batch, int64_t kv_heads, int64_t seq,
                               int64_t head_dim, int64_t n_rep, float* out) {
  // kv [B,kv,seq,hd] -> out [B,kv*n_rep,seq,hd]
  int64_t out_heads = kv_heads * n_rep;
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = batch * out_heads * seq * head_dim;
  if (idx >= total) return;
  int64_t b = idx / (out_heads * seq * head_dim);
  int64_t rem = idx % (out_heads * seq * head_dim);
  int64_t h = rem / (seq * head_dim);
  int64_t rem2 = rem % (seq * head_dim);
  int64_t s = rem2 / head_dim;
  int64_t d = rem2 % head_dim;
  int64_t kh = h / n_rep;
  out[idx] = kv[((b * kv_heads + kh) * seq + s) * head_dim + d];
}

__global__ void AttnScoresKernel(const float* q, const float* k,
                                 const float* mask, const float* gbias,
                                 int64_t batch, int64_t heads, int64_t seq, int64_t head_dim,
                                 float* scores) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = batch * heads * seq * seq;
  if (idx >= total) return;
  int64_t b = idx / (heads * seq * seq);
  int64_t rem = idx % (heads * seq * seq);
  int64_t h = rem / (seq * seq);
  int64_t rem2 = rem % (seq * seq);
  int64_t i = rem2 / seq;
  int64_t j = rem2 % seq;

  float s = 0.0f;
  for (int64_t d = 0; d < head_dim; ++d) {
    float qv = q[((b * heads + h) * seq + i) * head_dim + d];
    float kv = k[((b * heads + h) * seq + j) * head_dim + d];
    s += qv * kv;
  }
  s /= sqrtf(static_cast<float>(head_dim));
  if (mask) s += mask[(b * seq + i) * seq + j];
  if (gbias) s += gbias[((b * heads + h) * seq + i) * seq + j];
  scores[idx] = s;
}

__global__ void SoftmaxRowKernel(float* scores, int64_t rows, int64_t cols) {
  int64_t r = blockIdx.x;
  if (r >= rows) return;
  float mx = -INFINITY;
  for (int64_t c = 0; c < cols; ++c) mx = fmaxf(mx, scores[r * cols + c]);
  float sum = 0.0f;
  for (int64_t c = 0; c < cols; ++c) {
    float e = expf(scores[r * cols + c] - mx);
    scores[r * cols + c] = e;
    sum += e;
  }
  for (int64_t c = 0; c < cols; ++c) scores[r * cols + c] /= sum;
}

__global__ void AttnWeightedSumKernel(const float* probs, const float* v,
                                      int64_t batch, int64_t heads, int64_t seq, int64_t head_dim,
                                      float* out) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = batch * heads * seq * head_dim;
  if (idx >= total) return;
  int64_t b = idx / (heads * seq * head_dim);
  int64_t rem = idx % (heads * seq * head_dim);
  int64_t h = rem / (seq * head_dim);
  int64_t rem2 = rem % (seq * head_dim);
  int64_t i = rem2 / head_dim;
  int64_t d = rem2 % head_dim;

  float s = 0.0f;
  for (int64_t j = 0; j < seq; ++j) {
    float p = probs[((b * heads + h) * seq + i) * seq + j];
    float vv = v[((b * heads + h) * seq + j) * head_dim + d];
    s += p * vv;
  }
  out[idx] = s;
}

__global__ void MergeHeadsKernel(const float* x, int64_t batch, int64_t heads, int64_t seq, int64_t head_dim, float* out) {
  // x [B,H,S,D] -> out [B,S,H*D]
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t dim = heads * head_dim;
  int64_t total = batch * seq * dim;
  if (idx >= total) return;
  int64_t b = idx / (seq * dim);
  int64_t rem = idx % (seq * dim);
  int64_t s = rem / dim;
  int64_t d = rem % dim;
  int64_t h = d / head_dim;
  int64_t hd = d % head_dim;
  out[idx] = x[((b * heads + h) * seq + s) * head_dim + hd];
}

__global__ void SiLUMulKernel(const float* a, const float* b, int64_t total, float* out) {
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= total) return;
  float x = a[i];
  float silu = x / (1.0f + expf(-x));
  out[i] = silu * b[i];
}

__global__ void GELUKernel(const float* x, int64_t total, float* out) {
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= total) return;
  float v = x[i];
  float c = 0.044715f;
  float y = 0.5f * v * (1.0f + tanhf(0.7978845608f * (v + c * v * v * v)));
  out[i] = y;
}

}  // namespace

GraphTransformerDecoderCUDA::GraphTransformerDecoderCUDA(DecoderWeights weights) : w_(std::move(weights)) {}

void GraphTransformerDecoderCUDA::Forward(const DecoderIO& io) const {
  if (!io.output || !io.graph_features || !io.input_features || !io.subnode_ids || !io.token_mask_len) {
    throw std::invalid_argument("GraphTransformerDecoderCUDA::Forward: null input/output");
  }

  const int64_t B = io.batch;
  const int64_t Seq = io.seq;
  const int64_t Dim = w_.dim;
  const int64_t Full = Seq + 2;
  const int64_t Heads = w_.n_heads;
  const int64_t KvHeads = w_.n_kv_heads > 0 ? w_.n_kv_heads : Heads;
  const int64_t HeadDim = Dim / Heads;
  const int64_t Rep = Heads / KvHeads;

  float* input_proj = nullptr;
  cudaMalloc(&input_proj, sizeof(float) * B * Seq * Dim);
  LinearKernel<<<BlocksFor(B * Seq * Dim), 256>>>(
      io.input_features, B * Seq, io.feat_dim,
      w_.input_projection.weight, w_.input_projection.bias, Dim, input_proj);

  float* tokens = nullptr;
  cudaMalloc(&tokens, sizeof(float) * B * (Seq + 1) * Dim);
  BuildTokenEmbKernel<<<BlocksFor(B * (Seq + 1) * Dim), 256>>>(
      input_proj, w_.sos_token, w_.node_embeddings, io.subnode_ids,
      w_.type_embeddings, B, Seq, Dim,
      w_.sub_node_id_size, tokens);

  float* h = nullptr;
  cudaMalloc(&h, sizeof(float) * B * Full * Dim);
  BuildFullSeqKernel<<<BlocksFor(B * Full * Dim), 256>>>(
      io.graph_features, w_.type_embeddings, tokens, B, Seq, Dim, h);
  AddPosKernel<<<BlocksFor(B * Full * Dim), 256>>>(w_.pos_embeddings, B, Full, Dim, h);

  float* mask = nullptr;
  cudaMalloc(&mask, sizeof(float) * B * Full * Full);
  BuildTokenMaskKernel<<<BlocksFor(B * Full * Full), 256>>>(io.token_mask_len, B, Full, 1, mask);

  float* gbias = nullptr;
  cudaMalloc(&gbias, sizeof(float) * B * Heads * Full * Full);
  BuildGraphBiasKernel<<<BlocksFor(B * Heads * Full * Full), 256>>>(
      io.spd_indices, w_.spd_bias_embedding, B, Seq, w_.max_spd, Heads, gbias);

  float* tmp = nullptr;
  cudaMalloc(&tmp, sizeof(float) * B * Full * Dim);

  for (int64_t l = 0; l < w_.n_layers; ++l) {
    const auto& L = w_.layers[l];

    RMSNormKernel<<<B * Full, Dim>>>(h, L.attention_norm.weight, B * Full, Dim, L.attention_norm.eps, tmp);

    float *q = nullptr, *k = nullptr, *v = nullptr;
    cudaMalloc(&q, sizeof(float) * B * Full * Heads * HeadDim);
    cudaMalloc(&k, sizeof(float) * B * Full * KvHeads * HeadDim);
    cudaMalloc(&v, sizeof(float) * B * Full * KvHeads * HeadDim);
    LinearKernel<<<BlocksFor(B * Full * Heads * HeadDim), 256>>>(tmp, B * Full, Dim, L.wq.weight, nullptr, Heads * HeadDim, q);
    LinearKernel<<<BlocksFor(B * Full * KvHeads * HeadDim), 256>>>(tmp, B * Full, Dim, L.wk.weight, nullptr, KvHeads * HeadDim, k);
    LinearKernel<<<BlocksFor(B * Full * KvHeads * HeadDim), 256>>>(tmp, B * Full, Dim, L.wv.weight, nullptr, KvHeads * HeadDim, v);

    float *q4 = nullptr, *k4 = nullptr, *v4 = nullptr;
    cudaMalloc(&q4, sizeof(float) * B * Heads * Full * HeadDim);
    cudaMalloc(&k4, sizeof(float) * B * KvHeads * Full * HeadDim);
    cudaMalloc(&v4, sizeof(float) * B * KvHeads * Full * HeadDim);
    ReshapeQKVKernel<<<BlocksFor(B * Heads * Full * HeadDim), 256>>>(q, B, Full, Heads, HeadDim, q4);
    ReshapeQKVKernel<<<BlocksFor(B * KvHeads * Full * HeadDim), 256>>>(k, B, Full, KvHeads, HeadDim, k4);
    ReshapeQKVKernel<<<BlocksFor(B * KvHeads * Full * HeadDim), 256>>>(v, B, Full, KvHeads, HeadDim, v4);

    float *k_rep = k4, *v_rep = v4;
    if (Rep > 1) {
      cudaMalloc(&k_rep, sizeof(float) * B * Heads * Full * HeadDim);
      cudaMalloc(&v_rep, sizeof(float) * B * Heads * Full * HeadDim);
      RepeatKVKernel<<<BlocksFor(B * Heads * Full * HeadDim), 256>>>(k4, B, KvHeads, Full, HeadDim, Rep, k_rep);
      RepeatKVKernel<<<BlocksFor(B * Heads * Full * HeadDim), 256>>>(v4, B, KvHeads, Full, HeadDim, Rep, v_rep);
    }

    float* scores = nullptr;
    cudaMalloc(&scores, sizeof(float) * B * Heads * Full * Full);
    AttnScoresKernel<<<BlocksFor(B * Heads * Full * Full), 256>>>(q4, k_rep, mask, gbias, B, Heads, Full, HeadDim, scores);
    SoftmaxRowKernel<<<B * Heads * Full, 1>>>(scores, B * Heads * Full, Full);

    float* attn = nullptr;
    cudaMalloc(&attn, sizeof(float) * B * Heads * Full * HeadDim);
    AttnWeightedSumKernel<<<BlocksFor(B * Heads * Full * HeadDim), 256>>>(scores, v_rep, B, Heads, Full, HeadDim, attn);

    float* merged = nullptr;
    cudaMalloc(&merged, sizeof(float) * B * Full * Dim);
    MergeHeadsKernel<<<BlocksFor(B * Full * Dim), 256>>>(attn, B, Heads, Full, HeadDim, merged);

    float* attn_out = nullptr;
    cudaMalloc(&attn_out, sizeof(float) * B * Full * Dim);
    LinearKernel<<<BlocksFor(B * Full * Dim), 256>>>(merged, B * Full, Dim, L.wo.weight, nullptr, Dim, attn_out);
    AddInplaceKernel<<<BlocksFor(B * Full * Dim), 256>>>(h, attn_out, B * Full * Dim);

    RMSNormKernel<<<B * Full, Dim>>>(h, L.ffn_norm.weight, B * Full, Dim, L.ffn_norm.eps, tmp);

    int64_t hid = L.w1.out_dim;
    float *ff1 = nullptr, *ff3 = nullptr, *ffg = nullptr, *ff2 = nullptr;
    cudaMalloc(&ff1, sizeof(float) * B * Full * hid);
    cudaMalloc(&ff3, sizeof(float) * B * Full * hid);
    cudaMalloc(&ffg, sizeof(float) * B * Full * hid);
    cudaMalloc(&ff2, sizeof(float) * B * Full * Dim);
    LinearKernel<<<BlocksFor(B * Full * hid), 256>>>(tmp, B * Full, Dim, L.w1.weight, nullptr, hid, ff1);
    LinearKernel<<<BlocksFor(B * Full * hid), 256>>>(tmp, B * Full, Dim, L.w3.weight, nullptr, hid, ff3);
    SiLUMulKernel<<<BlocksFor(B * Full * hid), 256>>>(ff1, ff3, B * Full * hid, ffg);
    LinearKernel<<<BlocksFor(B * Full * Dim), 256>>>(ffg, B * Full, hid, L.w2.weight, nullptr, Dim, ff2);
    AddInplaceKernel<<<BlocksFor(B * Full * Dim), 256>>>(h, ff2, B * Full * Dim);

    cudaFree(ff1); cudaFree(ff3); cudaFree(ffg); cudaFree(ff2);
    cudaFree(attn_out); cudaFree(merged); cudaFree(attn); cudaFree(scores);
    if (Rep > 1) { cudaFree(k_rep); cudaFree(v_rep); }
    cudaFree(q); cudaFree(k); cudaFree(v); cudaFree(q4); cudaFree(k4); cudaFree(v4);
  }

  RMSNormKernel<<<B * Full, Dim>>>(h, w_.final_norm.weight, B * Full, Dim, w_.final_norm.eps, tmp);

  // h_seq = h[:, 1:, :]  (graph token count = 1)
  float* h_seq = tmp + Dim;
  int64_t out_rows = B * (Seq + 1);

  float* out_mid = nullptr;
  cudaMalloc(&out_mid, sizeof(float) * out_rows * Dim);
  LinearKernel<<<BlocksFor(out_rows * Dim), 256>>>(h_seq, out_rows, Dim, w_.out1.weight, w_.out1.bias, Dim, out_mid);
  GELUKernel<<<BlocksFor(out_rows * Dim), 256>>>(out_mid, out_rows * Dim, out_mid);
  LinearKernel<<<BlocksFor(out_rows * io.feat_dim), 256>>>(out_mid, out_rows, Dim, w_.out2.weight, w_.out2.bias, io.feat_dim, io.output);

  cudaFree(out_mid);
  cudaFree(tmp);
  cudaFree(gbias);
  cudaFree(mask);
  cudaFree(h);
  cudaFree(tokens);
  cudaFree(input_proj);
}

}  // namespace neugn::decoder
