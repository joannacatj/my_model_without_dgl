# NeuGN C++ 推理对齐说明（当前实现）

本文档逐函数对齐当前 Python 代码路径，便于后续 C++ 端并行实现：

- `NeuGN/mpnn_encoder.py::GNN.forward`
- `NeuGN/model.py::Transformer.forward`
- `demo.py::NeuGNInferenceWrapper.predict_scores`

---

## 0. 记号与默认约定

| 符号 | 含义 |
|---|---|
| `B` | batch size（推理常见为 1） |
| `N` | 当前 batched graph 中节点总数 |
| `E` | 边数（`edge_index` 第二维） |
| `Seq` | 路径 token 数（不含 Graph/SOS 时指路径长度） |
| `Heads` | 多头注意力头数 (`n_heads`) |
| `HeadDim` | 每个头维度 (`dim / n_heads`) |
| `Dim` | decoder hidden dim (`decoder_config.dim`) |
| `FeatDim` | graph feature dim (`encoder_config.graph_feature_dim`) |

> **dtype/device 标注原则**：
> - 线性层/注意力主通路推荐 `fp16`（权重可 fp16/fp32，累加建议 fp32）；
> - `softmax`、`norm`、相似度归一化推荐使用 `fp32` 计算后再 cast；
> - 索引类张量（id、mask、spd）使用 `int64`；
> - 目标部署设备为 **CUDA**。

---

## 1. `GNN.forward` 对齐（`NeuGN/mpnn_encoder.py`）

### 输入/输出

| 名称 | Shape | dtype | device | 说明 |
|---|---:|---|---|---|
| `batched_graph.ndata['feat_id']` | `[N]` | int64 | CUDA | 节点离散特征 id |
| `batched_graph.edge_index` | `[2, E]` | int64 | CUDA | COO 边索引 |
| `batched_graph.ndata['rwse']`(可选) | `[N, rwse_dim]` | fp32/fp16 | CUDA | 结构编码 |
| 输出 `graph_feature` | `[B, Dim]` | fp16/fp32 | CUDA | 图级 embedding |
| 输出 `h` | `[N, Dim]` | fp16/fp32 | CUDA | 节点级 embedding |

### 分层流程（按代码顺序）

| 步骤 | 关键操作 | 输入 Shape | 输出 Shape | dtype/device |
|---|---|---|---|---|
| 1 | `h_id_mapped = feat_id % fixed_input_dim` | `[N]` | `[N]` | int64 / CUDA |
| 2 | `one_hot` + `value_projection` | `[N] -> [N, fixed_input_dim]` | `[N, FeatDim]` | fp32(输入) -> fp16/fp32 / CUDA |
| 3 | `rwse_projection`(若启用)并相加 | `[N, rwse_dim]` | `[N, FeatDim]` | fp16/fp32 / CUDA |
| 4 | `degree_embedding(in_degrees)` | `[N]` | `[N, FeatDim]` | fp16/fp32 / CUDA |
| 5 | `h = h_attr + h_deg` | `[N, FeatDim]` | `[N, FeatDim]` | fp16/fp32 / CUDA |
| 6 | GNN layer `i=0..L-1` | `[N, in_dim_i]` | `[N, Dim]` | fp16/fp32 / CUDA |
| 6.1 | GIN: 邻居聚合 + `MLP` / GCN: 归一化聚合+线性 | `[N,*]`,`[2,E]` | `[N, Dim]` | fp16/fp32 / CUDA |
| 6.2 | `BatchNorm1d` + `ReLU` | `[N, Dim]` | `[N, Dim]` | fp16/fp32 / CUDA |
| 7 | `global_mean_pool(h, batch_idx)` | `[N, Dim]` | `[B, Dim]` | fp16/fp32 / CUDA |

---

## 2. `Transformer.forward` 对齐（`NeuGN/model.py`）

### 输入/输出

| 名称 | Shape | dtype | device | 说明 |
|---|---:|---|---|---|
| `graph_features` | `[B, 1, Dim]` | fp16/fp32 | CUDA | 来自 encoder 图向量 |
| `input_features` | `[B, Seq, FeatDim]` | fp16/fp32 | CUDA | 已匹配路径节点特征 |
| `subnode_ids` | `[B, Seq]` | int64 | CUDA | 子图局部 id |
| `token_mask_len` | `[B]` | int64 | CUDA | 有效路径长度 |
| `spd_indices`(可选) | `[B, Seq, Seq]` | int64 | CUDA | 节点间 SPD |
| 输出 `output` | `[B, Seq+1, FeatDim]` | fp32 | CUDA | 含 SOS 位置输出 |

### Token 组装与偏置构建

| 步骤 | 操作 | 输入 Shape | 输出 Shape | dtype/device |
|---|---|---|---|---|
| 1 | `input_projection` | `[B, Seq, FeatDim]` | `[B, Seq, Dim]` | fp16/fp32 / CUDA |
| 2 | `sos_token.expand` + `cat` | `[1,1,Dim] + [B,Seq,Dim]` | `[B, Seq+1, Dim]` | fp16/fp32 / CUDA |
| 3 | `node_embeddings(subnode_ids_extended)` | `[B, Seq+1]` | `[B, Seq+1, Dim]` | fp16/fp32 / CUDA |
| 4 | `type_embeddings`（token=0, graph=1） | `[B, Seq+1]`, `[B,1]` | 同 shape | fp16/fp32 / CUDA |
| 5 | `h_graph = graph_features + type_emb(graph)` | `[B,1,Dim]` | `[B,1,Dim]` | fp16/fp32 / CUDA |
| 6 | `h = cat(h_graph, h_tokens)` | `[B,1,Dim] + [B,Seq+1,Dim]` | `[B, Seq+2, Dim]` | fp16/fp32 / CUDA |
| 7 | SPD bias 扩展并查表 | `[B, Seq, Seq]` | `[B, Heads, Seq+2, Seq+2]` | fp16/fp32 / CUDA |

### Attention / FFN 堆叠（每层）

设 `FullSeq = Seq + 2`。

| 子步骤 | 输入 Shape | 输出 Shape | dtype/device |
|---|---|---|---|
| `attention_norm` | `[B, FullSeq, Dim]` | `[B, FullSeq, Dim]` | RMSNorm 内部建议 fp32 计算 / CUDA |
| `wq/wk/wv` | `[B, FullSeq, Dim]` | `q:[B,FullSeq,Heads,HeadDim]` 等 | fp16/fp32 / CUDA |
| score matmul + mask + `graph_bias` | `q,k -> [B,Heads,FullSeq,FullSeq]` | `[B,Heads,FullSeq,FullSeq]` | softmax 建议 fp32 / CUDA |
| `wo` | `[B, FullSeq, Heads*HeadDim]` | `[B, FullSeq, Dim]` | fp16/fp32 / CUDA |
| 残差后 `ffn_norm` | `[B, FullSeq, Dim]` | `[B, FullSeq, Dim]` | fp16/fp32 / CUDA |
| `w1/w3` + SiLU + `w2` | `[B, FullSeq, Dim]` | `[B, FullSeq, Dim]` | fp16/fp32 / CUDA |

### 尾部输出

| 步骤 | 输入 Shape | 输出 Shape | dtype/device |
|---|---|---|---|
| `norm` | `[B, FullSeq, Dim]` | `[B, FullSeq, Dim]` | fp16/fp32 / CUDA |
| `h_seq = h[:, graph_tokens:, :]` | `[B, FullSeq, Dim]` | `[B, Seq+1, Dim]` | fp16/fp32 / CUDA |
| `output MLP` + `.float()` | `[B, Seq+1, Dim]` | `[B, Seq+1, FeatDim]` | **fp32** / CUDA |

---

## 3. `predict_scores` 对齐（`demo.py`）

### 输入/输出

| 名称 | Shape | dtype | device | 说明 |
|---|---:|---|---|---|
| `target_query_node` | 标量 | int | CPU | 待扩展 query 节点 |
| `partial_mapping` | 字典 | int->int | CPU | query 到 data 节点映射 |
| `candidates` | `[K]`(逻辑) | int | CPU | 候选 data 节点 |
| 输出 `candidate_scores` | `K` 项 | float | CPU | 候选分数 |

### 推理主路径

| 步骤 | 操作 | 输入 Shape | 输出 Shape | dtype/device |
|---|---|---|---|---|
| 1 | 根据 `target_idx` 截断路径 | path list | `input_path_nodes` 长度 `Seq` | CPU |
| 2 | 构建 `input_features` | `Seq` 个向量 | `[1, Seq, FeatDim]` 或 `[1,0,FeatDim]` | fp16/fp32 / CUDA |
| 3 | 构建 `seq_spd_indices` | query SPD 全矩阵 | `[1, Seq, Seq]` 或 `None` | int64 / CUDA |
| 4 | 构建 `subnodes_tensor`/`token_mask_len` | `Seq` | `[1, Seq]` / `[1]` | int64 / CUDA |
| 5 | 调 `decoder(...)` | 见上 | `[1, Seq+1, FeatDim]` | fp32 / CUDA |
| 6 | 取 `pred_vector = outputs[:, -1, :]` | `[1, Seq+1, FeatDim]` | `[1, FeatDim]` | fp32 / CUDA |
| 7 | `F.normalize` + 候选向量点积 | `[1,FeatDim]`, `[K,FeatDim]` | `[K]` | fp32 / CUDA |
| 8 | 回写字典 | `[K]` | `{cand: score}` | float / CPU |

---

## 4. C++ 迁移时建议的精度策略

- 参数文件可保留 fp32，加载后按层选择 cast 到 fp16（Tensor Core 路径）。
- Attention logits、softmax、RMSNorm、最终 cosine 归一化建议 fp32 计算。
- SPD 索引与 token id 均维持 int64（与 PyTorch embedding 索引一致）。

---

## 5. 与导出脚本的衔接

- 权重命名与顺序以 `tools/export_for_cpp.py` 的 `weights_manifest.json` 为准。
- C++ 端按 `name/shape/dtype/offset` 逐项装载，可先完成结构对齐，再接 CUDA kernel/框架（LibTorch/TensorRT/自研）。

---

## 6. CUDA Encoder 实现位置与对齐点

已在 `cpp/src/encoder/` 提供 CUDA 版本实现：

- `encoder_cuda_kernels.cuh/.cu`
  - `PureGINAggregateCUDA`: `out = (1+eps)*x + Sum(neighbor)`（供 GIN MLP 前使用）
  - `PureGraphConvAggregateCUDA`: `D^-1/2 (A+I) D^-1/2 X` 聚合
  - `GlobalMeanPoolCUDA`
  - `DegreeEmbeddingForwardCUDA`
  - `ValueProjectionForwardCUDA`
  - `RWSEProjectionForwardCUDA`
  - `BatchNormInferenceReLUCUDA`（固定 running mean/var）
- `graph_encoder_cuda.h/.cu`
  - `GraphEncoderCUDA::Forward` 串联层次：
    `value_projection + degree_embedding (+rwse)` -> conv -> `BN(infer)+ReLU` -> `global_mean_pool`

实现细节：

- 图聚合主路径使用 **CSR**（`BuildCSRFromCOO`）以对齐“优先 CSR”要求。
- BN 推理模式只用 running 均值/方差，不更新统计量。
- 层后处理顺序与 Python 对齐：每层后 `BN + ReLU`。

---

## 7. Python vs CUDA 一致性测试示例

示例脚本：`cpp/tests/test_encoder_parity_example.py`

```bash
python cpp/tests/test_encoder_parity_example.py \
  --checkpoint checkpoints/wikics/gin_checkpoint.pth \
  --config-path .
```

该脚本流程：

1. 加载 Python `GraphDecoder.encoder` 作为基线输出。
2. 调用 CUDA 扩展 `neugn_encoder_cuda_ext.run_encoder(...)` 获取 CUDA 输出。
3. 计算并打印 `graph_feature` 与 `all_node_features` 的 `max_abs_diff`。

> 注意：脚本假定你已将 `cpp/src/encoder/*` 通过 pybind 暴露为 `neugn_encoder_cuda_ext`。本文先提供“可对齐的测试范式”，便于你马上验证。 
