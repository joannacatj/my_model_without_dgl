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
- COO->CSR 已增加 **CUB device 侧构建** 快路径（`DeviceRadixSort + DeviceRunLengthEncode + DeviceScan`），减少 host 参与与内存往返。
- BN 推理模式只用 running 均值/方差，不更新统计量。
- 层后处理顺序与 Python 对齐：每层后 `BN + ReLU`。
- `global_mean_pool` 新增 **CUB segmented reduce** 快路径（`cub::DeviceSegmentedReduce::Sum`），在 batch 索引按图分组时可替代原子加汇聚，提升吞吐；未分组时自动回退到原实现，保证结果正确。

---

## 7. Python / CUDA 双文件手工比对示例（无 pybind）

提供两个独立文件，分别直接输出结果，供你人工比对：

1. Python: `cpp/tests/encoder_python_example.py`
2. CUDA: `cpp/tests/encoder_cuda_example.cu`

Python 运行：

```bash
python cpp/tests/encoder_python_example.py \
  --checkpoint checkpoints/wikics/gin_checkpoint.pth \
  --config-path .
```

CUDA 编译与运行：

```bash
# 先导出参数
python tools/export_for_cpp.py \
  --checkpoint checkpoints/wikics/gin_checkpoint.pth \
  --config-path . \
  --output-dir checkpoints/wikics/export_cpp

# 编译
nvcc -std=c++17 \
  cpp/tests/encoder_cuda_example.cu \
  cpp/src/encoder/encoder_cuda_kernels.cu \
  cpp/src/encoder/graph_encoder_cuda.cu \
  -Icpp/src/encoder -o /tmp/encoder_cuda_example

# 运行（从导出参数加载）
/tmp/encoder_cuda_example \
  --manifest checkpoints/wikics/export_cpp/weights_manifest.json \
  --weights checkpoints/wikics/export_cpp/graphdecoder_weights.bin \
  --config checkpoints/wikics/export_cpp/export_config.json
```

说明：

- Python 端会加载**已训练 checkpoint**并运行真实 `encoder`。
- CUDA 端会读取**导出参数**（bin + manifest + export_config）并运行 `GraphEncoderCUDA`。
- 两边使用同一组固定图结构输入，输出字段一致：`graph_feature` 与 `all_node_features`。

### Decoder 验证（新增）

Python（已训练模型直接调用 decoder）：

```bash
python cpp/tests/decoder_python_example.py \
  --checkpoint checkpoints/wikics/gin_checkpoint.pth \
  --config-path .
```

CUDA（读取导出参数运行 `GraphTransformerDecoderCUDA`）：

```bash
nvcc -std=c++17 \
  cpp/tests/decoder_cuda_example.cu \
  cpp/src/decoder/graph_transformer_decoder_cuda.cu \
  -Icpp/src/decoder -o /tmp/decoder_cuda_example

/tmp/decoder_cuda_example \
  --manifest checkpoints/wikics/export_cpp/weights_manifest.json \
  --weights checkpoints/wikics/export_cpp/graphdecoder_weights.bin \
  --config checkpoints/wikics/export_cpp/export_config.json
```

一键对比（自动执行导出、Python/CUDA 运行、逐元素误差检查）：

```bash
python tools/export_for_cpp.py --checkpoint checkpoints/wikics/gin_checkpoint.pth --config-path . --output-dir checkpoints/wikics/export_cpp \
&& python cpp/tests/decoder_python_example.py --checkpoint checkpoints/wikics/gin_checkpoint.pth --config-path . --device cpu > /tmp/decoder_py_out.txt \
&& nvcc -std=c++17 cpp/tests/decoder_cuda_example.cu cpp/src/decoder/graph_transformer_decoder_cuda.cu -Icpp/src/decoder -o /tmp/decoder_cuda_example \
&& /tmp/decoder_cuda_example --manifest checkpoints/wikics/export_cpp/weights_manifest.json --weights checkpoints/wikics/export_cpp/graphdecoder_weights.bin --config checkpoints/wikics/export_cpp/export_config.json > /tmp/decoder_cuda_out.txt \
&& python - <<'PY'
import re, sys
from pathlib import Path

def parse_shape(text):
    m = re.search(r"decoder_output shape=\\s*\\(([^)]*)\\)", text)
    if not m:
        raise RuntimeError("cannot parse decoder_output shape")
    return tuple(int(x.strip()) for x in m.group(1).split(",") if x.strip())

def parse_vals(text):
    m = re.search(r"decoder_output values=\\s*(\\[[^\\]]*\\]|[^\\n]+)", text, re.S)
    if not m:
        raise RuntimeError("cannot parse decoder_output values")
    return [float(x) for x in re.findall(r"[-+]?\\d*\\.?\\d+(?:[eE][-+]?\\d+)?", m.group(1))]

py_out = Path("/tmp/decoder_py_out.txt").read_text()
cu_out = Path("/tmp/decoder_cuda_out.txt").read_text()
py_shape, cu_shape = parse_shape(py_out), parse_shape(cu_out)
if py_shape != cu_shape:
    print(f"[FAIL] shape mismatch: python={py_shape}, cuda={cu_shape}")
    sys.exit(2)

py_vals, cu_vals = parse_vals(py_out), parse_vals(cu_out)
if len(py_vals) != len(cu_vals):
    print(f"[FAIL] value length mismatch: python={len(py_vals)}, cuda={len(cu_vals)}")
    sys.exit(2)

atol = 1e-3
mad = max(abs(a-b) for a,b in zip(py_vals, cu_vals)) if py_vals else 0.0
if mad > atol:
    print(f"[FAIL] max_abs_diff={mad:.6e} > atol={atol:.6e}")
    sys.exit(1)

print(f"[PASS] shape={py_shape}, max_abs_diff={mad:.6e} <= atol={atol:.6e}")
PY
```

说明：

- 当前 `decoder_python_example.py` 与 `decoder_cuda_example.cu` 的输入已统一为固定常量：
  - `graph_feat` 全 0.01
  - `input_feat` 全 0.02
  - `subnode_ids=[0,1,2]`，`token_mask_len=[3]`，`spd=[[0,1,2],[1,0,1],[2,1,0]]`
- 上面的 bash 命令会先检查 `decoder_output shape`，再计算 `max_abs_diff` 并按 `atol=1e-3` 判定 PASS/FAIL。

---

## 8. 预处理 C++ 版本（结果一致优先）

在 `cpp/src/preprocess/` 提供了与 `NeuGN/utils.py` 对齐的 CPU 参考实现：

- `compute_rwse`：随机游走结构编码（dense 方式，返回 `[N, k_steps]`）。
- `compute_subgraph_spd`：支持 BFS / Floyd-Warshall，并统一截断到 `max_spd+1`。
- `graph2path_v2_pure`：路径生成逻辑（新增 deterministic 模式，便于对齐验收）。
- `simple_node_subgraph`：节点子图抽取与本地重编号。

可用以下命令快速编译/运行示例：

```bash
g++ -std=c++17 \
  cpp/tests/preprocess_cpp_example.cpp \
  cpp/src/preprocess/preprocess_utils.cpp \
  -Icpp/src -o /tmp/preprocess_cpp_example \
  && /tmp/preprocess_cpp_example
```

CUDA 版本（`__global__` kernel 实现，结果一致优先）：

```bash
nvcc -std=c++17 \
  cpp/tests/preprocess_cuda_example.cu \
  cpp/src/preprocess/preprocess_cuda.cu \
  -Icpp/src -o /tmp/preprocess_cuda_example \
  && /tmp/preprocess_cuda_example
```
