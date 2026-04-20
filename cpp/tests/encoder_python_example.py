"""Python 示例：直接调用与 CUDA 示例同构的 encoder 逻辑并打印输出。

用法:
  python cpp/tests/encoder_python_example.py
"""

import torch


def build_demo_inputs(device="cpu"):
    # 固定小图（与 C++/CUDA 示例保持一致）
    num_nodes = 4
    src = torch.tensor([0, 1, 2, 2, 3, 0], dtype=torch.long, device=device)
    dst = torch.tensor([1, 2, 0, 3, 1, 3], dtype=torch.long, device=device)

    feat_id = torch.tensor([1, 4, 2, 0], dtype=torch.long, device=device)
    batch_idx = torch.tensor([0, 0, 0, 0], dtype=torch.long, device=device)

    return num_nodes, src, dst, feat_id, batch_idx


def build_demo_weights(device="cpu"):
    fixed_input_dim = 8
    feat_dim = 4

    # value_projection: [feat_dim, fixed_input_dim]
    value_w = torch.tensor(
        [
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            [0.2, 0.1, 0.4, 0.3, 0.6, 0.5, 0.8, 0.7],
            [0.3, 0.4, 0.1, 0.2, 0.7, 0.8, 0.5, 0.6],
            [0.4, 0.3, 0.2, 0.1, 0.8, 0.7, 0.6, 0.5],
        ],
        dtype=torch.float32,
        device=device,
    )
    value_b = torch.tensor([0.01, -0.02, 0.03, -0.04], dtype=torch.float32, device=device)

    # degree_embedding: [1001, feat_dim]，这里只填前几行
    deg_emb = torch.zeros(1001, feat_dim, dtype=torch.float32, device=device)
    deg_emb[0] = torch.tensor([0.0, 0.0, 0.0, 0.0], device=device)
    deg_emb[1] = torch.tensor([0.1, 0.0, 0.0, 0.1], device=device)
    deg_emb[2] = torch.tensor([0.2, 0.1, 0.1, 0.2], device=device)
    deg_emb[3] = torch.tensor([0.3, 0.2, 0.2, 0.3], device=device)

    # GCN linear：identity
    gcn_w = torch.eye(feat_dim, dtype=torch.float32, device=device)
    gcn_b = torch.zeros(feat_dim, dtype=torch.float32, device=device)

    # BN inference 参数
    bn_gamma = torch.ones(feat_dim, dtype=torch.float32, device=device)
    bn_beta = torch.zeros(feat_dim, dtype=torch.float32, device=device)
    bn_mean = torch.zeros(feat_dim, dtype=torch.float32, device=device)
    bn_var = torch.ones(feat_dim, dtype=torch.float32, device=device)
    bn_eps = 1e-5

    return {
        "fixed_input_dim": fixed_input_dim,
        "feat_dim": feat_dim,
        "value_w": value_w,
        "value_b": value_b,
        "deg_emb": deg_emb,
        "gcn_w": gcn_w,
        "gcn_b": gcn_b,
        "bn_gamma": bn_gamma,
        "bn_beta": bn_beta,
        "bn_mean": bn_mean,
        "bn_var": bn_var,
        "bn_eps": bn_eps,
    }


def build_dst_csr(num_nodes, src, dst):
    # destination-centered CSR，与 CUDA 实现一致
    e = src.numel()
    row_ptr = torch.zeros(num_nodes + 1, dtype=torch.long)
    for i in range(e):
        row_ptr[dst[i] + 1] += 1
    row_ptr = row_ptr.cumsum(dim=0)

    col = torch.empty(e, dtype=torch.long)
    cursor = row_ptr.clone()
    for i in range(e):
        d = dst[i]
        s = src[i]
        col[cursor[d]] = s
        cursor[d] += 1
    return row_ptr, col


def forward_python(num_nodes, src, dst, feat_id, batch_idx, w):
    fixed_input_dim = w["fixed_input_dim"]

    # value_projection(one_hot(feat_id)) 等价：取 weight[:, token] + bias
    token = feat_id % fixed_input_dim
    h_attr = w["value_w"][:, token].T + w["value_b"]

    # in-degree + degree embedding
    in_deg = torch.bincount(dst, minlength=num_nodes).clamp(0, 1000)
    h_deg = w["deg_emb"][in_deg]
    h = h_attr + h_deg

    # GCN aggregate (按 CUDA 版公式)
    row_ptr, col = build_dst_csr(num_nodes, src, dst)
    out = torch.zeros_like(h)
    for n in range(num_nodes):
        deg_n = (row_ptr[n + 1] - row_ptr[n]).item() + 1
        out[n] += h[n] / float(deg_n)
        for eidx in range(row_ptr[n].item(), row_ptr[n + 1].item()):
            s = col[eidx].item()
            deg_s = (row_ptr[s + 1] - row_ptr[s]).item() + 1
            norm = (deg_n * deg_s) ** (-0.5)
            out[n] += h[s] * norm

    # linear(identity) + BN(infer) + ReLU
    out = out @ w["gcn_w"].T + w["gcn_b"]
    out = (out - w["bn_mean"]) / torch.sqrt(w["bn_var"] + w["bn_eps"])
    out = out * w["bn_gamma"] + w["bn_beta"]
    out = torch.relu(out)

    # global mean pool
    bsz = int(batch_idx.max().item()) + 1
    graph_feature = torch.zeros(bsz, out.size(1), dtype=torch.float32)
    count = torch.zeros(bsz, dtype=torch.float32)
    for i in range(num_nodes):
        b = int(batch_idx[i].item())
        graph_feature[b] += out[i]
        count[b] += 1.0
    graph_feature = graph_feature / count.unsqueeze(1).clamp(min=1.0)

    return graph_feature, out


def print_tensor(name, x):
    print(f"{name} shape={tuple(x.shape)}")
    flat = x.flatten().tolist()
    print(f"{name} values=", [round(v, 6) for v in flat])


def main():
    num_nodes, src, dst, feat_id, batch_idx = build_demo_inputs(device="cpu")
    w = build_demo_weights(device="cpu")
    graph_feature, all_node_features = forward_python(num_nodes, src, dst, feat_id, batch_idx, w)

    print("=== Python Encoder Output ===")
    print_tensor("graph_feature", graph_feature)
    print_tensor("all_node_features", all_node_features)


if __name__ == "__main__":
    main()
