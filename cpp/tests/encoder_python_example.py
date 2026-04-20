"""Python 验证：加载已训练 NeuGN checkpoint，运行 encoder 并输出结果。

用法:
  python cpp/tests/encoder_python_example.py \
    --checkpoint checkpoints/wikics/gin_checkpoint.pth \
    --config-path .
"""

import argparse
import torch

from NeuGN.model import GraphDecoder
from NeuGN.utils import SimpleGraph, load_model_args


def load_trained_encoder(checkpoint: str, config_path: str, device: torch.device):
    args = load_model_args(config_path)
    model = GraphDecoder(args).to(device).eval()

    ckpt = torch.load(checkpoint, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)
    state_dict = {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    return model.encoder, args


def build_demo_graph(args, device: torch.device):
    # 与 C++ 示例保持一致的小图输入
    src = torch.tensor([0, 1, 2, 2, 3, 0], dtype=torch.long, device=device)
    dst = torch.tensor([1, 2, 0, 3, 1, 3], dtype=torch.long, device=device)
    edge_index = torch.stack([src, dst], dim=0)

    fixed_input_dim = getattr(args.encoder_config, "fixed_input_dim", 50)
    feat_id = torch.tensor([1, 4, 2, 0], dtype=torch.long, device=device) % fixed_input_dim

    graph = SimpleGraph(edge_index=edge_index, num_nodes=4, node_feat=feat_id)
    graph.batch = torch.zeros(graph.num_nodes, dtype=torch.long, device=device)

    rwse_dim = getattr(args.encoder_config, "rwse_dim", 0)
    if rwse_dim > 0:
        # 与 C++ 侧保持一致：使用确定性构造而非随机数
        rwse = torch.empty(graph.num_nodes, rwse_dim, dtype=torch.float32, device=device)
        for i in range(graph.num_nodes):
            for j in range(rwse_dim):
                rwse[i, j] = 0.01 * (i + 1) * (j + 1)
        graph.ndata["rwse"] = rwse
    return graph


def print_tensor(name, x):
    print(f"{name} shape={tuple(x.shape)}")
    vals = x.detach().cpu().flatten().tolist()
    print(f"{name} values=", [round(v, 6) for v in vals])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/wikics/gin_checkpoint.pth")
    parser.add_argument("--config-path", default=".")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()

    device = torch.device(args.device)
    encoder, model_args = load_trained_encoder(args.checkpoint, args.config_path, device)
    graph = build_demo_graph(model_args, device)

    with torch.no_grad():
        graph_feature, all_node_features = encoder(graph)

    print("=== Python Trained NeuGN Encoder Output ===")
    print_tensor("graph_feature", graph_feature)
    print_tensor("all_node_features", all_node_features)


if __name__ == "__main__":
    main()
