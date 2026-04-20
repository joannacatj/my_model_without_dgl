"""Example parity test between Python GNN encoder and CUDA encoder implementation.

Usage (after you expose a pybind module named `neugn_encoder_cuda_ext`):

  python cpp/tests/test_encoder_parity_example.py \
    --checkpoint checkpoints/wikics/gin_checkpoint.pth \
    --config-path .

The extension is expected to provide:
  run_encoder(feat_id, edge_index, batch_idx, rwse, weights_dict) -> (graph_feature, node_feature)
where all tensor arguments are CUDA tensors.
"""

import argparse
from typing import Dict

import torch

from NeuGN.model import GraphDecoder
from NeuGN.utils import load_model_args, SimpleGraph


def _load_model(checkpoint: str, config_path: str, device: torch.device):
    args = load_model_args(config_path)
    model = GraphDecoder(args).to(device).eval()
    state = torch.load(checkpoint, map_location=device)
    state_dict = state.get("model_state_dict", state)
    state_dict = {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    return model


def _build_random_graph(num_nodes: int, num_edges: int, fixed_input_dim: int, rwse_dim: int, device: torch.device):
    src = torch.randint(0, num_nodes, (num_edges,), device=device)
    dst = torch.randint(0, num_nodes, (num_edges,), device=device)
    edge_index = torch.stack([src, dst], dim=0)

    feat_id = torch.randint(0, fixed_input_dim, (num_nodes,), dtype=torch.long, device=device)
    graph = SimpleGraph(edge_index=edge_index, num_nodes=num_nodes, node_feat=feat_id)
    graph.batch = torch.zeros(num_nodes, dtype=torch.long, device=device)

    if rwse_dim > 0:
        graph.ndata["rwse"] = torch.randn(num_nodes, rwse_dim, device=device)
    return graph


def _extract_encoder_weights(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: v for k, v in state_dict.items() if k.startswith("encoder.")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/wikics/gin_checkpoint.pth")
    parser.add_argument("--config-path", default=".")
    parser.add_argument("--num-nodes", type=int, default=64)
    parser.add_argument("--num-edges", type=int, default=256)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this parity example.")

    device = torch.device("cuda")
    model = _load_model(args.checkpoint, args.config_path, device)

    fixed_input_dim = model.params.encoder_config.fixed_input_dim
    rwse_dim = getattr(model.params.encoder_config, "rwse_dim", 0)

    graph = _build_random_graph(args.num_nodes, args.num_edges, fixed_input_dim, rwse_dim, device)

    with torch.no_grad():
        py_graph_feature, py_node_feature = model.encoder(graph)

    # CUDA extension call (to be provided by pybind wrapper around cpp/src/encoder/*).
    try:
        import neugn_encoder_cuda_ext  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Please build neugn_encoder_cuda_ext first, then rerun. "
            "Expected API: run_encoder(feat_id, edge_index, batch_idx, rwse, weights_dict)."
        ) from e

    weights = _extract_encoder_weights(model.state_dict())
    rwse = graph.ndata.get("rwse")
    cuda_graph_feature, cuda_node_feature = neugn_encoder_cuda_ext.run_encoder(
        graph.ndata["feat_id"], graph.edge_index, graph.batch, rwse, weights
    )

    max_abs_graph = (py_graph_feature - cuda_graph_feature).abs().max().item()
    max_abs_node = (py_node_feature - cuda_node_feature).abs().max().item()

    print("=== Encoder parity report ===")
    print(f"graph_feature max_abs_diff: {max_abs_graph:.6e}")
    print(f"node_feature  max_abs_diff: {max_abs_node:.6e}")
    print("(建议阈值示例: fp32<=1e-5, fp16<=5e-3; 以最终定义为准)")


if __name__ == "__main__":
    main()
