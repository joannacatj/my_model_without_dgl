"""Python decoder验证：直接调用已训练NeuGN decoder并输出结果。"""

import argparse
import torch

from NeuGN.model import GraphDecoder
from NeuGN.utils import load_model_args


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/wikics/gin_checkpoint.pth")
    p.add_argument("--config-path", default=".")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = p.parse_args()

    device = torch.device(args.device)
    cfg = load_model_args(args.config_path)
    model = GraphDecoder(cfg).to(device).eval()

    ckpt = torch.load(args.checkpoint, map_location=device)
    sd = ckpt.get("model_state_dict", ckpt)
    sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
    model.load_state_dict(sd)

    B = 1
    Seq = 3
    Dim = cfg.decoder_config.dim
    Feat = cfg.encoder_config.graph_feature_dim

    # 与 CUDA 示例保持完全一致的固定输入，便于逐元素比对。
    graph_feat = torch.full((B, 1, Dim), 0.01, dtype=torch.float32, device=device)
    input_feat = torch.full((B, Seq, Feat), 0.02, dtype=torch.float32, device=device)
    subnode_ids = torch.tensor([[0, 1, 2]], dtype=torch.long, device=device)
    token_mask_len = torch.tensor([Seq], dtype=torch.long, device=device)

    max_spd = getattr(cfg.decoder_config, "max_spd", 20)
    spd = torch.tensor([[[0, 1, 2], [1, 0, 1], [2, 1, 0]]], dtype=torch.long, device=device)
    spd = torch.clamp(spd, 0, max_spd + 1)

    with torch.no_grad():
        out = model.decoder(graph_feat, input_feat, subnode_ids, token_mask_len, start_pos=0, spd_indices=spd)

    print("=== Python Trained NeuGN Decoder Output ===")
    print("decoder_output shape=", tuple(out.shape))
    print("decoder_output values=", [round(v, 6) for v in out.flatten().tolist()])


if __name__ == "__main__":
    main()
