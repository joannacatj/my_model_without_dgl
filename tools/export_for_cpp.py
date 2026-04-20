import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import torch

# Ensure local repo root is importable when running as:
#   python tools/export_for_cpp.py ...
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from NeuGN.model import GraphDecoder
from NeuGN.utils import load_model_args


def _strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Align with load_inference_checkpoint: remove leading `module.` when present."""
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict


def _load_model(checkpoint_path: Path, config_path: Path, device: torch.device) -> GraphDecoder:
    args = load_model_args(str(config_path))
    model = GraphDecoder(args).to(device)
    model.eval()

    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    state_dict = _strip_module_prefix(state_dict)
    model.load_state_dict(state_dict)
    return model


def _collect_export_tensors(model: GraphDecoder) -> Dict[str, torch.Tensor]:
    state = model.state_dict()
    selected = {
        name: tensor
        for name, tensor in state.items()
        if name.startswith("encoder.") or name.startswith("decoder.")
    }
    if not selected:
        raise RuntimeError("No encoder/decoder tensors found in model.state_dict().")
    return selected


def _match_names(all_names: Iterable[str], prefix: str) -> List[str]:
    return sorted([name for name in all_names if name.startswith(prefix)])


def _ordered_names(all_names: List[str], n_layers: int) -> List[str]:
    ordered: List[str] = []
    used = set()

    def add_matches(matches: Iterable[str]):
        for name in matches:
            if name not in used:
                ordered.append(name)
                used.add(name)

    # Encoder fixed order
    add_matches(_match_names(all_names, "encoder.value_projection"))
    add_matches(_match_names(all_names, "encoder.degree_embedding"))
    add_matches(_match_names(all_names, "encoder.rwse_projection"))

    for layer_idx in range(n_layers):
        add_matches(_match_names(all_names, f"encoder.convs.{layer_idx}.mlp"))
        add_matches(_match_names(all_names, f"encoder.convs.{layer_idx}.linear"))
        add_matches(_match_names(all_names, f"encoder.convs.{layer_idx}.eps"))
        add_matches(_match_names(all_names, f"encoder.batch_norms.{layer_idx}"))

    # Decoder fixed order
    add_matches(_match_names(all_names, "decoder.input_projection"))
    add_matches(_match_names(all_names, "decoder.sos_token"))
    add_matches(_match_names(all_names, "decoder.type_embeddings"))
    add_matches(_match_names(all_names, "decoder.spd_bias_embedding"))

    for layer_idx in range(n_layers):
        base = f"decoder.layers.{layer_idx}"
        add_matches(_match_names(all_names, f"{base}.attention.wq"))
        add_matches(_match_names(all_names, f"{base}.attention.wk"))
        add_matches(_match_names(all_names, f"{base}.attention.wv"))
        add_matches(_match_names(all_names, f"{base}.attention.wo"))
        add_matches(_match_names(all_names, f"{base}.feed_forward.w1"))
        add_matches(_match_names(all_names, f"{base}.feed_forward.w2"))
        add_matches(_match_names(all_names, f"{base}.feed_forward.w3"))
        add_matches(_match_names(all_names, f"{base}.attention_norm"))
        add_matches(_match_names(all_names, f"{base}.ffn_norm"))

    add_matches(_match_names(all_names, "decoder.norm"))
    add_matches(_match_names(all_names, "decoder.output"))

    # Keep remaining encoder/decoder tensors stable and deterministic.
    remaining = sorted([name for name in all_names if name not in used])
    ordered.extend(remaining)
    return ordered


def _write_bin(weights: Dict[str, torch.Tensor], ordered_names: List[str], output_file: Path) -> List[Dict]:
    manifest: List[Dict] = []
    offset = 0

    with output_file.open("wb") as f:
        for name in ordered_names:
            tensor = weights[name].detach().cpu().contiguous()
            # Use NumPy bytes export directly (supports scalar tensors such as BN num_batches_tracked).
            raw = tensor.numpy().tobytes()
            nbytes = tensor.numel() * tensor.element_size()

            f.write(raw)
            manifest.append(
                {
                    "name": name,
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "offset": offset,
                    "nbytes": nbytes,
                }
            )
            offset += nbytes

    return manifest


def _export_config(config_path: Path) -> Dict:
    args = load_model_args(str(config_path))
    return {
        "dim": args.decoder_config.dim,
        "n_layers": args.decoder_config.n_layers,
        "n_heads": args.decoder_config.n_heads,
        "max_spd": getattr(args.decoder_config, "max_spd", None),
        "rwse_dim": getattr(args.encoder_config, "rwse_dim", None),
        "fixed_input_dim": getattr(args.encoder_config, "fixed_input_dim", None),
    }


def main():
    parser = argparse.ArgumentParser(description="Export GraphDecoder weights for C++ inference.")
    parser.add_argument("--checkpoint", default="checkpoints/wikics/gin_checkpoint.pth", help="Checkpoint path")
    parser.add_argument("--config-path", default=".", help="Directory containing model_args.yaml")
    parser.add_argument("--output-dir", default="checkpoints/wikics/export_cpp", help="Export output directory")
    parser.add_argument("--weights-file", default="graphdecoder_weights.bin", help="Binary weights file name")
    parser.add_argument("--manifest-file", default="weights_manifest.json", help="Manifest file name")
    parser.add_argument("--config-file", default="export_config.json", help="Export config snapshot file name")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    config_path = Path(args.config_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")
    model = _load_model(checkpoint_path=checkpoint_path, config_path=config_path, device=device)
    weights = _collect_export_tensors(model)

    all_names = list(weights.keys())
    n_layers = model.params.decoder_config.n_layers
    ordered_names = _ordered_names(all_names, n_layers=n_layers)

    weights_file = output_dir / args.weights_file
    manifest = _write_bin(weights=weights, ordered_names=ordered_names, output_file=weights_file)

    manifest_path = output_dir / args.manifest_file
    manifest_payload = {
        "weights_file": args.weights_file,
        "total_tensors": len(manifest),
        "total_bytes": int(sum(item["nbytes"] for item in manifest)),
        "tensors": manifest,
    }
    manifest_path.write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")

    export_config_path = output_dir / args.config_file
    export_config_path.write_text(json.dumps(_export_config(config_path), indent=2), encoding="utf-8")

    print(f"Export complete: {weights_file}")
    print(f"Manifest: {manifest_path}")
    print(f"Config snapshot: {export_config_path}")


if __name__ == "__main__":
    main()
