# export_to_cpp_getters.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from NeuGN.model import GraphDecoder
from NeuGN.utils import load_model_args


class PatchedGINConv(nn.Module):
    def __init__(self, conv: nn.Module):
        super().__init__()
        self.eps = conv.eps
        self.mlp = conv.mlp

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src = edge_index[0]
        dst = edge_index[1]
        out = torch.zeros_like(x)
        out.index_add_(0, dst, x.index_select(0, src))
        out = out + (1.0 + self.eps) * x
        return self.mlp(out)


class PatchedGraphConv(nn.Module):
    def __init__(self, conv: nn.Module):
        super().__init__()
        self.linear = conv.linear

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        num_nodes = x.size(0)
        device = x.device
        self_loops = torch.arange(num_nodes, device=device).unsqueeze(0).repeat(2, 1)
        edge_index_loop = torch.cat([edge_index, self_loops], dim=1)

        src = edge_index_loop[0]
        dst = edge_index_loop[1]

        deg = torch.zeros(num_nodes, device=device, dtype=x.dtype)
        deg.index_add_(0, dst, torch.ones(src.size(0), device=device, dtype=x.dtype))
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt = torch.where(deg > 0, deg_inv_sqrt, torch.zeros_like(deg_inv_sqrt))

        norm = deg_inv_sqrt.index_select(0, src) * deg_inv_sqrt.index_select(0, dst)

        out = torch.zeros_like(x)
        out.index_add_(0, dst, x.index_select(0, src) * norm.unsqueeze(1))
        return self.linear(out)


class ConvBNReLU(nn.Module):
    def __init__(self, conv: nn.Module, bn: nn.Module):
        super().__init__()
        self.conv = conv
        self.bn = bn

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv(x, edge_index)
        x = self.bn(x)
        x = F.relu(x)
        return x


class PatchedEncoder(nn.Module):
    def __init__(self, original_encoder: nn.Module):
        super().__init__()
        self.fixed_input_dim = int(original_encoder.fixed_input_dim)
        self.rwse_dim = int(original_encoder.rwse_dim)

        self.value_projection = original_encoder.value_projection
        self.rwse_projection = getattr(original_encoder, "rwse_projection", None)
        self.degree_embedding = original_encoder.degree_embedding

        blocks = []
        for conv, bn in zip(original_encoder.convs, original_encoder.batch_norms):
            name = conv.__class__.__name__
            if "GIN" in name:
                patched_conv = PatchedGINConv(conv)
            elif "GraphConv" in name:
                patched_conv = PatchedGraphConv(conv)
            else:
                patched_conv = conv
            blocks.append(ConvBNReLU(patched_conv, bn))
        self.blocks = nn.ModuleList(blocks)

    def forward(
        self,
        feat_id: torch.Tensor,
        edge_index: torch.Tensor,
        batch_idx: torch.Tensor,
        rwse: torch.Tensor,
        in_degrees: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h_id = torch.remainder(feat_id, self.fixed_input_dim)
        h_one_hot = F.one_hot(h_id, num_classes=self.fixed_input_dim).to(torch.float32)
        h_attr = self.value_projection(h_one_hot)

        if self.rwse_dim > 0 and self.rwse_projection is not None:
            h_attr = h_attr + self.rwse_projection(rwse)

        in_degrees = torch.clamp(in_degrees, 0, 1000)
        h = h_attr + self.degree_embedding(in_degrees)

        for block in self.blocks:
            h = block(h, edge_index)

        num_graphs = int(batch_idx.max().item()) + 1
        out = torch.zeros((num_graphs, h.size(1)), device=h.device, dtype=h.dtype)
        out.index_add_(0, batch_idx, h)

        count = torch.zeros((num_graphs, 1), device=h.device, dtype=h.dtype)
        ones = torch.ones((h.size(0), 1), device=h.device, dtype=h.dtype)
        count.index_add_(0, batch_idx, ones)

        graph_feature = out / torch.clamp(count, min=1.0)
        return graph_feature, h


class PatchedTransformer(nn.Module):
    def __init__(self, original_decoder: nn.Module):
        super().__init__()
        self.input_projection = original_decoder.input_projection
        self.sos_token = original_decoder.sos_token
        self.node_embeddings = original_decoder.node_embeddings
        self.type_embeddings = original_decoder.type_embeddings
        self.pos_embeddings = original_decoder.pos_embeddings
        self.layers = original_decoder.layers
        self.norm = original_decoder.norm
        self.output = original_decoder.output
        self.spd_bias_embedding = getattr(original_decoder, "spd_bias_embedding", None)

    def forward(
        self,
        graph_features: torch.Tensor,
        input_features: torch.Tensor,
        subnode_ids: torch.Tensor,
        token_mask_len: torch.Tensor,
        start_pos: int,
        spd_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        graph_bias: Optional[torch.Tensor] = None
        _bsz = input_features.size(0)

        h_data = self.input_projection(input_features)
        sos_emb = self.sos_token.expand(_bsz, -1, -1)
        h_tokens = torch.cat([sos_emb, h_data], dim=1)

        sos_subnode_id = torch.zeros((_bsz, 1), dtype=subnode_ids.dtype, device=subnode_ids.device)
        subnode_ids_ext = torch.cat([sos_subnode_id, subnode_ids], dim=1)
        h_tokens = h_tokens + self.node_embeddings(subnode_ids_ext)

        token_types = torch.zeros((_bsz, h_tokens.size(1)), dtype=torch.long, device=h_tokens.device)
        h_tokens = h_tokens + self.type_embeddings(token_types)

        graph_feature_types = torch.ones(
            (graph_features.size(0), graph_features.size(1)),
            dtype=torch.long,
            device=graph_features.device,
        )
        h_graph = graph_features + self.type_embeddings(graph_feature_types)

        h = torch.cat([h_graph, h_tokens], dim=1)
        seqlen = h.size(1)

        if spd_indices is not None and self.spd_bias_embedding is not None:
            device = spd_indices.device
            node_seq_len = spd_indices.size(1)
            full_len = node_seq_len + 2

            current_max_spd = int(self.spd_bias_embedding.num_embeddings) - 2
            full_spd = torch.full((_bsz, full_len, full_len),
                                  current_max_spd + 1,
                                  dtype=torch.long,
                                  device=device)
            full_spd[:, 2:, 2:] = spd_indices
            full_spd[:, 0, 0] = 0
            full_spd[:, 1, 1] = 0

            bias_vals = self.spd_bias_embedding(full_spd)
            graph_bias = bias_vals.permute(0, 3, 1, 2).contiguous()

        h = h + self.pos_embeddings(h)

        positions = torch.arange(seqlen, device=h.device).unsqueeze(0).repeat(_bsz, 1)
        token_mask_len_uns = token_mask_len.unsqueeze(1)
        valid = positions < (token_mask_len_uns + 1 + graph_features.size(1))

        mask_1d = torch.full((_bsz, seqlen), float("-inf"), device=h.device, dtype=h.dtype)
        mask_1d = torch.where(valid, torch.zeros_like(mask_1d), mask_1d)
        mask = mask_1d.unsqueeze(1).unsqueeze(3).repeat(1, 1, 1, seqlen)

        for layer in self.layers:
            h = layer(h, start_pos, mask, graph_bias)

        h = self.norm(h)
        h_seq = h[:, graph_features.size(1):, :]
        return self.output(h_seq).to(torch.float32)


class ExportModule(nn.Module):
    """
    Exposes:
      - encode(...) -> (graph_features, all_node_features)
      - decode(...) -> output
      - getters: fixed_input_dim, rwse_dim, max_spd, subnode_vocab_size
    """
    def __init__(self, original_model: nn.Module):
        super().__init__()
        self.encoder = PatchedEncoder(original_model.encoder)
        self.decoder = PatchedTransformer(original_model.decoder)

        encoder_type = getattr(original_model, "encoder_type", "")
        self.need_unsqueeze = bool(encoder_type in ["gat", "gcn", "gin"])

        # ---- cache scalars for C++ ----
        # fixed_input_dim: use actual linear in_features (most reliable)
        self._fixed_input_dim: int = int(self.encoder.value_projection.in_features)
        self._rwse_dim: int = int(getattr(original_model.encoder, "rwse_dim", 0))

        # decoder vocab sizes
        self._subnode_vocab_size: int = int(self.decoder.node_embeddings.num_embeddings)

        if getattr(self.decoder, "spd_bias_embedding", None) is not None:
            # embedding size = max_spd + 2 in your implementation
            self._spd_vocab_size: int = int(self.decoder.spd_bias_embedding.num_embeddings)
            self._max_spd: int = self._spd_vocab_size - 2
        else:
            self._spd_vocab_size = 0
            self._max_spd = 0

    @torch.jit.export
    def get_fixed_input_dim(self) -> int:
        return self._fixed_input_dim

    @torch.jit.export
    def get_rwse_dim(self) -> int:
        return self._rwse_dim

    @torch.jit.export
    def get_subnode_vocab_size(self) -> int:
        return self._subnode_vocab_size

    @torch.jit.export
    def get_max_spd(self) -> int:
        return self._max_spd

    @torch.jit.export
    def get_spd_vocab_size(self) -> int:
        return self._spd_vocab_size

    @torch.jit.export
    def encode(
        self,
        feat_id: torch.Tensor,
        edge_index: torch.Tensor,
        batch_idx: torch.Tensor,
        rwse: torch.Tensor,
        in_degrees: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        graph_features, all_node_features = self.encoder(feat_id, edge_index, batch_idx, rwse, in_degrees)
        if self.need_unsqueeze:
            graph_features = graph_features.unsqueeze(1)
        return graph_features, all_node_features

    @torch.jit.export
    def decode(
        self,
        graph_features: torch.Tensor,
        input_vectors: torch.Tensor,
        subnode_ids: torch.Tensor,
        token_mask_len: torch.Tensor,
        spd_indices: torch.Tensor,
    ) -> torch.Tensor:
        return self.decoder(graph_features, input_vectors, subnode_ids, token_mask_len, 0, spd_indices)

    def forward(
        self,
        feat_id: torch.Tensor,
        edge_index: torch.Tensor,
        batch_idx: torch.Tensor,
        rwse: torch.Tensor,
        in_degrees: torch.Tensor,
        input_features_idx: torch.Tensor,
        subnode_ids: torch.Tensor,
        token_mask_len: torch.Tensor,
        spd_indices: torch.Tensor,
    ) -> torch.Tensor:
        graph_features, all_node_features = self.encode(feat_id, edge_index, batch_idx, rwse, in_degrees)

        bsz, seq_len = input_features_idx.size(0), input_features_idx.size(1)
        flat_idx = input_features_idx.reshape(-1)
        input_vectors = all_node_features.index_select(0, flat_idx).reshape(bsz, seq_len, -1)

        return self.decode(graph_features, input_vectors, subnode_ids, token_mask_len, spd_indices)


def export_model(config_path: str, checkpoint_path: str, output_name: str = "model_scripted.pt"):
    print("Loading configuration...")
    args = load_model_args(config_path)

    if not hasattr(args.encoder_config, "rwse_dim"):
        args.encoder_config.rwse_dim = 20
    if not hasattr(args.decoder_config, "max_spd"):
        args.decoder_config.max_spd = 20
    args.encoder_config.fixed_input_dim = getattr(args.encoder_config, "fixed_input_dim", 10000)

    print("Initializing model...")
    device = torch.device("cpu")
    model = GraphDecoder(args).to(device)

    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    new_state_dict = {}
    for k, v in state_dict.items():
        new_state_dict[k[7:]] = v if k.startswith("module.") else v
    # correct strip
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict, strict=False)
    model.eval()

    print("Wrapping model (encode/decode + getters)...")
    wrapper = ExportModule(model).eval()

    print("Scripting model with TorchScript...")
    scripted = torch.jit.script(wrapper)
    scripted.save(output_name)
    print(f"Success! Model saved to {output_name}")


if __name__ == "__main__":
    CONFIG_DIR = "/home/hujiayue/my_model_without_dgl"
    CKPT_FILE = "/home/hujiayue/my_model_without_dgl/checkpoints/wikics/gin_checkpoint.pth"
    export_model(CONFIG_DIR, CKPT_FILE, "model_scripted.pt")