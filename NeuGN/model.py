import math
from dataclasses import dataclass
from typing import Optional, Tuple
from NeuGN.mpnn_encoder import GNN
import torch
import torch.nn.functional as F
from torch import nn

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )

class Attention(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.n_kv_heads = args.decoder_config.n_heads if args.decoder_config.n_kv_heads is None else args.decoder_config.n_kv_heads
        model_parallel_size = 1
        self.n_local_heads = args.decoder_config.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.decoder_config.dim // args.decoder_config.n_heads
        self.wq = nn.Linear(
            args.decoder_config.dim,
            args.decoder_config.n_heads * self.head_dim,
            bias=False
        )
        self.wk = nn.Linear(
            args.decoder_config.dim,
            self.n_kv_heads * self.head_dim,
            bias=False
        )
        self.wv = nn.Linear(
            args.decoder_config.dim,
            self.n_kv_heads * self.head_dim,
            bias=False
        )
        self.wo = nn.Linear(
            args.decoder_config.n_heads * self.head_dim,
            args.decoder_config.dim,
            bias=False
        )

    # [修改] 增加 graph_bias 参数
    def forward(self, x: torch.Tensor, start_pos: int, mask: Optional[torch.Tensor], graph_bias: Optional[torch.Tensor] = None):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        keys = xk
        values = xv

        keys = repeat_kv(keys, self.n_rep)
        values = repeat_kv(values, self.n_rep)

        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask
            
        # [新增] 应用 Graph Bias
        if graph_bias is not None:
            # graph_bias: [B, n_heads, Seq, Seq] or broadcastable
            # Ensure dims match
            scores = scores + graph_bias
            
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)

class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, ffn_dim_multiplier: Optional[float]):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args):
        super().__init__()
        self.n_heads = args.decoder_config.n_heads
        self.dim = args.decoder_config.dim
        self.head_dim = args.decoder_config.dim // args.decoder_config.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.decoder_config.dim,
            hidden_dim=4 * args.decoder_config.dim,
            multiple_of=args.decoder_config.multiple_of,
            ffn_dim_multiplier=args.decoder_config.ffn_dim_multiplier,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.decoder_config.dim, eps=args.decoder_config.norm_eps)
        self.ffn_norm = RMSNorm(args.decoder_config.dim, eps=args.decoder_config.norm_eps)

    def forward(self, x: torch.Tensor, start_pos: int, mask: Optional[torch.Tensor], graph_bias: Optional[torch.Tensor] = None):
        h = x + self.attention(self.attention_norm(x), start_pos, mask, graph_bias)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]
    
class NodeIdEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        ne = torch.empty(max_len, d_model)
        torch.nn.init.orthogonal_(ne)
        ne.require_grad = False
        self.register_buffer('ne', ne)  
        self.num_embeddings = max_len 
        self.embedding_dim = d_model   

    def forward(self, node_ids):
        return self.ne[node_ids]

class Transformer(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.n_layers = params.decoder_config.n_layers
        self.subnodeid_size = params.decoder_config.sub_node_id_size
        
        self.feature_dim = params.encoder_config.graph_feature_dim 
        self.hidden_dim = params.decoder_config.dim

        self.input_projection = nn.Linear(self.feature_dim, self.hidden_dim)
        
        self.sos_token = nn.Parameter(torch.randn(1, 1, self.hidden_dim))

        self.node_embeddings = NodeIdEmbedding(
             params.decoder_config.dim, params.decoder_config.sub_node_id_size,
        )

        self.pos_embeddings = PositionalEmbedding(
             params.decoder_config.dim, params.decoder_config.pos_size,
        )

        self.type_embeddings = nn.Embedding(2, params.decoder_config.dim)
        
        # [新增] SPD Bias Embedding
        # max_spd + 1 (for masked/far) + 1 (for special tokens like SOS/GraphToken)
        self.max_spd = getattr(params.decoder_config, 'max_spd', 20)
        self.spd_bias_embedding = nn.Embedding(self.max_spd + 2, params.decoder_config.n_heads)

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.decoder_config.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.decoder_config.dim, eps=params.decoder_config.norm_eps)

        self.output = nn.Sequential(
            nn.Linear(params.decoder_config.dim, params.decoder_config.dim),
            nn.GELU(),
            nn.Linear(params.decoder_config.dim, self.feature_dim),
        )

    def forward(self, graph_features: torch.Tensor, input_features: torch.Tensor, subnode_ids: torch.Tensor, 
                token_mask_len: torch.Tensor, start_pos: int, spd_indices: Optional[torch.Tensor] = None):
        
        # input_features: [Batch, Seq_Len, Feature_Dim]
        if input_features is not None:
            _bsz = input_features.shape[0]
            
            h_data = self.input_projection(input_features) 
            sos_emb = self.sos_token.expand(_bsz, -1, -1) 
            h_tokens = torch.cat([sos_emb, h_data], dim=1) 
            
            sos_subnode_id = torch.zeros((_bsz, 1), dtype=subnode_ids.dtype, device=subnode_ids.device)
            subnode_ids_extended = torch.cat([sos_subnode_id, subnode_ids], dim=1)
            
            h_tokens_subnode = self.node_embeddings(subnode_ids_extended)
            token_types = torch.zeros((_bsz, h_tokens.shape[1]), dtype=torch.long, device=input_features.device)
            h_tokens = h_tokens + self.type_embeddings(token_types) + h_tokens_subnode
            
            graph_feature_types = torch.ones(graph_features.size(0), graph_features.size(1), dtype=torch.long, device=graph_features.device)
            h_graph = graph_features + self.type_embeddings(graph_feature_types)
            
            # Final Sequence: [Graph, SOS, Node1, Node2...]
            h = torch.cat((h_graph, h_tokens), dim=1)
            seqlen = h.shape[1]
            
            # [新增] 构建 Graph Bias Matrix
            # spd_indices: [B, Node_Seq, Node_Seq]
            # 我们需要构建 Full Bias: [B, Full_Seq, Full_Seq]
            # Full_Seq = 1 (Graph) + 1 (SOS) + Node_Seq
            graph_bias = None
            if spd_indices is not None:
                # 扩展 SPD indices 矩阵以包含 Graph 和 SOS Token
                # 假设: 
                #   Graph/SOS <-> Node: 距离设为特殊值 (max_spd + 1)
                #   Graph <-> SOS: 距离设为 0 或特殊值
                
                device = spd_indices.device
                node_seq_len = spd_indices.shape[1]
                full_len = node_seq_len + 2
                
                # 初始化为特殊值 (max_spd + 1)
                full_spd_indices = torch.full((_bsz, full_len, full_len), self.max_spd + 1, dtype=torch.long, device=device)
                
                # 填入 Node-Node SPD
                full_spd_indices[:, 2:, 2:] = spd_indices
                
                # 可以自定义特殊 Token 的连接性，这里简单处理为 "远距离" 或 "自环"
                # Graph Token self-loop
                full_spd_indices[:, 0, 0] = 0
                # SOS Token self-loop
                full_spd_indices[:, 1, 1] = 0
                
                # 查找 Bias 值 -> [B, Full, Full, Heads]
                bias_vals = self.spd_bias_embedding(full_spd_indices)
                # Permute to [B, Heads, Full, Full]
                graph_bias = bias_vals.permute(0, 3, 1, 2)

        else:
            h = graph_features + self.type_embeddings(torch.ones(graph_features.size(0), graph_features.size(1), dtype=torch.long, device=graph_features.device))
            seqlen = h.shape[1]
            _bsz = graph_features.shape[0]
            graph_bias = None

        h = h + self.pos_embeddings(h)

        positions = torch.arange(seqlen, device=graph_features.device).unsqueeze(0)
        positions_repeated = positions.repeat(_bsz, 1) 
        token_mask_len_uns = token_mask_len.unsqueeze(1)
        # 注意 +2 因为 graph_feature(1) + sos(1)
        valid_mask = positions_repeated < (token_mask_len_uns + 1 + graph_features.shape[1]) 
        
        mask = torch.full_like(valid_mask, float('-inf'), device=graph_features.device).type_as(h)
        mask[valid_mask] = 0.0
        mask = mask.unsqueeze(2).repeat(1 , 1, seqlen).unsqueeze(1)

        for layer in self.layers:
            h = layer(h, start_pos, mask, graph_bias) # 传入 Bias
        h = self.norm(h)
        
        h_seq = h[:, graph_features.shape[1]:, :] 
        output = self.output(h_seq).float()
        return output

class GraphDecoder(nn.Module):
    def __init__(self, params):
        super(GraphDecoder, self).__init__()
        self.params = params 
        self.encoder_type = params.encoder_config.encoder_name.lower()
        self.decoder_type = params.decoder_config.decoder_type
        
        self.gnn_list = ['gat', 'gcn', 'gin']
        if self.encoder_type in self.gnn_list:
            self.encoder = GNN(params)
            
        if self.decoder_type == 'llama':
            self.decoder = Transformer(params)
            
        self.dim = params.decoder_config.dim

    def forward(self, batch_graphs, input_features: torch.Tensor, subnode_ids: torch.Tensor, 
                token_mask_len: torch.Tensor, start_pos: int, device=None, spd_indices=None):
        if self.encoder_type in self.gnn_list:
            graph_features, all_node_features = self.encoder(batch_graphs)
            graph_features = graph_features.unsqueeze(dim=1)
        
        if self.decoder_type == 'llama':
            # [修改] 传入 spd_indices
            output = self.decoder(graph_features, input_features, subnode_ids, token_mask_len, start_pos, spd_indices)
            
        return output, all_node_features