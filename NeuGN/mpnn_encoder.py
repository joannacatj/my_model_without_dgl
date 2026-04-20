import torch
from torch import nn
import torch.nn.functional as F

class PureGraphConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        
    def forward(self, x, edge_index):
        # Full GCN layer: H' = D^{-1/2} * (A + I) * D^{-1/2} * X * W
        # edge_index: [2, E]
        num_nodes = x.size(0)
        device = x.device

        # Add self-loops
        self_loops = torch.arange(num_nodes, device=device)
        self_loops = self_loops.unsqueeze(0).repeat(2, 1)
        edge_index = torch.cat([edge_index, self_loops], dim=1)
        src, dst = edge_index

        # Compute degree for normalization
        deg = torch.zeros(num_nodes, device=device)
        deg.index_add_(0, dst, torch.ones(src.size(0), device=device))
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg == 0] = 0

        # Message passing with symmetric normalization
        norm = deg_inv_sqrt[src] * deg_inv_sqrt[dst]
        out = torch.zeros_like(x)
        out.index_add_(0, dst, x[src] * norm.unsqueeze(1))

        # Linear transform
        return self.linear(out)

class PureGINConv(nn.Module):
    def __init__(self, mlp, learn_eps=False):
        super().__init__()
        self.mlp = mlp
        self.eps = nn.Parameter(torch.Tensor([0])) if learn_eps else 0.0
        
    def forward(self, x, edge_index):
        # GIN: MLP((1+eps) * x + Sum(Neighbors))
        src, dst = edge_index
        
        # 1. Aggregate (Sum)
        out = torch.zeros_like(x)
        out.index_add_(0, dst, x[src])
        
        # 2. Combine
        out = out + (1 + self.eps) * x
        
        # 3. MLP
        return self.mlp(out)

class GNN(nn.Module):
    def __init__(self, params):
        super(GNN, self).__init__()
        self.in_dim = params.encoder_config.graph_feature_dim
        self.out_dim = params.decoder_config.dim
        self.num_layers = params.encoder_config.encoder_layers
        self.fixed_input_dim = getattr(params.encoder_config, 'fixed_input_dim', 10000)
        self.encoder_name = params.encoder_config.encoder_name.lower()
        
        # [新增] RWSE 配置
        self.rwse_dim = getattr(params.encoder_config, 'rwse_dim', 0)
        
        def make_mlp(in_dim, out_dim):
            return nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.ReLU(),
                nn.Linear(out_dim, out_dim)
            )

        self.value_projection = nn.Linear(self.fixed_input_dim, self.in_dim)
        self.degree_embedding = nn.Embedding(1001, self.in_dim)
        
        # [新增] RWSE 投影层
        if self.rwse_dim > 0:
            self.rwse_projection = nn.Linear(self.rwse_dim, self.in_dim)
        
        self.input_noise_scale = 0.01 

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList() 
        
        if self.encoder_name == 'gin':
            self.convs.append(PureGINConv(make_mlp(self.in_dim, self.out_dim)))
        else:
            self.convs.append(PureGraphConv(self.in_dim, self.out_dim))
        self.batch_norms.append(nn.BatchNorm1d(self.out_dim))
        
        for _ in range(1, self.num_layers):
            if self.encoder_name == 'gin':
                self.convs.append(PureGINConv(make_mlp(self.out_dim, self.out_dim)))
            else:
                self.convs.append(PureGraphConv(self.out_dim, self.out_dim))
            self.batch_norms.append(nn.BatchNorm1d(self.out_dim))
        
    def global_mean_pool(self, x, batch):
        if batch is None: return x.mean(dim=0, keepdim=True)
        num_graphs = batch.max().item() + 1
        out = torch.zeros(num_graphs, x.size(1), device=x.device)
        out.index_add_(0, batch, x)
        count = torch.zeros(num_graphs, 1, device=x.device)
        count.index_add_(0, batch, torch.ones(x.size(0), 1, device=x.device))
        return out / count.clamp(min=1)

    def forward(self, batched_graph):
        h_id = batched_graph.ndata['feat_id']
        edge_index = batched_graph.edge_index
        batch_idx = getattr(batched_graph, 'batch', None)
        
        h_id_mapped = h_id % self.fixed_input_dim
        h_one_hot = F.one_hot(h_id_mapped, num_classes=self.fixed_input_dim).float()
        h_attr = self.value_projection(h_one_hot)
        
        # [新增] RWSE 注入
        if self.rwse_dim > 0 and 'rwse' in batched_graph.ndata:
            h_rwse = self.rwse_projection(batched_graph.ndata['rwse'])
            h_attr = h_attr + h_rwse

        if self.training:
            noise = torch.randn_like(h_attr) * self.input_noise_scale
            h_attr = h_attr + noise

        in_degrees = batched_graph.in_degrees()
        in_degrees = torch.clamp(in_degrees, 0, 1000)
        h_deg = self.degree_embedding(in_degrees)
        
        h = h_attr + h_deg
        
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index)
            h = self.batch_norms[i](h)
            h = F.relu(h)
            
        graph_feature = self.global_mean_pool(h, batch_idx)
        return graph_feature, h
