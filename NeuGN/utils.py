import os
import csv
import yaml
import random
import torch
import numpy as np
import pandas as pd
import torch.distributed as dist
from box import Box
import networkx as nx
from typing import List, Tuple
import torch.nn.functional as F

# ==========================================
# 轻量级图结构 (SimpleGraph)
# ==========================================
class SimpleGraph:
    def __init__(self, edge_index, num_nodes, node_feat=None, nid=None, rwse=None):
        """
        edge_index: [2, E] tensor (src, dst)
        num_nodes: int
        node_feat: [N, D] tensor or None (feat_id)
        nid: [N] tensor (原始节点ID)
        rwse: [N, K] tensor (结构特征)
        """
        self.edge_index = edge_index
        self.num_nodes = num_nodes
        self.ndata = {}
        if node_feat is not None:
            self.ndata['feat_id'] = node_feat
        if nid is not None:
            self.ndata['nid'] = nid
        else:
            self.ndata['nid'] = torch.arange(num_nodes, dtype=torch.long)
        
        # [新增] 存储 RWSE 特征
        if rwse is not None:
            self.ndata['rwse'] = rwse
            
    def in_degrees(self):
        dst = self.edge_index[1]
        deg = torch.bincount(dst, minlength=self.num_nodes)
        return deg

    def to(self, device):
        self.edge_index = self.edge_index.to(device)
        for k, v in self.ndata.items():
            if isinstance(v, torch.Tensor):
                self.ndata[k] = v.to(device)
        
        if hasattr(self, 'batch') and isinstance(self.batch, torch.Tensor):
            self.batch = self.batch.to(device)
        return self

# ==========================================
# [新增] 特征计算函数
# ==========================================
def compute_rwse(edge_index, num_nodes, k_steps=20):
    """计算随机游走结构编码 (RWSE)"""
    # 构造邻接矩阵 (Dense for simplicity on subgraphs)
    adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float)
    src, dst = edge_index
    # 无向图处理：确保双向
    adj[src, dst] = 1.0
    adj[dst, src] = 1.0 
    
    # 计算度矩阵 D^-1
    deg = adj.sum(dim=1)
    deg_inv = deg.pow(-1)
    deg_inv[deg_inv == float('inf')] = 0
    P = torch.matmul(torch.diag(deg_inv), adj) # Random Walk Matrix P = D^-1 A
    
    rwse_list = []
    Pk = P.clone()
    
    # P^1 对角线
    rwse_list.append(torch.diagonal(Pk))
    
    for _ in range(k_steps - 1):
        Pk = torch.matmul(Pk, P)
        rwse_list.append(torch.diagonal(Pk))
        
    rwse_embed = torch.stack(rwse_list, dim=1) # [N, k_steps]
    return rwse_embed

def compute_subgraph_spd(edge_index, num_nodes, max_dist=20):
    """
    计算子图内所有节点对的最短路径距离 (Floyd-Warshall 或 BFS)
    考虑到子图较小 (num_nodes ~32-64), 使用 NetworkX 计算最方便且足够快
    """
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    edge_list = edge_index.t().tolist()
    G.add_edges_from(edge_list)
    
    # 计算所有对的最短路径
    length = dict(nx.all_pairs_shortest_path_length(G))
    
    # 填充矩阵
    spd_matrix = torch.full((num_nodes, num_nodes), max_dist + 1, dtype=torch.long)
    for i in range(num_nodes):
        spd_matrix[i, i] = 0
        if i in length:
            for j, dist in length[i].items():
                if dist <= max_dist:
                    spd_matrix[i, j] = dist
                else:
                    spd_matrix[i, j] = max_dist + 1 # 截断
                    
    return spd_matrix

# ==========================================
# 辅助函数 (保持不变或微调)
# ==========================================
def graph2path_v2_pure(graph: SimpleGraph) -> List[Tuple[int, int]]:
    edge_list = graph.edge_index.t().cpu().tolist()
    G = nx.Graph()
    G.add_nodes_from(range(graph.num_nodes))
    G.add_edges_from(edge_list)
    
    if not nx.is_connected(G):
        S = [G.subgraph(c).copy() for c in nx.connected_components(G)]
    else:
        S = [G]
    
    random.shuffle(S)
    s = S[0]
    path = connected_graph2path(s)
    
    if len(S) > 1:
        prev_connect_node = list(s.nodes)[0] if len(path) == 0 else path[-1][-1]
        for s in S[1:]:
            spath = connected_graph2path(s)
            curr_connect_node = list(s.nodes)[0] if len(spath) == 0 else spath[0][0]
            jump_edge = (prev_connect_node, curr_connect_node)
            path.append(jump_edge)
            path.extend(spath)
            prev_connect_node = path[-1][-1]
    return path

def connected_graph2path(G) -> List[Tuple[int, int]]:
    if len(G.nodes) <= 1:
        path = []
    else:
        if not nx.is_eulerian(G):
            G = nx.eulerize(G)
        try:
            # fix for networkx versions dealing with source in eulerian path
            node = random.choice(list(G.nodes()))
            if random.random() < 0.5:
                raw_path = list(nx.eulerian_path(G, source=node))
            else:
                raw_path = list(nx.eulerian_circuit(G, source=node))
        except:
             # Fallback
             return []

        triangle_path = [(src, tgt) if src < tgt else (tgt, src) for src, tgt in raw_path]
        unique_edges = set(triangle_path)
        idx = len(raw_path)
        for i in range(1, len(raw_path) + 1):
            short_path = triangle_path[:i]
            if set(short_path) == unique_edges:
                idx = i
                break
        path = raw_path[:idx]
    return path

def simple_node_subgraph(graph: SimpleGraph, nodes: List[int]):
    nodes_tensor = torch.tensor(nodes, dtype=torch.long, device=graph.edge_index.device)
    subset_mask = torch.zeros(graph.num_nodes, dtype=torch.bool, device=graph.edge_index.device)
    subset_mask[nodes_tensor] = True
    
    src, dst = graph.edge_index
    edge_mask = subset_mask[src] & subset_mask[dst]
    new_src = src[edge_mask]
    new_dst = dst[edge_mask]
    
    node_map = torch.full((graph.num_nodes,), -1, dtype=torch.long, device=graph.edge_index.device)
    node_map[nodes_tensor] = torch.arange(len(nodes), device=graph.edge_index.device)
    
    mapped_src = node_map[new_src]
    mapped_dst = node_map[new_dst]
    new_edge_index = torch.stack([mapped_src, mapped_dst], dim=0)
    
    new_node_feat = None
    if 'feat_id' in graph.ndata:
        new_node_feat = graph.ndata['feat_id'][nodes_tensor]
        
    old_nid = graph.ndata['nid'][nodes_tensor]
    
    return SimpleGraph(new_edge_index, len(nodes), new_node_feat, nid=old_nid)

def simple_batch(graphs: List[SimpleGraph]):
    if len(graphs) == 0: return None
    total_nodes = 0
    edge_indices = []
    node_feats = []
    nids = []
    rwse_feats = [] # [新增]
    batch_index = [] 
    
    for i, g in enumerate(graphs):
        num_nodes = g.num_nodes
        edge_indices.append(g.edge_index + total_nodes)
        
        if 'feat_id' in g.ndata: node_feats.append(g.ndata['feat_id'])
        if 'nid' in g.ndata: nids.append(g.ndata['nid'])
        if 'rwse' in g.ndata: rwse_feats.append(g.ndata['rwse']) # [新增]
            
        batch_index.append(torch.full((num_nodes,), i, dtype=torch.long, device=g.edge_index.device))
        total_nodes += num_nodes
        
    batched_edge_index = torch.cat(edge_indices, dim=1) if edge_indices else torch.empty((2,0), dtype=torch.long)
    batched_feat = torch.cat(node_feats, dim=0) if node_feats else None
    batched_nid = torch.cat(nids, dim=0) if nids else None
    batched_rwse = torch.cat(rwse_feats, dim=0) if rwse_feats else None # [新增]
    batched_batch_idx = torch.cat(batch_index, dim=0)
    
    batched_graph = SimpleGraph(batched_edge_index, total_nodes, batched_feat, nid=batched_nid, rwse=batched_rwse)
    batched_graph.batch = batched_batch_idx
    batched_graph.batch_num_nodes = [g.num_nodes for g in graphs]
    
    return batched_graph

# ==========================================
# 数据处理核心 (Modified)
# ==========================================
def process_input_data(graph, graph_tokenizer, start_nodes, graph_len, walk_len, epoch, params):
    num_negatives = getattr(params.dataprocess_config, 'num_negatives', 64)
    # [新增] 参数读取
    rwse_dim = getattr(params.encoder_config, 'rwse_dim', 0)
    max_spd = getattr(params.decoder_config, 'max_spd', 20)
    
    valid_walks_global = [] 
    max_walk_len = torch.max(graph_len).item()
    walks = graph_tokenizer.random_walks(start_nodes, max_walk_len)
    
    for i, walk in enumerate(walks):
        current_len = graph_len[i].item()
        if len(walk) >= current_len:
            valid_walks_global.append(walk[:current_len])

    if not valid_walks_global: valid_walks_global = [[0, 1]] 

    sub_graphs = []
    spd_bias_list_unpadded = [] # [新增] 用于存储序列的 SPD Bias
    
    for walk in valid_walks_global:
        # 1. 提取子图
        sg = simple_node_subgraph(graph, list(set(walk)))
        
        # 2. [新增] 计算子图 RWSE
        if rwse_dim > 0:
            rwse = compute_rwse(sg.edge_index, sg.num_nodes, k_steps=rwse_dim)
            sg.ndata['rwse'] = rwse
        
        # 3. [新增] 计算子图 SPD 矩阵 (用于后续 Decoder Bias)
        # 注意：这里计算的是子图内所有节点的 SPD
        sg_full_spd = compute_subgraph_spd(sg.edge_index, sg.num_nodes, max_dist=max_spd)
        sg.full_spd = sg_full_spd # 暂存
        
        sub_graphs.append(sg)

    batch_graphs = simple_batch(sub_graphs)
    subgraph_node_counts = batch_graphs.batch_num_nodes
    batch_offsets = np.cumsum([0] + subgraph_node_counts)

    input_batch_indices_list = []
    target_batch_indices_list = []
    neg_batch_indices_list = [] 
    input_subnodes_list = []
    spd_seq_list = [] # [新增]
    
    max_seq_len = 0
    sub_node_id_mod = params.decoder_config.sub_node_id_size

    for i, sub_graph in enumerate(sub_graphs):
        offset = batch_offsets[i]
        path_tuples = graph2path_v2_pure(sub_graph)
        
        path_local_ids = []
        if len(path_tuples) > 0:
            for edge in path_tuples: path_local_ids.append(edge[0])
            path_local_ids.append(path_tuples[-1][-1])
        else:
            path_local_ids = [0]
            
        path_len = len(path_local_ids)
        if path_len < 2:
            input_local = []
            target_local = path_local_ids[0]
        else:
            cut_idx = random.randint(0, path_len - 1)
            input_local = path_local_ids[:cut_idx]
            target_local = path_local_ids[cut_idx]
            
        # [新增] 构建 SPD Bias for Sequence
        # input_local 是序列中节点在子图中的 local index
        # 我们需要 input_local 之间的 pairwise distance
        if len(input_local) > 0:
            local_indices_tensor = torch.tensor(input_local, dtype=torch.long)
            # Gather row/cols from full_spd
            # shape: [Seq_Len, Seq_Len]
            # grid_x, grid_y = torch.meshgrid(local_indices_tensor, local_indices_tensor, indexing='ij')
            # seq_spd = sub_graph.full_spd[grid_x, grid_y] 
            seq_spd = sub_graph.full_spd[local_indices_tensor][:, local_indices_tensor]
        else:
            seq_spd = torch.empty((0,0), dtype=torch.long)
        spd_seq_list.append(seq_spd)

        all_subgraph_nodes = list(range(sub_graph.num_nodes))
        candidates = [n for n in all_subgraph_nodes if n != target_local]
        
        neg_locals = []
        if len(candidates) > 0:
            while len(neg_locals) < num_negatives:
                neg_locals.extend(candidates)
            random.shuffle(neg_locals)
            neg_locals = neg_locals[:num_negatives]
        else:
            neg_locals = [0] * num_negatives

        input_batch = [lid + offset for lid in input_local]
        target_batch = target_local + offset
        neg_batch = [lid + offset for lid in neg_locals]
        
        global_ids_map = sub_graph.ndata['nid'].tolist()
        input_global_ids = [global_ids_map[lid] for lid in input_local]
        rand_start = random.randint(0, sub_node_id_mod - 1)
        subnode_seq = [((gid + rand_start) % sub_node_id_mod) for gid in input_global_ids]
        
        input_batch_indices_list.append(input_batch)
        target_batch_indices_list.append(target_batch)
        neg_batch_indices_list.append(neg_batch)
        input_subnodes_list.append(subnode_seq)
        
        if len(input_batch) > max_seq_len: max_seq_len = len(input_batch)

    # Padding
    padded_input_indices = []
    padded_subnodes = []
    padded_spd_bias = [] # [新增]
    token_len_list = []

    for idx in range(len(input_batch_indices_list)):
        seq = input_batch_indices_list[idx]
        sub = input_subnodes_list[idx]
        spd = spd_seq_list[idx] # [L, L]
        
        curr_len = len(seq)
        token_len_list.append(curr_len)
        pad_len = max_seq_len - curr_len
        
        padded_input_indices.append(seq + [0] * pad_len)
        padded_subnodes.append(sub + [0] * pad_len)
        
        # Pad SPD Matrix [Max, Max]
        # 初始化为 Max_Dist + 1 (Masked/Far)
        # 注意: 序列长度会 + 2 (SOS, GraphToken)，这里只处理 Node Sequence
        # SOS 和 GraphToken 的处理在 Model 内部动态添加
        full_padded_spd = torch.full((max_seq_len, max_seq_len), max_spd + 1, dtype=torch.long)
        if curr_len > 0:
            full_padded_spd[:curr_len, :curr_len] = spd
        padded_spd_bias.append(full_padded_spd)

    input_indices = torch.tensor(padded_input_indices, dtype=torch.long)
    input_subnode_seqs = torch.tensor(padded_subnodes, dtype=torch.long)
    targets = torch.tensor(target_batch_indices_list, dtype=torch.long)
    negatives = torch.tensor(neg_batch_indices_list, dtype=torch.long) 
    token_mask_len_seq = torch.tensor(token_len_list, dtype=torch.long)
    spd_bias_tensor = torch.stack(padded_spd_bias) # [B, Max_Seq, Max_Seq]
    
    return batch_graphs, input_indices, input_subnode_seqs, targets, negatives, token_mask_len_seq, spd_bias_tensor

# ==========================================
# 保持原有的 IO 函数不变
# ==========================================
def load_ori_graph(path, name):
    nodes, node_values, src_ids, dst_ids = [], [], [], []
    file_path = os.path.join(path, f'{name}.graph')
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Graph file not found: {file_path}")
    with open(file_path, "r") as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip()
            if not line: continue
            if line.startswith("v"):
                parts = line.split()
                nodes.append(int(parts[1]))
                node_values.append(int(parts[2]))
            elif line.startswith("e"):
                parts = line.split()
                src_ids.append(int(parts[1]))
                dst_ids.append(int(parts[2]))
    edge_src_ids = src_ids + dst_ids
    edge_dst_ids = dst_ids + src_ids
    node_values_uni = set(node_values)
    return node_values, node_values_uni, edge_src_ids, edge_dst_ids

def load_amazon(path):
    edge_path = os.path.join(path, 'new_edges.csv')
    node_path = os.path.join(path, 'new_nodes.csv')
    edge_data = pd.read_csv(edge_path)
    node_data = pd.read_csv(node_path)
    node_values = node_data['attribute'].tolist()
    node_value_uni = set(node_values)
    src = edge_data['node1_id'].tolist()
    dst = edge_data['node2_id'].tolist()
    edge_src_ids = src + dst
    edge_dst_ids = dst + src
    return node_values, node_value_uni, edge_src_ids, edge_dst_ids

def save_value2id(value_set, path, name):
    value2id = {str(element): idx for idx, element in enumerate(sorted(value_set))}
    file_path = os.path.join(path, f'{name}_value2id_mapping.csv')
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Element", "ID"]) 
        for element, idx in value2id.items():
            writer.writerow([element, idx]) 
    return value2id
    
def load_value2id(path, name=None):
    value2id = {}
    if name is None:
        dataset_list = ['lastfm', 'hamster', 'nell', 'wikics', 'amazon']
        for dname in dataset_list:
            if dname in path:
                name = dname
                break
    if name is None: name = 'default'
    file_path = os.path.join(path, f'{name}_value2id_mapping.csv')
    if not os.path.exists(file_path): return {}
    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)
        for line in reader:
            if line: value2id[line[0]] = int(line[1])
    return value2id

def load_model_args(config_path: str):
    config_file = os.path.join(config_path, 'model_args.yaml')
    with open(config_file, 'r') as f:
        args_dict = yaml.safe_load(f)
    return Box(args_dict)

def load_checkpoint(filepath, model, optimizer, scheduler, rank):
    if dist.is_initialized():
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
    else:
        map_location = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state = torch.load(filepath, map_location=map_location)

    if hasattr(model, 'module'):
        model_to_load = model.module
    else:
        model_to_load = model

    try:
        model_to_load.load_state_dict(state['model_state_dict'])
    except RuntimeError:
        new_state_dict = {}
        for k, v in state['model_state_dict'].items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        model_to_load.load_state_dict(new_state_dict)

    optimizer.load_state_dict(state['optimizer_state_dict'])
    scheduler.load_state_dict(state['scheduler_state_dict'])
    if dist.is_initialized(): dist.barrier()
    return state['epoch'] + 1

def save_checkpoint(model, optimizer, scheduler, epoch, filepath):
    should_save = False
    if not dist.is_initialized():
        should_save = True
    elif dist.get_rank() == 0:
        should_save = True
        
    if should_save:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        if hasattr(model, 'module'):
            model_to_save = model.module
        else:
            model_to_save = model
        state = {
            'epoch': epoch,
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(), 
        }
        torch.save(state, filepath)

def metrics(ranks: List[int]):
    ranks = np.array(ranks)
    if len(ranks) == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    acc1 = (ranks <= 1).mean()
    acc10 = (ranks <= 10).mean()
    acc20 = (ranks <= 20).mean()
    acc50 = (ranks <= 50).mean()
    acc100 = (ranks <= 100).mean()
    mrr = (1.0 / ranks).mean()
    ndcg = (1.0 / np.log2(ranks + 1)).mean()
    return acc1, acc10, acc20, acc50, acc100, mrr, ndcg