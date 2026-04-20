import torch
import networkx as nx
import numpy as np
import os
import argparse
import random
import time
import sys
import datetime
from collections import Counter
import torch.nn.functional as F

from NeuGN.model import GraphDecoder
from NeuGN.graph_tokenizer import GraphTokenizer
from NeuGN.utils import (
    load_model_args, load_ori_graph, load_value2id, 
    graph2path_v2_pure, SimpleGraph, simple_batch,
    compute_rwse, compute_subgraph_spd, simple_node_subgraph
)

class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def load_inference_checkpoint(model, filepath, device):
    if not os.path.exists(filepath):
        print(f"Warning: Checkpoint not found at {filepath}")
        return False
    try:
        checkpoint = torch.load(filepath, map_location=device)
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
            
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict)
        
        epoch_info = checkpoint.get('epoch', '?') if isinstance(checkpoint, dict) else '?'
        print(f"Successfully loaded model from {filepath} (Epoch {epoch_info})")
        return True
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return False

# ================= 随机查询图生成器 =================
def generate_random_query(data_graph_simple, size):
    num_nodes = data_graph_simple.num_nodes
    edge_list = data_graph_simple.edge_index.t().tolist()
    g_nx = nx.Graph()
    g_nx.add_edges_from(edge_list)
    
    max_retries = 100
    for _ in range(max_retries):
        if num_nodes > 0:
            start_node = random.randint(0, num_nodes - 1)
        else:
            return None
            
        sub_nodes = set([start_node])
        queue = [start_node]
        
        while len(sub_nodes) < size and queue:
            curr = queue.pop(0)
            neighbors = list(g_nx.neighbors(curr))
            random.shuffle(neighbors)
            
            for n in neighbors:
                if n not in sub_nodes:
                    sub_nodes.add(n)
                    queue.append(n)
                    if len(sub_nodes) == size:
                        break
        
        if len(sub_nodes) == size:
            sub_nodes_list = list(sub_nodes)
            q_g = simple_node_subgraph(data_graph_simple, sub_nodes_list)
            return q_g
            
    return None

# ================= NeuGN Inference Wrapper =================
class NeuGNInferenceWrapper:
    def __init__(self, model, data_node_features, tokenizer, device, params):
        self.model = model
        self.data_node_features = data_node_features 
        self.tokenizer = tokenizer 
        self.device = device
        self.params = params
        
        self.rwse_dim = getattr(params.encoder_config, 'rwse_dim', 0)
        self.max_spd = getattr(params.decoder_config, 'max_spd', 20)
        
        self.current_query_path = None
        self.current_query_feat = None 
        self.current_query_graph = None
        self.current_query_spd = None 
        self.query_node_to_path_indices = {}

    def set_query_graph(self, query_graph):
        # 计算特征的时间不计入匹配时间，通常这可以在预处理阶段完成
        if self.rwse_dim > 0:
            rwse = compute_rwse(query_graph.edge_index, query_graph.num_nodes, k_steps=self.rwse_dim)
            query_graph.ndata['rwse'] = rwse
            
        full_spd = compute_subgraph_spd(query_graph.edge_index, query_graph.num_nodes, max_dist=self.max_spd)
        self.current_query_spd = full_spd.to(self.device)
        
        self.current_query_graph = query_graph.to(self.device)
        
        if not hasattr(self.current_query_graph, 'batch'):
            self.current_query_graph.batch = torch.zeros(self.current_query_graph.num_nodes, dtype=torch.long, device=self.device)
            
        with torch.no_grad():
            h_Q, _ = self.model.encoder(self.current_query_graph)
            self.current_query_feat = h_Q.unsqueeze(1) 

        path_edges = graph2path_v2_pure(query_graph) 
        path_nodes = []
        if len(path_edges) > 0:
            for u, v in path_edges:
                path_nodes.append(u)
            path_nodes.append(path_edges[-1][1])
        else:
            path_nodes = list(range(query_graph.num_nodes))

        self.current_query_path = path_nodes
        self.query_node_to_path_indices = {}
        for idx, node in enumerate(path_nodes):
            node = int(node)
            if node not in self.query_node_to_path_indices:
                self.query_node_to_path_indices[node] = []
            self.query_node_to_path_indices[node].append(idx)

    def predict_scores(self, target_query_node, partial_mapping, candidates):
        if target_query_node not in self.query_node_to_path_indices:
            return {c: random.random() for c in candidates}
            
        target_idx = self.query_node_to_path_indices[target_query_node][0]
        input_path_nodes = self.current_query_path[:target_idx]
        
        input_vectors_list = []
        if len(input_path_nodes) == 0:
            input_features = torch.zeros(1, 0, self.data_node_features.shape[1], device=self.device)
            seq_spd_indices = None
        else:
            for q_node in input_path_nodes:
                q_node = int(q_node)
                if q_node in partial_mapping:
                    data_node = partial_mapping[q_node]
                    input_vectors_list.append(self.data_node_features[data_node])
                else:
                    input_vectors_list.append(torch.zeros(self.data_node_features.shape[1], device=self.device))
            
            input_features = torch.stack(input_vectors_list).unsqueeze(0)
            
            indices_tensor = torch.tensor(input_path_nodes, dtype=torch.long, device=self.device)
            seq_spd_indices = self.current_query_spd[indices_tensor][:, indices_tensor].unsqueeze(0)

        sub_node_size = self.params.decoder_config.sub_node_id_size
        current_seq_len = len(input_path_nodes)
        subnode_seq = [(i % sub_node_size) for i in range(current_seq_len)]
        subnodes_tensor = torch.tensor([subnode_seq], dtype=torch.long).to(self.device)
        token_mask_len = torch.tensor([current_seq_len], dtype=torch.long).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.decoder(
                self.current_query_feat, 
                input_features,          
                subnodes_tensor,         
                token_mask_len, 
                start_pos=0,
                spd_indices=seq_spd_indices 
            )
            
        pred_vector = outputs[:, -1, :] 
        pred_norm = F.normalize(pred_vector, dim=-1)
        
        candidate_scores = {}
        if len(candidates) == 0:
            return candidate_scores

        cand_indices = torch.tensor(candidates, dtype=torch.long, device=self.device)
        cand_vectors = self.data_node_features[cand_indices]
        cand_norm = F.normalize(cand_vectors, dim=-1)
        scores = torch.matmul(pred_norm, cand_norm.T).squeeze(0) 
        
        for i, cand in enumerate(candidates):
            candidate_scores[cand] = scores[i].item()
            
        return candidate_scores

# ================= General Matcher =================
class GeneralMatcher:
    def __init__(self, data_graph_simple, inference_wrapper, method_name="VF3", use_neugn=False, verify=False, data_label_freq=None):
        edge_list = data_graph_simple.edge_index.t().tolist()
        self.data_graph_nx = nx.Graph()
        self.data_graph_nx.add_edges_from(edge_list)
        if data_graph_simple.num_nodes > self.data_graph_nx.number_of_nodes():
            self.data_graph_nx.add_nodes_from(range(data_graph_simple.num_nodes))
            
        self.data_graph_simple = data_graph_simple
        self.wrapper = inference_wrapper
        self.method_name = method_name
        self.use_neugn = use_neugn
        self.verify = verify
        self.data_label_freq = data_label_freq 
        
        self.steps = 0
        self.found_matches = []
        self.mapping = {}

    def get_query_order(self, query_graph_nx, query_graph_simple):
        nodes = list(query_graph_nx.nodes())
        if self.method_name in ["VF3", "QSI"]:
            return sorted(nodes, key=lambda n: query_graph_nx.degree[n], reverse=True)
        elif self.method_name == "GQL":
            def gql_key(n):
                feat = self.wrapper.current_query_graph.ndata['feat_id'][n].item()
                freq = self.data_label_freq.get(feat, 999999)
                return (freq, -query_graph_nx.degree[n]) 
            return sorted(nodes, key=gql_key)
        else:
            return sorted(nodes, key=lambda n: query_graph_nx.degree[n], reverse=True)

    def get_candidates(self, query_node, query_graph_nx, current_mapping):
        candidates = []
        q_neighbors = list(query_graph_nx.neighbors(query_node))
        matched_q_neighbors = [n for n in q_neighbors if n in current_mapping]
        
        if matched_q_neighbors:
            potential_nodes = set(self.data_graph_nx.neighbors(current_mapping[matched_q_neighbors[0]]))
            for nbr in matched_q_neighbors[1:]:
                potential_nodes &= set(self.data_graph_nx.neighbors(current_mapping[nbr]))
            potential_nodes = list(potential_nodes)
        else:
            potential_nodes = list(self.data_graph_nx.nodes())

        q_feat = self.wrapper.current_query_graph.ndata['feat_id'][query_node].item()
        data_feats = self.data_graph_simple.ndata['feat_id']

        for u in potential_nodes:
            if u in current_mapping.values(): continue
            if self.data_graph_nx.degree[u] < query_graph_nx.degree[query_node]: continue
            
            if data_feats.is_cuda:
                u_feat = data_feats[u].item()
            else:
                u_feat = data_feats[u]
                
            if u_feat != q_feat: continue
            candidates.append(u)
        return candidates

    def run_match(self, query_graph_simple, find_all=False):
        self.steps = 0
        self.found_matches = []
        self.mapping = {}
        
        edge_list = query_graph_simple.edge_index.t().tolist()
        query_graph_nx = nx.Graph()
        query_graph_nx.add_edges_from(edge_list)
        if query_graph_simple.num_nodes > query_graph_nx.number_of_nodes():
            query_graph_nx.add_nodes_from(range(query_graph_simple.num_nodes))
        
        self.wrapper.set_query_graph(query_graph_simple)
        query_nodes_order = self.get_query_order(query_graph_nx, query_graph_simple)
        
        if find_all:
            self._match_all_recursive(query_nodes_order, 0, query_graph_nx)
        else:
            self._match_first_recursive(query_nodes_order, 0, query_graph_nx)
        return self.found_matches, self.steps

    def _match_first_recursive(self, query_nodes_order, idx, query_graph_nx):
        self.steps += 1
        if idx == len(query_nodes_order):
            self.found_matches.append(self.mapping.copy())
            return True 
        u = query_nodes_order[idx]
        candidates = self.get_candidates(u, query_graph_nx, self.mapping)
        if not candidates: return False
        if self.use_neugn and len(candidates) > 1:
            scores = self.wrapper.predict_scores(u, self.mapping, candidates)
            candidates.sort(key=lambda x: scores.get(x, 0), reverse=True)
        else:
            candidates.sort()
        for v in candidates:
            self.mapping[u] = v
            if self._match_first_recursive(query_nodes_order, idx + 1, query_graph_nx):
                return True
            del self.mapping[u]
        return False
    
    def _match_all_recursive(self, query_nodes_order, idx, query_graph_nx):
        self.steps += 1
        if idx == len(query_nodes_order):
            self.found_matches.append(self.mapping.copy())
            return False 
        u = query_nodes_order[idx]
        candidates = self.get_candidates(u, query_graph_nx, self.mapping)
        if not candidates: return False
        if self.use_neugn and len(candidates) > 1:
            scores = self.wrapper.predict_scores(u, self.mapping, candidates)
            candidates.sort(key=lambda x: scores.get(x, 0), reverse=True)
        else:
            candidates.sort()
        found_any = False
        for v in candidates:
            self.mapping[u] = v
            if self._match_all_recursive(query_nodes_order, idx + 1, query_graph_nx):
                found_any = True
            del self.mapping[u]
        return found_any

# ================= Main =================
def main(args):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"gen_query_results_{timestamp}.txt"
    sys.stdout = Logger(log_file)
    
    print(f"=== NeuGN Generated Query Demo: {timestamp} ===")
    print(f"Log: {log_file}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset_list = ['lastfm', 'hamster', 'nell', 'wikics', 'dblp', 'youtube']
    dataset_name = next((name for name in dataset_list if name in args.graph_path), 'wikics')
    print(f"Dataset: {dataset_name}")

    params = load_model_args(args.config_path)
    
    if not hasattr(params.encoder_config, 'rwse_dim'):
        params.encoder_config.rwse_dim = 20
    if not hasattr(params.decoder_config, 'max_spd'):
        params.decoder_config.max_spd = 20
    if not hasattr(params.encoder_config, 'fixed_input_dim'):
        params.encoder_config.fixed_input_dim = 10000
    params.encoder_config.graph_feature_dim = 512
    params.encoder_config.encoder_name = args.encoder_name 
    
    print("Loading Graph Data...")
    node_values, node_values_uni, edge_src_ids, edge_dst_ids = load_ori_graph(args.graph_path, dataset_name)
    print("Loading Mapping...")
    value2id = load_value2id(args.config_path, dataset_name)

    num_nodes = len(node_values)
    edge_index = torch.tensor([edge_src_ids, edge_dst_ids], dtype=torch.long)
    node_values_id = torch.tensor([value2id.get(str(v), 0) for v in node_values], dtype=torch.long)
    
    data_graph = SimpleGraph(edge_index, num_nodes, node_feat=node_values_id)
    
    if params.encoder_config.rwse_dim > 0:
        print(f"Computing RWSE (dim={params.encoder_config.rwse_dim}) for full graph...")
        rwse = compute_rwse(data_graph.edge_index, data_graph.num_nodes, k_steps=params.encoder_config.rwse_dim)
        data_graph.ndata['rwse'] = rwse
        print("RWSE Computed.")
        
    label_counts = Counter(node_values_id.tolist())
    tokenizer = GraphTokenizer(graph=data_graph)
    
    print("Loading Model...")
    model = GraphDecoder(params).to(device)
    
    if args.model_path:
        checkpoint_path = args.model_path
    else:
        checkpoint_path = os.path.join(params.checkpoint_path, dataset_name, f'{args.encoder_name}_checkpoint.pth')
    
    print(f"Target Checkpoint: {checkpoint_path}")
    
    if load_inference_checkpoint(model, checkpoint_path, device):
        print(f"Checkpoint loaded successfully.")
    else:
        print(f"Warning: Checkpoint NOT found. Using random weights.")

    model.eval()
    
    print("Pre-computing Data Graph Features (Encoder)...")
    if not hasattr(data_graph, 'batch'):
        data_graph.batch = torch.zeros(data_graph.num_nodes, dtype=torch.long)
        
    data_graph_gpu = data_graph.to(device)
    with torch.no_grad():
        _, all_data_features = model.encoder(data_graph_gpu)
    print(f"Data Features Shape: {all_data_features.shape}")

    inference_wrapper = NeuGNInferenceWrapper(model, all_data_features, tokenizer, device, params)

    baselines = ['VF3', 'GQL'] 
    query_sizes = [4, 8, 16]
    
    print(f"\n{'='*20} EXPERIMENT START {'='*20}")
    print(f"Generating {args.num_queries} queries for sizes: {query_sizes}")
    
    data_graph_cpu = SimpleGraph(edge_index.cpu(), num_nodes, node_values_id.cpu())
    
    for size in query_sizes:
        print(f"\n>>> Query Size: {size} Nodes")
        
        # [修改] 统计列表，增加时间统计
        avg_steps = {b: [] for b in baselines}
        avg_steps_neu = {b: [] for b in baselines}
        avg_times = {b: [] for b in baselines}
        avg_times_neu = {b: [] for b in baselines}
        
        stats = {b: {'better': 0, 'worse': 0, 'same': 0} for b in baselines}
        
        for i in range(args.num_queries):
            q_graph = generate_random_query(data_graph_cpu, size)
            if q_graph is None:
                print(f"  [Q{i+1}] Failed to generate connected subgraph.")
                continue
            
            print(f"  [Q{i+1}] Generated {q_graph.num_nodes} nodes, {q_graph.edge_index.shape[1]} edges.")
            
            for base in baselines:
                # 1. Run Original (Measured Time)
                matcher = GeneralMatcher(data_graph_cpu, inference_wrapper, method_name=base, use_neugn=False, data_label_freq=label_counts)
                
                t_start = time.perf_counter()
                _, steps = matcher.run_match(q_graph, find_all=args.find_all)
                t_end = time.perf_counter()
                
                avg_steps[base].append(steps)
                avg_times[base].append(t_end - t_start)
                
                # 2. Run NeuGN (Measured Time)
                matcher_neu = GeneralMatcher(data_graph_cpu, inference_wrapper, method_name=base, use_neugn=True, data_label_freq=label_counts)
                
                t_start_neu = time.perf_counter()
                _, steps_neu = matcher_neu.run_match(q_graph, find_all=args.find_all)
                t_end_neu = time.perf_counter()
                
                avg_steps_neu[base].append(steps_neu)
                avg_times_neu[base].append(t_end_neu - t_start_neu)
                
                # Compare Steps
                if steps_neu < steps:
                    stats[base]['better'] += 1
                    status = "Better"
                elif steps_neu > steps:
                    stats[base]['worse'] += 1
                    status = "Worse"
                else:
                    stats[base]['same'] += 1
                    status = "Same"
                
                print(f"    {base:<6}: Steps: {steps:<5} -> {steps_neu:<5} ({status}) | Time: {avg_times[base][-1]:.4f}s -> {avg_times_neu[base][-1]:.4f}s")
        
        print("-" * 70)
        print(f"Size {size} Summary:") 
        for base in baselines:
            # 步数统计
            m_pure = np.median(avg_steps[base]) if avg_steps[base] else 0
            m_neu = np.median(avg_steps_neu[base]) if avg_steps_neu[base] else 0
            imp = (m_pure - m_neu) / m_pure * 100 if m_pure > 0 else 0
            
            # 时间统计
            m_time_pure = np.median(avg_times[base]) if avg_times[base] else 0
            m_time_neu = np.median(avg_times_neu[base]) if avg_times_neu[base] else 0
            imp_time = (m_time_pure - m_time_neu) / m_time_pure * 100 if m_time_pure > 0 else 0
            
            total_valid = len(avg_steps[base])
            if total_valid > 0:
                pct_better = stats[base]['better'] / total_valid * 100
                pct_worse = stats[base]['worse'] / total_valid * 100
                pct_same = stats[base]['same'] / total_valid * 100
            else:
                pct_better = pct_worse = pct_same = 0.0
            
            print(f"  {base:<6}:")
            print(f"          [Steps] Median Imp: {imp:.1f}% (Orig={m_pure:.1f} -> NeuGN={m_neu:.1f})")
            print(f"                  Counts: Better: {stats[base]['better']}/{total_valid} ({pct_better:.1f}%) | "
                  f"Worse: {stats[base]['worse']}/{total_valid} ({pct_worse:.1f}%) | "
                  f"Same: {stats[base]['same']}/{total_valid} ({pct_same:.1f}%)")
            print(f"          [Time]  Median Imp: {imp_time:.1f}% (Orig={m_time_pure:.4f}s -> NeuGN={m_time_neu:.4f}s)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default='./model_params/wikics', type=str)
    parser.add_argument('--graph_path', default='./datasets/wikics', type=str)
    parser.add_argument('--num_queries', default=5, type=int, help="Number of queries to generate per size")
    parser.add_argument('--encoder_name', default='gcn', type=str)
    parser.add_argument('--model_path', default=None, type=str, help="Path to a specific checkpoint file (.pth)")
    parser.add_argument('--find_all', action='store_true', default=False)
    args = parser.parse_args()
    main(args)