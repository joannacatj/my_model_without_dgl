import torch
import random
import numpy as np
from typing import List, Sequence, Dict

class GraphTokenizer:
    """
    Tokenizer using pure PyTorch/Python logic.
    Removes all dependencies on DGL's random_walk.
    """

    def __init__(self, graph):
        # graph expects SimpleGraph from utils
        self.graph = graph
        
        # [关键修复] 属性访问不加括号
        self.node_num = self.graph.num_nodes
        
        self.padding_id = self.node_num
        self.sos_id = self.padding_id + 1

        # Token mappings
        self.node_to_token = {node_id: idx for idx, node_id in enumerate(range(self.node_num))}
        self.node_to_token[self.padding_id] = self.padding_id
        
        self.token_to_node = {idx: node_id for node_id, idx in self.node_to_token.items()}

        # [关键修复] 构建邻接表用于纯 Python 随机游走
        # edge_index is [2, E] tensor
        print("Building Adjacency List for Random Walk...")
        src, dst = self.graph.edge_index
        src = src.tolist()
        dst = dst.tolist()
        
        self.adj_list: Dict[int, List[int]] = {}
        for s, d in zip(src, dst):
            if s not in self.adj_list:
                self.adj_list[s] = []
            self.adj_list[s].append(d)
        print("Adjacency List Built.")

    def node_nums(self):
        return self.node_num

    def token_nums(self):
        return self.node_num + 2

    def _single_random_walk(self, start_node: int, length: int) -> List[int]:
        """
        Internal helper for pure python random walk
        """
        walk = [start_node]
        curr = start_node
        for _ in range(length - 1):
            neighbors = self.adj_list.get(curr, [])
            if not neighbors:
                break
            curr = random.choice(neighbors)
            walk.append(curr)
        return walk

    def random_walks(self, start_nodes: Sequence[int], length: int) -> List[List[int]]:
        """
        Perform random walks using pure Python lists.
        Replaces DGL's C++ random_walk.
        """
        walks = []
        # Convert tensor to list if necessary
        if isinstance(start_nodes, torch.Tensor):
            start_nodes = start_nodes.tolist()
            
        for start_node in start_nodes:
            walks.append(self._single_random_walk(start_node, length))
        return walks

    # 兼容性接口：如果外部代码调用了 neighborhoods_sampling
    def neighborhoods_sampling(self, start_nodes, fanouts_max):
        # Fallback to random walk
        return self.random_walks(start_nodes, length=len(fanouts_max)+1)

    def encode_walks(self, walks: List[List[int]]) -> List[List[int]]:
        return [[self.node_to_token.get(node, self.padding_id) for node in walk] for walk in walks]

    def encode_walk(self, walk: List[int]) -> List[int]:
        return [self.node_to_token.get(node, self.padding_id) for node in walk]

    def decode_walks(self, token_sequences: List[List[int]]) -> List[List[int]]:
        return [[self.token_to_node.get(token, -1) for token in sequence] for sequence in token_sequences]