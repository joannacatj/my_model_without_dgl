# nx_utils.py
import random
import networkx as nx
from typing import List, Tuple

# 注意：此文件中的 graph2path_v2 依赖 DGLGraph，已在 utils.py 中用 graph2path_v2_pure 替代
# 这里保留函数定义以防其他引用，但内部逻辑指向空或简单实现
# 建议在 utils.py 中直接使用 v2_pure，此处仅保留 connected_graph2path 供 utils 调用

def connected_graph2path(G) -> List[Tuple[int, int]]:
    # ... (逻辑同 utils.py 中实现，可直接复制 utils 中的代码到此，或让 utils 导入)
    if len(G.nodes) <= 1:
        path = []
    else:
        if not nx.is_eulerian(G):
            G = nx.eulerize(G)
        node = random.choice(list(G.nodes()))
        if random.random() < 0.5:
            path_iter = nx.eulerian_path(G, source=node)
        else:
            path_iter = nx.eulerian_circuit(G, source=node)
        
        raw_path = list(path_iter)
        
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