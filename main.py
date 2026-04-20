import torch
import os
from datetime import datetime
from torch import nn, optim
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
from NeuGN.graph_tokenizer import GraphTokenizer
from NeuGN.model import GraphDecoder
from NeuGN.utils import metrics, load_amazon, load_ori_graph, save_value2id, load_model_args, load_checkpoint, save_checkpoint, process_input_data, SimpleGraph
from NeuGN.datasets import GraphWalkDataset 
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
import yaml
import argparse
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.1): 
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, query, positives, negatives):
        query = F.normalize(query, dim=-1)
        positives = F.normalize(positives, dim=-1)
        negatives = F.normalize(negatives, dim=-1)

        pos_sim = torch.sum(query * positives, dim=-1, keepdim=True)
        neg_sim = torch.sum(query.unsqueeze(1) * negatives, dim=-1)
        
        logits = torch.cat([pos_sim, neg_sim], dim=1)
        logits /= self.temperature
        
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=query.device)
        loss = F.cross_entropy(logits, labels)
        return loss

def train_one_epoch(model, graph, graph_tokenizer, dataloader, criterion, optimizer, params, device, epoch, writer):
    model.train()
    total_loss = 0
    if hasattr(dataloader.sampler, "set_epoch"):
        dataloader.sampler.set_epoch(epoch)
    
    raw_model = model.module if isinstance(model, DDP) else model
    step = 0
    progress_bar = tqdm(dataloader, desc=f"Training Epoch {epoch}")

    for start_nodes, graph_len, walk_len in progress_bar:
        # [关键修复] 这里必须解包 7 个返回值，接收 spd_bias_tensor
        batch_graphs, input_indices, input_subnode_seqs, target_indices, neg_indices, token_mask_len_seq, spd_bias_tensor = process_input_data(
            graph, graph_tokenizer, start_nodes, graph_len, walk_len, epoch, params
        )

        batch_graphs = batch_graphs.to(device)
        input_indices = input_indices.to(device)
        input_subnode_seqs = input_subnode_seqs.to(device)
        target_indices = target_indices.to(device)
        neg_indices = neg_indices.to(device) 
        token_mask_len_seq = token_mask_len_seq.to(device)
        
        # [关键修复] 将 spd_bias_tensor 移动到设备
        spd_bias_tensor = spd_bias_tensor.to(device)
        
        optimizer.zero_grad()
        
        graph_features, all_node_features = raw_model.encoder(batch_graphs)
        if raw_model.encoder_type in raw_model.gnn_list:
            graph_features = graph_features.unsqueeze(dim=1)

        input_feature_vectors = all_node_features[input_indices]
        target_vectors = all_node_features[target_indices]
        
        K = neg_indices.shape[1]
        neg_vectors = all_node_features[neg_indices.view(-1)].view(-1, K, all_node_features.shape[-1]) 

        # [关键修复] 将 spd_bias_tensor 传入 decoder
        outputs = raw_model.decoder(graph_features, input_feature_vectors, input_subnode_seqs, token_mask_len_seq, 0, spd_indices=spd_bias_tensor)
        
        gather_index = token_mask_len_seq.view(-1, 1, 1).expand(-1, -1, outputs.size(-1))
        pred_vectors = torch.gather(outputs, 1, gather_index).squeeze(1)

        loss = criterion(pred_vectors, target_vectors, neg_vectors)
        
        if step % 50 == 0:
            with torch.no_grad():
                p_v = F.normalize(pred_vectors, dim=-1)
                t_v = F.normalize(target_vectors, dim=-1)
                n_v = F.normalize(neg_vectors, dim=-1)
                pos_sim = torch.sum(p_v * t_v, dim=-1).mean().item()
                neg_sim = torch.sum(p_v.unsqueeze(1) * n_v, dim=-1).mean().item()
                diff = pos_sim - neg_sim
                progress_bar.write(f"[Debug Step {step}] Loss: {loss.item():.3f} | Pos Sim: {pos_sim:.3f} | Neg Sim: {neg_sim:.3f} | Diff: {diff:.3f}")

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        step += 1
        current_lr = optimizer.param_groups[0]['lr']
        progress_bar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{current_lr:.2e}")

    avg_loss = total_loss / len(dataloader)
    if writer is not None:
        writer.add_scalar(f'train_average_loss', avg_loss, epoch)
    return avg_loss

def eval_one_epoch(model, graph, graph_tokenizer, dataloader, criterion, device, epoch, writer, name):
    model.eval()
    total_loss = 0
    all_ranks = []
    
    raw_model = model.module if isinstance(model, DDP) else model
    progress_bar = tqdm(dataloader, desc=f"{name} Eval")
    
    with torch.no_grad():
        for start_nodes, graph_len, walk_len in progress_bar:
            # [关键修复] 解包 7 个返回值
            batch_graphs, input_indices, input_subnode_seqs, target_indices, neg_indices, token_mask_len_seq, spd_bias_tensor = process_input_data(
                graph, graph_tokenizer, start_nodes, graph_len, walk_len, epoch, raw_model.params
            )
            
            batch_graphs = batch_graphs.to(device)
            input_indices = input_indices.to(device)
            input_subnode_seqs = input_subnode_seqs.to(device)
            target_indices = target_indices.to(device)
            neg_indices = neg_indices.to(device)
            token_mask_len_seq = token_mask_len_seq.to(device)
            
            # [关键修复] 移动到设备
            spd_bias_tensor = spd_bias_tensor.to(device)

            graph_features, all_node_features = raw_model.encoder(batch_graphs)
            if raw_model.encoder_type in raw_model.gnn_list:
                graph_features = graph_features.unsqueeze(dim=1)

            input_feature_vectors = all_node_features[input_indices]
            target_vectors = all_node_features[target_indices]
            
            K = neg_indices.shape[1]
            neg_vectors = all_node_features[neg_indices.view(-1)].view(-1, K, all_node_features.shape[-1])

            # [关键修复] 传入 spd_indices
            outputs = raw_model.decoder(graph_features, input_feature_vectors, input_subnode_seqs, token_mask_len_seq, 0, spd_indices=spd_bias_tensor)
            gather_index = token_mask_len_seq.view(-1, 1, 1).expand(-1, -1, outputs.size(-1))
            pred_vectors = torch.gather(outputs, 1, gather_index).squeeze(1)
            
            loss = criterion(pred_vectors, target_vectors, neg_vectors)
            total_loss += loss.item()

            pred_norm = F.normalize(pred_vectors, dim=-1)
            all_node_norm = F.normalize(all_node_features, dim=-1)
            
            sim_matrix = torch.matmul(pred_norm, all_node_norm.T)
            
            for i in range(len(target_indices)):
                true_idx = target_indices[i].item()
                true_sim = sim_matrix[i, true_idx]
                rank = (sim_matrix[i] > true_sim).sum().item() + 1
                all_ranks.append(rank)

    avg_loss = total_loss / len(dataloader)
    acc1, acc10, acc20, acc50, acc100, mrr, ndcg = metrics(all_ranks)
    print(f'[{name}] Epoch {epoch}: Loss: {avg_loss:.4f} | MRR: {mrr*100:.2f}% | Hit@1: {acc1*100:.2f}% | Hit@10: {acc10*100:.2f}%')
    
    if writer is not None:
        writer.add_scalar(f'{name}_average_loss', avg_loss, epoch)
        writer.add_scalar(f'{name}_mrr', mrr, epoch)
    return avg_loss

def main(args):
    local_rank = args.local_rank
    if local_rank != -1:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        dist.init_process_group(backend='nccl')
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    params = load_model_args(args.config_path)
    
    # [新增] 注入默认参数以支持 GDT 新特性
    if not hasattr(params.encoder_config, 'rwse_dim'):
        params.encoder_config.rwse_dim = 20
    if not hasattr(params.decoder_config, 'max_spd'):
        params.decoder_config.max_spd = 20
    if not hasattr(params.dataprocess_config, 'num_negatives'):
        params.dataprocess_config.num_negatives = 64
        
    if not hasattr(params.encoder_config, 'fixed_input_dim'):
        params.encoder_config.fixed_input_dim = 10000
    params.encoder_config.graph_feature_dim = 512
    params.encoder_config.encoder_name = 'gin' 
    
    print(f"Model Config: Dim={params.decoder_config.dim}, Negatives={params.dataprocess_config.num_negatives}, Encoder={params.encoder_config.encoder_name}, RWSE={params.encoder_config.rwse_dim}, MaxSPD={params.decoder_config.max_spd}")

    dataset_list = ['lastfm', 'hamster', 'nell', 'wikics', 'amazon', 'youtube', 'dblp']
    dataset_name = 'wikics' 
    for name in dataset_list:
        if name in params.graph_path:
            dataset_name = name
            break
            
    print(f"Loading dataset: {dataset_name} from {params.graph_path}")
    if dataset_name == 'amazon':
         node_values, node_values_uni, edge_src_ids, edge_dst_ids = load_amazon(params.graph_path)
    else:
         node_values, node_values_uni, edge_src_ids, edge_dst_ids = load_ori_graph(params.graph_path, dataset_name)

    now = datetime.now()
    formatted_time = now.strftime("%Y_%m_%d_%H_%M_%S")
    writer_path = os.path.join('experiments', 'results', f'logs_{params.encoder_config.encoder_name}_{dataset_name}_{formatted_time}')
    
    writer = None
    if local_rank <= 0:
        writer = SummaryWriter(writer_path)
        with open(os.path.join(writer_path, 'model_args.yaml'), 'w') as f:
            yaml.dump(params.to_dict(), f, default_flow_style=False, sort_keys=False)
        
    value2id = save_value2id(node_values_uni, args.config_path, dataset_name)
    
    num_nodes = len(node_values)
    edge_index = torch.tensor([edge_src_ids, edge_dst_ids], dtype=torch.long)
    node_values_id = torch.tensor([value2id[str(node_value)] for node_value in node_values], dtype=torch.long)
    
    graph = SimpleGraph(edge_index, num_nodes, node_feat=node_values_id)
    
    print(f"Graph created: {graph.num_nodes} nodes, {graph.edge_index.shape[1]} edges.")
    tokenizer = GraphTokenizer(graph=graph)
    
    params.device = device
    
    model = GraphDecoder(params).to(device)
    if local_rank != -1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
        
    optimizer = optim.Adam(model.parameters(), lr=5e-5, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    criterion = InfoNCELoss(temperature=0.2)
    
    checkpoint_dir = os.path.join(params.checkpoint_path, dataset_name)
    if not os.path.exists(checkpoint_dir) and local_rank <= 0:
        os.makedirs(checkpoint_dir)
    checkpoint_file = os.path.join(checkpoint_dir, f'{params.encoder_config.encoder_name}_checkpoint.pth')

    # [注意] 这里引用了正确的 datasets.py 路径 ( NeuGN.datasets)
    train_dataset = GraphWalkDataset(tokenizer, max_len=params.dataprocess_config.walk_len, mode='train', rand_pre=False)
    if local_rank != -1:
        train_sampler = DistributedSampler(train_dataset)
    else:
        train_sampler = None
    
    train_dataloader = DataLoader(train_dataset, batch_size=128, sampler=train_sampler, shuffle=(train_sampler is None), num_workers=4)

    test_dataset = GraphWalkDataset(tokenizer, max_len=params.dataprocess_config.walk_len, mode='trans_test', rand_pre=False)
    if local_rank != -1:
        test_sampler = DistributedSampler(test_dataset, shuffle=False)
    else:
        test_sampler = None
    test_dataloader = DataLoader(test_dataset, batch_size=128, sampler=test_sampler, shuffle=False, num_workers=4)

    start_epoch = 0
    if args.load_params and os.path.exists(checkpoint_file):
        if local_rank != -1:
            start_epoch = load_checkpoint(checkpoint_file, model, optimizer, scheduler, local_rank)
        else:
            state = torch.load(checkpoint_file, map_location=device)
            state_dict = state['model_state_dict']
            model_dict = model.state_dict()
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.') and 'module.' + k[7:] not in model_dict:
                    new_state_dict[k[7:]] = v 
                elif not k.startswith('module.') and 'module.' + k in model_dict:
                    new_state_dict['module.' + k] = v 
                else:
                    new_state_dict[k] = v
            model.load_state_dict(new_state_dict)
            optimizer.load_state_dict(state['optimizer_state_dict'])
            scheduler.load_state_dict(state['scheduler_state_dict'])
            start_epoch = state['epoch']
            print(f"Checkpoint loaded from epoch {start_epoch}")
    
    for epoch in range(start_epoch, args.epochs):
        if local_rank <= 0:
            print(f"\n=== Epoch {epoch+1}/{args.epochs} ===")
            
        train_loss = train_one_epoch(model, graph, tokenizer, train_dataloader, criterion, optimizer, params, device, epoch, writer)
        scheduler.step()
        
        if (epoch + 1) % 20 == 0:
             eval_loss = eval_one_epoch(model, graph, tokenizer, test_dataloader, criterion, device, epoch, writer, "Val")
            
        if (epoch + 1) % 20 == 0 and local_rank <= 0:
            save_checkpoint(model, optimizer, scheduler, epoch + 1, checkpoint_file)
            print(f"Checkpoint saved to {checkpoint_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-rank", default=-1, type=int)
    parser.add_argument("--epochs", default=5000, type=int)
    parser.add_argument('--load_params', default=0, type=int)
    parser.add_argument('--config_path', default='./model_params/wikics', type=str)

    args = parser.parse_args()
    main(args)