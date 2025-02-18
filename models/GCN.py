import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import add_self_loops

class DensityGraphBuilder:
    def __init__(self, k=4):
        self.k = k

    def build_batch_graph(self, density_maps):
        B, C, H, W = density_maps.shape
        num_nodes = H * W

        flat_density = density_maps.view(B, -1) 

        dist = torch.abs(flat_density.unsqueeze(2) - flat_density.unsqueeze(1))
        sorted_indices = torch.argsort(dist, dim=2)[:, :, 1:self.k+1] 

        device = density_maps.device
        src_nodes = torch.arange(num_nodes, device=device).view(1, num_nodes, 1).expand(B, num_nodes, self.k)
        tgt_nodes = sorted_indices

        batch_offset = torch.arange(B, device=device).view(B,1,1)*num_nodes
        src_nodes = src_nodes + batch_offset
        tgt_nodes = tgt_nodes + batch_offset

        src_nodes = src_nodes.reshape(-1)
        tgt_nodes = tgt_nodes.reshape(-1)
        edge_index = torch.stack([src_nodes, tgt_nodes], dim=0) 

        num_nodes_total = B * num_nodes
        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes_total)

        return edge_index, num_nodes_total, H, W

class FeatureGraphBuilder:
    def __init__(self, k=4):
        self.k = k

    def build_batch_graph(self, feature_maps):
        B, C, H, W = feature_maps.shape
        num_nodes = H * W
        device = feature_maps.device

        flat_features = feature_maps.permute(0,2,3,1).contiguous().view(B, num_nodes, C) 

        norm_features = F.normalize(flat_features, p=2, dim=-1) 
        
        sim = torch.matmul(norm_features, norm_features.transpose(-1,-2)) 

        _, sorted_indices = torch.topk(sim, k=self.k+1, dim=2, largest=True)
        sorted_indices = sorted_indices[:, :, 1:]  

        src_nodes = torch.arange(num_nodes, device=device).view(1,num_nodes,1).expand(B, num_nodes, self.k)
        tgt_nodes = sorted_indices

        batch_offset = torch.arange(B, device=device).view(B,1,1)*num_nodes
        src_nodes = src_nodes + batch_offset
        tgt_nodes = tgt_nodes + batch_offset

        src_nodes = src_nodes.reshape(-1)
        tgt_nodes = tgt_nodes.reshape(-1)

        edge_index = torch.stack([src_nodes, tgt_nodes], dim=0)
        num_nodes_total = B * num_nodes
        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes_total)

        return edge_index, num_nodes_total, H, W

class GCNModel(nn.Module):
    def __init__(self, in_channels=256, hidden_channels=512, out_channels=256):
        super(GCNModel, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, tensor, edge_index):
        x = self.conv1(tensor, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5)
        return x

class DensityGCNProcessor(nn.Module):
    def __init__(self, k=4, in_channels=256, hidden_channels=512, out_channels=256):
        super(DensityGCNProcessor, self).__init__()
        self.graph_builder = DensityGraphBuilder(k)
        self.gcn = GCNModel(in_channels, hidden_channels, out_channels)

    def forward(self, density_maps, feature_maps):
        B, in_channels, H, W = feature_maps.shape
        num_nodes = H * W

        edge_index, _, H, W = self.graph_builder.build_batch_graph(density_maps)

        node_features = feature_maps.permute(0,2,3,1).contiguous().view(-1, in_channels)

        out = self.gcn(node_features, edge_index)  

        Density_GCN_map = out.view(B, H, W, in_channels).permute(0,3,1,2).contiguous()

        return Density_GCN_map

class FeatureGCNProcessor(nn.Module):
    def __init__(self, k=4, in_channels=256, hidden_channels=512, out_channels=256):
        super(FeatureGCNProcessor, self).__init__()
        self.graph_builder = FeatureGraphBuilder(k)
        self.gcn = GCNModel(in_channels, hidden_channels, out_channels)

    def forward(self, feature_maps):
        B, C, H, W = feature_maps.shape

        edge_index, _, H, W = self.graph_builder.build_batch_graph(feature_maps)

        node_features = feature_maps.permute(0,2,3,1).contiguous().view(-1, C)

        out = self.gcn(node_features, edge_index)
        Feature_GCN_map = out.view(B,H,W,C).permute(0,3,1,2).contiguous()
        return Feature_GCN_map