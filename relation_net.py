from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F

from utils import *

class GCN_Module(nn.Module):
    def __init__(self, cfg):
        super(GCN_Module, self).__init__()

        self.cfg = cfg

        NFR = cfg.num_features_relation

        NG = cfg.num_graph

        NFG = cfg.num_features_gcn
        NFG_ONE = NFG

        self.fc_rn_theta_list = torch.nn.ModuleList(
            [nn.Linear(NFG, NFR) for i in range(NG)])
        self.fc_rn_phi_list = torch.nn.ModuleList(
            [nn.Linear(NFG, NFR) for i in range(NG)])

        self.fc_gcn_list = torch.nn.ModuleList(
            [nn.Linear(NFG, NFG_ONE, bias=False) for i in range(NG)])

        self.nl_gcn_list = torch.nn.ModuleList(
                [nn.LayerNorm([NFG_ONE]) for i in range(NG)])

    def forward(self, graph_boxes_features, boxes_in_flat):
        """
        graph_boxes_features  [B*T,N,NFG]
        """

        # GCN graph modeling
        # Prepare boxes similarity relation
        B, N, NFG = graph_boxes_features.shape
        NFR = self.cfg.num_features_relation
        NG = self.cfg.num_graph
        NFG_ONE = NFG

        OH, OW = self.cfg.out_size
        pos_threshold = self.cfg.pos_threshold

        # Prepare position mask
        graph_boxes_positions = boxes_in_flat  # B*T*N, 4
        graph_boxes_positions[:, 0] = (
            graph_boxes_positions[:, 0] + graph_boxes_positions[:, 2]) / 2
        graph_boxes_positions[:, 1] = (
            graph_boxes_positions[:, 1] + graph_boxes_positions[:, 3]) / 2
        graph_boxes_positions = graph_boxes_positions[:, :2].reshape(
            B, N, 2)  # B*T, N, 2

        graph_boxes_distances = calc_pairwise_distance_3d(
            graph_boxes_positions, graph_boxes_positions)  # B, N, N

        position_mask = (graph_boxes_distances > (pos_threshold*OW))

        relation_graph = None
        graph_boxes_features_list = []
        for i in range(NG):
            graph_boxes_features_theta = self.fc_rn_theta_list[i](
                graph_boxes_features)  # B,N,NFR
            graph_boxes_features_phi = self.fc_rn_phi_list[i](
                graph_boxes_features)  # B,N,NFR

#             graph_boxes_features_theta=self.nl_rn_theta_list[i](graph_boxes_features_theta)
#             graph_boxes_features_phi=self.nl_rn_phi_list[i](graph_boxes_features_phi)

            similarity_relation_graph = torch.matmul(
                graph_boxes_features_theta, graph_boxes_features_phi.transpose(1, 2))  # B,N,N

            similarity_relation_graph = similarity_relation_graph/np.sqrt(NFR)

            # B*N*N, 1
            similarity_relation_graph = similarity_relation_graph.reshape(
                -1, 1)

            # Build relation graph
            relation_graph = similarity_relation_graph

            relation_graph = relation_graph.reshape(B, N, N)

            relation_graph[position_mask] = -float('inf')

            relation_graph = torch.softmax(relation_graph, dim=2)

            # Graph convolution
            one_graph_boxes_features = self.fc_gcn_list[i](torch.matmul(
                relation_graph, graph_boxes_features))  # B, N, NFG_ONE
            one_graph_boxes_features = self.nl_gcn_list[i](
                one_graph_boxes_features)
            one_graph_boxes_features = F.relu(one_graph_boxes_features)

            graph_boxes_features_list.append(one_graph_boxes_features)

        graph_boxes_features = torch.sum(torch.stack(
            graph_boxes_features_list), dim=0)  # B, N, NFG

        return graph_boxes_features, relation_graph

class GCN_NN_Module(nn.Module):
    def __init__(self,cfg):
        super(GCN_NN_Module, self).__init__()
        self.cfg=cfg
        NFR = cfg.num_features_relation
        NG = cfg.num_graph
        self.fc_rn_theta_list = torch.nn.ModuleList([nn.Linear(NFR, NFR) for i in range(NG)])
        
        self.gru = nn.GRU(NG+1, NG+1, num_layers=1,bidirectional=True)
        self.norm_gru=nn.LayerNorm([NG+1])
        
        self.fc_edge=nn.Linear(NG+1,NG)
        self.ac=nn.ELU()
        self.norm=nn.LayerNorm([NG])
        

    def forward(self, features, adj_mask):
        B, T, N, C = features.shape
        device=features.device
        NG = self.cfg.num_graph
        _,T1,_,_=adj_mask.shape
        A=adj_mask[...,None].float()
        A=A.reshape(B,T1,N,N,1)
        graph_edges_features_list = []
        for i in range(NG):
            graph_boxes_features_theta = self.fc_rn_theta_list[i](
                features)  # B,T,N,C
            graph_boxes_features_phi = graph_boxes_features_theta
            NFR=graph_boxes_features_phi.shape[-1]
            similarity_relation_graph = torch.matmul(
                graph_boxes_features_theta.reshape(B*T,N,C), graph_boxes_features_phi.reshape(B*T,N,C).transpose(1, 2))  # B,N,N

            similarity_relation_graph = similarity_relation_graph/np.sqrt(NFR)
            graph_edges_features_list.append(similarity_relation_graph) #*A[0,...,0]
        graph_edges_features_list.append(A[0,...,0])
        edge_feature = torch.stack(graph_edges_features_list, dim=-1).reshape(B,T,N,N,NG+1)  # B, T,N,N,NG+1
        h1 = torch.empty((2*1, B * N*N, self.cfg.num_graph+1), device=device)
        torch.nn.init.xavier_normal_(h1)
        edge_feature=edge_feature.permute(1,0,2,3,4).reshape(T,B*N*N,-1)
        edge_feature=self.norm_gru(edge_feature)
        edge_feature, _ = self.gru(edge_feature, h1)
        edge_feature=edge_feature.reshape(T,B,N,N,2,-1).mean(dim=4).permute(1,0,2,3,4)
        edge_feature=self.fc_edge(edge_feature.reshape(B,T,N,N,self.cfg.num_graph+1))
        edge_feature=self.norm(edge_feature)
        edge_feature=self.ac(edge_feature)
        
        gru_outputs=(edge_feature+edge_feature.transpose(2,3))/2
        return gru_outputs

    
class RelationModule(nn.Module):

    def __init__(self, cfg):
        super(RelationModule, self).__init__()
        self.cfg = cfg
        self.stage=cfg.training_stage
        self.graph=GCN_NN_Module(self.cfg)
        
        self.fc_groups = nn.Sequential(
            nn.Linear(cfg.num_graph, self.cfg.num_groups)
        )
        self.dropout_global = nn.Dropout(p=self.cfg.train_dropout_prob)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m,nn.GRU):
                for name, param in m.named_parameters():
                    if 'bias' in name:
                        nn.init.constant_(param, 0.0)
                    elif 'weight' in name:
                        nn.init.orthogonal_(param)
        

    def loadmodel(self, filepath):
        state = torch.load(filepath)
        process_dict = OrderedDict()
        for key in state['state_dict'].keys():
            if key.startswith('module.'):
                process_dict[key[7:]] = state['state_dict'][key]
            elif 'Mixed_6' in key:
                continue
            else:
                process_dict[key] = state['state_dict'][key]
        self.load_state_dict(process_dict)
        print('Load all parameter from: ', filepath)

    def forward(self, features, A):
        att_fs=self.graph(features,A)
        group_scores=self.fc_groups(att_fs).mean(dim=1)
        return None,None,group_scores
    
