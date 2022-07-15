from collections import OrderedDict

from backbone import *
from utils import *

class GCN_NN_Module(nn.Module):
    def __init__(self,cfg):
        super(GCN_NN_Module, self).__init__()
        self.cfg=cfg
        number_app=cfg.num_features_relation
        # self.number_mt=number_mt

        NFR = cfg.num_features_relation

        NG = cfg.num_graph

        NFG = cfg.num_features_gcn
        NFG_ONE = NFG

        self.fc_rn_theta_list = torch.nn.ModuleList(
            [nn.Linear(NFG, NFR) for i in range(NG)])
        self.fc_rn_phi_list = torch.nn.ModuleList(
            [nn.Linear(NFG, NFR) for i in range(NG)])
        
        self.fc_edge=nn.Linear(NG,NG)
        self.ac=nn.ELU()
        self.norm=nn.LayerNorm([NG])

        # self.nl_mt = nn.LayerNorm([num_mt])
        # self.gru=nn.GRU(number_app,number_app,num_layers=2,bidirectional=True,dropout=self.cfg.train_dropout_prob) #
        # self.gru_mt=nn.GRU(num_mt,num_mt,num_layers=2,bidirectional=True,dropout=self.cfg.train_dropout_prob) #
        self.fc_final=nn.Sequential(nn.Linear(NG,1))
        # self.nl_final = nn.LayerNorm([1])

    def forward(self, features,adj_mask):
        B, T, N, C = features.shape
        device=features.device
        NG = self.cfg.num_graph
        # edge_feature= torch.relu(self.nl_app(self.fc_app()))
        # relation_graph = None
        graph_edges_features_list = []
        for i in range(NG):
            graph_boxes_features_theta = self.fc_rn_theta_list[i](
                features)  # B,T,N,C
            graph_boxes_features_phi = self.fc_rn_phi_list[i](
                features)  # B,T,N,C
            NFR=graph_boxes_features_phi.shape[-1]
            similarity_relation_graph = torch.matmul(
                graph_boxes_features_theta.reshape(B*T,N,C), graph_boxes_features_phi.reshape(B*T,N,C).transpose(1, 2))  # B,N,N

            similarity_relation_graph = similarity_relation_graph/np.sqrt(NFR)
            # similarity_relation_graph = graph_boxes_features_theta[:,:,:,None,:]+graph_boxes_features_phi[:,:,None,:,:] # B,T,N,N,C
            graph_edges_features_list.append(similarity_relation_graph)

        edge_feature = torch.stack(graph_edges_features_list, dim=-1)  # B, T,N,N,NG
        
        edge_feature=self.fc_edge(edge_feature.reshape(B,T,N,N,self.cfg.num_graph))
        edge_feature=self.norm(edge_feature)
        edge_feature=self.ac(edge_feature)
        
        gru_outputs=edge_feature
        output=self.fc_final(gru_outputs).squeeze(-1)
        return output,gru_outputs*adj_mask[...,None].float()
    

from reference.my_shift_gcn import MyModel
class RelationNet(nn.Module):
    """
    GATv2+GRU
    """

    def __init__(self, cfg):
        super(RelationNet, self).__init__()
        ################ Parameters #####################
        self.cfg = cfg
        self.stage=cfg.training_stage
        number_feature_relation = self.cfg.num_features_relation
        ################ Parameters #####################

        ################ Modules ########################
        self.shiftgcn=MyModel(num_class=number_feature_relation, num_point=(cfg.action_features+2)//2, in_channels=2)
        # graph update node
        self.graph=GCN_NN_Module(self.cfg)
        # output action
        self.fc_interactions = nn.Sequential(
            nn.Linear(cfg.num_graph, self.cfg.num_interactions)
        )
        #
        # use actions, plus relation, output which interaction, output who are the interactors.
        self.fc_avoids = nn.Sequential(
            nn.Linear(cfg.num_graph, self.cfg.num_avoids)
        )
        self.fc_groups = nn.Sequential(
            nn.Linear(cfg.num_graph, self.cfg.num_groups)
        )
        self.dropout_global = nn.Dropout(p=self.cfg.train_dropout_prob)
        ################ Modules ########################
        
        
        ################ Init ###########################
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
        if not self.cfg.train_backbone:
            for p in self.parameters():
                p.requires_grad = False
            for p in self.fc_interactions.parameters():
                p.requires_grad = True
            for p in self.fc_avoids.parameters():
                p.requires_grad = True
            for p in self.fc_groups.parameters():
                p.requires_grad = True

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
    # @autocast()
    def forward(self, features, A):
        # with autocast():
        # features: B,T,N,C
        # A: B,T,N,N
        # Output: B,T,N,N
        # Here B is set to 1 instead.

        # read config parameters
        B,T,MAX_N,C = features.shape
        device = features.device

        if self.stage==1:
            features=features.reshape(B,T,MAX_N,C//2,2)
            
            features=self.shiftgcn(features.permute(0,4,1,3,2))
            # B,T,MAX_N,C
            atts,att_fs=self.graph(features,A)
        else:
            with torch.no_grad():
                features=features.reshape(B,T,MAX_N,C//2,2)
            features=self.shiftgcn(features.permute(0,4,1,3,2))
            # B,T,MAX_N,C
            atts,att_fs=self.graph(features,A)

        if self.stage==1:
            return atts
        else:
            interaction_scores=self.fc_interactions(att_fs)
            avoid_scores=self.fc_avoids(att_fs)
            group_scores=self.fc_groups(att_fs).mean(dim=1)
            # atts=atts[...,None]
            # group_scores=torch.cat([1-atts,atts],dim=-1).mean(dim=1)
            return interaction_scores,avoid_scores,group_scores
