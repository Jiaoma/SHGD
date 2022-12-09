import torch
import torch.nn as nn
from reference.my_shift_gcn import Model as ShiftGCN
from reference.Shift_GCN.model.shift_gcn import Model as ShiftGCN
from reference.stgcn_model import st_gcn
from relation_net import RelationModule

class RBSModel2(nn.Module):
    def __init__(self, cfg, n_stgcnn =1,n_txpcnn=1,input_feat1=32,input_feat2=32,output_feat=32,
                 seq_len=8,pred_seq_len=12,kernel_size=3, mode=1):
        super(RBSModel2, self).__init__()
        self.cfg=cfg
        number_feature_relation = self.cfg.num_features_relation
        C2=2*cfg.num_joints
        C3=output_feat
        self.input_feat1=input_feat1
        self.input_feat2=input_feat2
        self.mode=mode
        
        self.encoder_shiftgcn=ShiftGCN(num_class=number_feature_relation, num_point=cfg.num_joints, in_channels=input_feat1//cfg.num_joints,onlyFeature=True)
        
        self.n_stgcnn= n_stgcnn
        self.n_txpcnn = n_txpcnn
                
        # self.st_gcns = nn.ModuleList()
        # self.st_gcns.append(st_gcn(input_feat2,input_feat2,(kernel_size,seq_len)))
        # self.st_gcns.append(st_gcn(input_feat2*4,input_feat2,(kernel_size,seq_len)))
        # for j in range(2):
        #     self.st_gcns.append(st_gcn(input_feat2,input_feat2,(kernel_size,seq_len)))
        # self.stdrop=nn.Dropout2d(0.2)
        # self.st_gcns.append(st_gcn(input_feat2,input_feat2,(kernel_size,seq_len)))
        # self.reduce_fc=nn.Linear(input_feat1+input_feat2+(self.n_stgcnn-1)*number_feature_relation+output_feat,number_feature_relation)
        self.c_st_gcns = nn.ModuleList()
        self.c_st_gcns.append(st_gcn(input_feat1+input_feat2,number_feature_relation,(kernel_size,seq_len)))
        for j in range(1,self.n_stgcnn-1):
            self.c_st_gcns.append(st_gcn(number_feature_relation,number_feature_relation,(kernel_size,seq_len)))
        self.c_st_gcns.append(st_gcn(number_feature_relation,input_feat2,(kernel_size,seq_len)))
        if mode==1:
            self.tpcnns = nn.ModuleList()
            self.tpcnns.append(nn.Conv2d(seq_len,pred_seq_len,3,padding=1))
            for j in range(1,self.n_txpcnn):
                self.tpcnns.append(nn.Conv2d(pred_seq_len,pred_seq_len,3,padding=1))
            self.tpcnn_ouput = nn.Conv2d(pred_seq_len,pred_seq_len,3,padding=1)
        else:
            self.relationnet=RelationModule(cfg)

            
        self.prelus = nn.ModuleList()
        for j in range(self.n_txpcnn):
            self.prelus.append(nn.PReLU())
        if not self.cfg.train_backbone and mode!=1:
            for p in self.parameters():
                p.requires_grad = False
            for p in self.relationnet.parameters():
                p.requires_grad = True
                print(p)
                
    
    def loadmodel(self, filepath,pretrain=False):
        state = torch.load(filepath)
        process_dict = self.state_dict()
        for key in state['state_dict'].keys():
            if pretrain and (key.startswith('relationnet') or key.startswith('tpcnns') ): # warning, here stgcn should be deleted if stage model is complete.  
                continue
            if key.startswith('module.') and key[7:] in process_dict:
                process_dict[key[7:]] = state['state_dict'][key]
            elif key in process_dict:
                process_dict[key] = state['state_dict'][key]
            else:
                print("Name: {}, in ckpt but not in current model".format(key))
        self.load_state_dict(process_dict)
        print('Load all parameter from: ', filepath)
        
    def forward(self,v,a):
        if len(a.shape)==4:
            a=a[0]
        a=a.float()
        # v: B, (C1+C2), T, M
        skeletons_o=v[:,:self.input_feat1,:,:]
        skeletons=skeletons_o.clone()
        position_o=v[:,self.input_feat1:,:,:]
        position=position_o.clone()
        B,_,T,M=position.shape
        V=self.cfg.num_joints
        skeletons=skeletons.reshape(B,2,V,T,M).permute(0,1,3,2,4) # B, 2, T, V, M
        # B, C, T, M
        
        # Shift GCN input is B C//V T V M, -> B, M, C1, T, V  ->> B, C1*V, T, M
        skeleton_feat=self.encoder_shiftgcn(skeletons) # B,M, 2,T,V
        skeleton_feat=skeleton_feat.permute(0,2,4,3,1).reshape(B,-1,T,M) # B C1 T M, C1=2*V
        
        # for k in range(2):
        #     position,a = self.st_gcns[k](position,a)
        # position=self.stdrop(position)
        # position,a = self.st_gcns[0](position,a)
        combination=torch.cat([skeleton_feat,position],dim=1) # B,C1+C2,T,M
        # combination=position
        # combination,a = self.st_gcns[-1](combination,a)
        # position: B C T M
        # v = v.view(v.shape[0],v.shape[2],v.shape[1],v.shape[3])
        st_features=[]
        for k in range(self.n_stgcnn-1):
            combination,a = self.c_st_gcns[k](combination,a)
            st_features.append(combination)
        if self.mode==1:
            combination,a = self.c_st_gcns[-1](combination,a)
            combination=combination.permute(0,2,1,3) # B, T , C2+C3+1, M 
            combination = self.prelus[0](self.tpcnns[0](combination))

            for k in range(1,self.n_txpcnn-1):
                combination =  self.prelus[k](self.tpcnns[k](combination)) + combination
                
            combination = self.tpcnn_ouput(combination) # B,T, C,M
            # v = v.view(v.shape[0],v.shape[2],v.shape[1],v.shape[3]) #B,C,T,M
            combination=combination.permute(0,1,3,2) # B, T,M,C
            # combination = combination.reshape(combination.shape[0],combination.shape[1],combination.shape[3],combination.shape[2]) #B,C,T,M
            v=v.permute(0,2,3,1)
            # combination=v*(1-mask_o)+combination*mask_o
            return combination,a
        else:
            st_features=combination.permute(0,2,3,1) # B, T,M ,C
            return self.relationnet(st_features,a[None,...]) # B=1
