from os import replace
import torch
from torch.utils import data
from feature_extract.panda import PANDA,GROUP_TYPES,INTERACTION_TYPES,AVOID_TYPES
from feature_extract.config import YOURPATH
import random
import numpy as np
import os
from utils import sincos_encoding_2d
from draw import drawBoxes
import tqdm
import cv2
class PANDADataset(data.Dataset):
    """
    1. Divide seq
    2. Give all frame its (dx, dy) estimation
    3. Give sample according to src_fid.
    """

    def __init__(self,cfg,videos,augment='train'):
        self.cfg=cfg
        self.C=self.cfg.action_features
        self.Cp=self.cfg.num_posf
        self.T=self.cfg.T
        self.aug=augment # 'train': multiple times and types, 'none':none, 'distill': 1 time, totally remove someone.
        if cfg.use_unlabeled:
            self.vfp_bboxfeatures=torch.load(cfg.all_features_path)
        else:
            self.vfp_bboxfeatures=torch.load(cfg.interaction_path)
        # Note that, the bbox from pth file is used for feature extractor, thus is scaled to the cropped small image. We need the original (x,y) coordinates in the original big scene.
        self.source=PANDA(os.path.join(YOURPATH,'PANDA'),videos)
        self.v=self.source.annotation_dict.keys()
        self.annos=self.source.annotation_dict
        self.vf=self.source.frame_dict.keys() # all the (vid,fid)
        self.vf_p=self.source.frame_dict
        # generate an available dict for all the vid and fid.
        ava_vf_list=[]
        ava_v_f_dict={}
        for vid in self.v:
            fids=[]
            for vf in self.vf:
                if vf[0]==vid:
                    fids.append(vf[1])
            fids.sort()
            fids=fids[self.T-1:] # The first T-1 frames couldn't be selected as the last frame of one segment.
            for fid in fids:
                ava_vf_list.append((vid,fid))
                if vid not in ava_v_f_dict:
                    ava_v_f_dict[vid]=[]
                ava_v_f_dict[vid].append(fid)
        self.ava_vf_list=ava_vf_list
        self.ava_v_f_dict=ava_v_f_dict
        # vid_pid_fid_interaction_dict
        vid_pid_fid_interaction_dict={}
        for vid in self.annos.keys():
            vid_pid_fid_interaction_dict[vid]={}
            item=self.annos[vid]
            interaction_list=item['interaction_anno']
            if interaction_list is not None:
                for pi_dict in interaction_list:
                    if pi_dict['person_id'] not in vid_pid_fid_interaction_dict[vid]:
                        vid_pid_fid_interaction_dict[vid][pi_dict['person_id']]={}
                    if pi_dict['second'] not in vid_pid_fid_interaction_dict[vid][pi_dict['person_id']]:
                        vid_pid_fid_interaction_dict[vid][pi_dict['person_id']][pi_dict['second']]=pi_dict
        self.vid_pid_fid_interaction_dict=vid_pid_fid_interaction_dict
        # vid_pid_group_dict
        vid_pid_group_dict={}
        for vid in self.annos.keys():
            vid_pid_group_dict[vid]={}
            item=self.annos[vid]
            group_list=item['group_anno']
            if group_list is not None:
                for pg_dict in group_list:
                    for member in pg_dict['members']:
                        if member not in vid_pid_group_dict[vid]:
                            vid_pid_group_dict[vid][member]=pg_dict
        self.vid_pid_group_dict=vid_pid_group_dict
        vid_group_dict={}
        for vid in self.annos.keys():
            vid_group_dict[vid]=[]
            for mt in self.annos[vid]['group_anno']:
                vid_group_dict[vid].append({'members':mt['members'],'relation':GROUP_TYPES.index(mt['relation'])+1})
        self.vid_group_dict=vid_group_dict
        
    def __len__(self):
        """
        Return the total number of samples
        """
        return len(self.ava_vf_list)

    def __getitem__(self, index):
        # index is a fid. This fid belongs to the first frame of a short segment of videos. We should first find all the fids.
        vid,fid_last=self.ava_vf_list[index]
        fid_first=fid_last-self.T+1
        fids=list(range(fid_first,fid_last+1))
        vfs=[(vid,fid) for fid in fids]
        # Read all the pids in these fids. And organize them into a big matrix with the shape of T,V,C, where V means number of nodes.
        # NOTE:
        '''
        Here exists many solutions for deciding which nodes should be selected. And we use all the nodes here.
        '''
        pid_vfp_dict={} # Here, only 1 video is selected, so pid is unique.
        for vf in vfs:
            for p in self.vf_p[vf].keys():
                if (*vf,p) in self.vfp_bboxfeatures:
                    if p not in pid_vfp_dict:
                        pid_vfp_dict[p]=[]
                    pid_vfp_dict[p].append((*vf,p)) # vid,fid,pid
                # Here we use the key of a dict as set to prevent duplicate pid.
        human_num=len(pid_vfp_dict.keys())
        humans=list(pid_vfp_dict.keys())
        humans.sort()
        feature_mat=torch.zeros(self.T,human_num,(self.C+self.Cp+2))
        heights=torch.zeros(self.T,human_num)
        # Find the bboxes of all the pids, attach them to the C channel of the big matrix, thus results in a matrix shaped in T,V,(C+2). For the abscent person in the matrix, the features will be filled with zeros.
        for person in pid_vfp_dict.keys():
            for (vid,fid,pid) in pid_vfp_dict[person]:
                bbox=self.vf_p[(vid,fid)][pid]['rect']
                bbox= [bbox['tl']['x'],bbox['tl']['y'],bbox['br']['x'],bbox['br']['y']]  # xtl,ytl,xbr,ybr
                person_height=bbox[3]-bbox[1] if (bbox[3]-bbox[1])!=0 else 1e5
                reg=lambda x: max(min(x,1.0),0.0)
                bbox= [reg(i) for i in bbox]
                center=[(bbox[0]+bbox[2])/2,bbox[3]] # Use the middle of xs, the bottom of y as center (Feet point).
                posf=sincos_encoding_2d(torch.tensor(center)[None,:],self.Cp)
                # posf=torch.tensor(center)
                # if fid==fid_first:
                    # print('debug')
                feature_mat[fid-fid_first,humans.index(pid),:self.C]=self.vfp_bboxfeatures[(vid,fid,pid)][1][0]/368.0 # Here, two zeros since the feature is (1,30) at position 1 in tuple.
                feature_mat[fid-fid_first,humans.index(pid),self.C:-2]=posf
                feature_mat[fid-fid_first,humans.index(pid),-2:]=torch.tensor(center)
                heights[fid-fid_first,humans.index(pid)]=person_height
        # Find all the interaction and group detection labels. 
        # The interaction label could be converted into a temporal graph connectivity matrix in shape of T,V,V,A. Here, A is the number of the interactions.
        interaction_gt_mat=torch.zeros(self.T,human_num,human_num)
        avoid_gt_mat=torch.zeros(self.T,human_num,human_num)
        group_gt_mat=torch.zeros(human_num,human_num)
        for person in pid_vfp_dict.keys():
            for (vid,fid,pid) in pid_vfp_dict[person]:
                if pid in self.vid_pid_fid_interaction_dict[vid]:
                    for fid in fids:
                        if fid in self.vid_pid_fid_interaction_dict[vid][pid]:
                            interaction_info=self.vid_pid_fid_interaction_dict[vid][pid][fid]
                            if interaction_info['is_interaction']:
                                for interacted_person in interaction_info['interaction_person_id']:
                                    if interacted_person not in humans:
                                            continue # Oops, a man hasn't in the sampled frames is the interacted one
                                    for interaction_type in interaction_info['interaction_type']:
                                        interaction_gt_mat[fid-fid_first,humans.index(pid),humans.index(interacted_person)]=1+INTERACTION_TYPES.index(interaction_type)
                            if interaction_info['is_avoid']:
                                for avoided_person in interaction_info['avoid_person_id']:
                                        if avoided_person not in humans:
                                            continue
                                        for avoid_type in interaction_info['avoid_type']:
                                            avoid_gt_mat[fid-fid_first,humans.index(pid),humans.index(avoided_person)]=1+AVOID_TYPES.index(avoid_type)
        # The group detection labels are annotated in video-level. Thus, it is one matrix per video, V,V,B.  B is the number of the group types.
        pid_group_info=self.vid_pid_group_dict[vid]
        if pid_group_info: # is not empty
            for person in pid_vfp_dict.keys():
                if person in pid_group_info:
                    for peer in pid_group_info[person]['members']:
                        if peer in humans:
                            group_gt_mat[humans.index(person),humans.index(peer)]=1+GROUP_TYPES.index(pid_group_info[person]['relation'])
        
        # return features: T,V,C+2, interaction labels: None or T,V,V,A. It is like a multi-class segmentation. group detection labels: None or V,V
        A=torch.zeros_like(feature_mat)
        A=torch.sqrt((((feature_mat[:,:,None,-2:]-feature_mat[:,None,:,-2:])*100)**2).sum(-1))
        A=1/A
        # A.clamp_max_(1)
        A=torch.nan_to_num(A,0,1)
        A[A<0.02]=0
        # A=A+torch.eye(human_num)[None,:,:]
        # A[torch.isinf(A)]=0
        # A[A!=A]=0
        # A=A+torch.eye(human_num)[None,:,:]
        T,V,C_=feature_mat.shape
        
        if self.aug=='train':
            mask_mat=torch.zeros(T,V,1)
            aug_feature_mat=feature_mat.clone()
            R=torch.eye(human_num).repeat(self.T,1,1)
            random_num=random.randint(1,human_num//10)
            for i in range(random_num):
                aug_feature_mat,mask_mat,R=self.augment(aug_feature_mat,mask_mat,R)
            
            A_=torch.zeros_like(aug_feature_mat)
            A_=torch.sqrt((((aug_feature_mat[:,:,None,-2:]-aug_feature_mat[:,None,:,-2:])*100)**2).sum(-1))
            A_=1/A_
            A_=torch.nan_to_num(A_,0,1)
            # A_[A_<0.02]=0
            # A_=A_+torch.eye(human_num)[None,:,:]
            feature_mat=feature_mat[:,:,:-2]
            feature_mat=torch.cat([feature_mat,mask_mat],dim=-1)
            aug_feature_mat=aug_feature_mat[:,:,:-2]
            aug_feature_mat=torch.cat([aug_feature_mat,mask_mat],dim=-1)
            return aug_feature_mat,feature_mat,A,interaction_gt_mat,avoid_gt_mat,group_gt_mat,heights,A_,R.float()
        elif self.aug=='distill':
            # mask_mat=torch.zeros(T,V,1)
            # aug_feature_mat=feature_mat.clone()
            # who=np.random.choice(V,1,replace=False)[0]
            # aug_feature_mat,who=self.augment_kill(aug_feature_mat)
            # aug_feature_mat=torch.cat([aug_feature_mat,mask_mat],dim=-1)
            # mask_mat[:,who,0]=1
            # feature_mat=torch.cat([feature_mat,mask_mat],dim=-1)
            '''
            Here, I want to let the train function to decide which one is the star.
            '''
            return feature_mat,A,interaction_gt_mat,avoid_gt_mat,group_gt_mat,heights
        else:
            feature_mat=feature_mat[:,:,:-2]
            # feature_mat=torch.cat([feature_mat,torch.ones(T,V,1)],dim=-1) # Add mask to the final channel.
            index_mat=torch.zeros(T,V,3)
            for t in range(T):
                for v in range(V):
                    index_mat[t,v,0]=vid
                    index_mat[t,v,1]=fids[t]
                    index_mat[t,v,2]=humans[v]
            return feature_mat,A,interaction_gt_mat,avoid_gt_mat,group_gt_mat,index_mat,heights
        
    def augment(self,feature_mat:torch.Tensor,mask_mat:torch.Tensor,R):
        # 1. swap different V -> learn the common sense of human roles in the crowd. Only for the last two positional dimensions.
        _,V,_=feature_mat.shape
        # forward or backward one's feature along time. -> learn the reasonable temporal actions in the crowd.
        def self_quicker_slower(a,T):
            temp=feature_mat[:,a,:].clone()
            new_feature=torch.zeros_like(temp)
            if T>0: # forward
                new_feature[:-T,:]=temp[T:,:]
            elif T<0:
                new_feature[-T:,:]=temp[:T,:]
            feature_mat[:,a,:]=new_feature
            mask_mat[:,a,0]=1
        
        def self_shift(a,D):
            feature_mat[:,a,-2:]+=D
            mask_mat[:,a,0]=1
        
        # def self_disappear(a,repeat_time=3):
        #     for _ in range(repeat_time):
        #         rand_t=random.randint(0,self.T-1)
        #         feature_mat[rand_t,a,:]=0
        #     mask_mat[:,a,0]=1
        
        # def self_backward(a):
        #     feature_mat[:,a,:]=feature_mat[:,a,:].flip(0)
        #     mask_mat[:,a,0]=1 # Let the time flow backward.
        
        def social_swap_roles(a,b):
            temp=feature_mat[:,b,self.C:].clone()
            feature_mat[:,b,self.C:]=feature_mat[:,a,self.C:]        
            feature_mat[:,a,self.C:]=temp
            mask_mat[:,b,0]=1
            mask_mat[:,a,0]=1
            R[:,a,a]=0
            R[:,a,b]=1
            R[:,b,a]=1
            R[:,b,b]=0

        # 3. combine both of them.
        # random_seed=random.uniform(0,1)
        randa,randb=np.random.choice(V,2,replace=False)
        # random_t=random.randint(-self.T//2,self.T//2)
        # random_D=random.uniform(-0.1,0.1) # Note! This may cause out of bound error in visualization! 
        # if random_seed<1/2:
        social_swap_roles(randa,randb)
        social_swap_roles(randa,randb)
        self_quicker_slower(randa,random.randint(-self.T//4,self.T//4))
        self_shift(randa,random.uniform(-0.1,0.1))
        self_quicker_slower(randb,random.randint(-self.T//4,self.T//4))
        self_shift(randb,random.uniform(-0.1,0.1))
        # if random_seed<1/2:
        #     self_quicker_slower(randa,random_t)
        # if random_seed<1/2:
        #     self_shift(randa,random_D)
        # if random_seed<1/2:
        #     self_disappear(randa)
        # if random_seed<1/2:
        #     self_backward(randa)
        
        return feature_mat,mask_mat,R
    
    def augment_kill(self,feature_mat:torch.Tensor):
        # 1. swap different V -> learn the common sense of human roles in the crowd. Only for the last two positional dimensions.
        _,V,_=feature_mat.shape
        randa=np.random.choice(V,1,replace=False)[0]
        feature_mat[:,randa,:]=0
        return feature_mat,randa
    
if __name__=='__main__':
    dataset=PANDA(YOURPATH)
    print('debug')