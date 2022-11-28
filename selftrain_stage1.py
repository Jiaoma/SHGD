from train_net import train_net,test_net
import torch
from config import *
# from distant import preprocess
from panda_dataset import PANDADataset
import sys
sys.path.append(".")
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'

cfg = Config('PANDA')

cfg.model_name='PANDA_stage1_V3Net'
cfg.train_test_split='91'
cfg.core_task='Group'
cfg.extra='rebuttal'

cfg.num_workers=6
cfg.device_list = "0"
cfg.use_multi_gpu = False
cfg.training_stage = 1
cfg.distill=False

cfg.relation_model_path = ''
cfg.train_backbone = True
cfg.test_before_train = False


cfg.all_features_path='/home/molijuly/fast/PANDA/train_all_features.pth.tar'
cfg.group_interaction_path='/home/molijuly/fast/PANDA/train_group_interaction_features.pth.tar'
cfg.interaction_path='/home/molijuly/fast/PANDA/train_interaction_features.pth.tar'
cfg.T=8 # each segment has 8 frames.
cfg.action_features=16*2
cfg.num_posf=32
cfg.num_graph=8

cfg.num_interactions=6+1
cfg.num_avoids=4+1
cfg.num_groups=1+1

cfg.num_features_relation=cfg.num_features_gcn=256
cfg.num_features_nodes=cfg.action_features+2

cfg.batch_size = 1
cfg.test_batch_size = 1

cfg.train_learning_rate = 1e-4
cfg.train_dropout_prob = 0
cfg.weight_decay = 0
cfg.lr_plan = {}
cfg.max_epoch = 200
cfg.stage2_model_path=''
# cfg.save_result_path=''
cfg.ablation=False
cfg.iscontinue=False


cfg.exp_note = '+'.join([cfg.model_name,cfg.train_test_split,cfg.core_task,cfg.extra])
cfg.continue_path=''
cfg.use_unlabeled=True
cfg.stage1_model_path =''#/home/molijuly/fast/PANDA/carefully/split%s/stage%d_epoch200.pth'%(cfg.exp_note.split('+')[-2],cfg.training_stage) #'/home/molijuly/fast/PANDA/[PANDA_stage1_V1Net_stage1]<2021-08-26_17-21-02>/stage1_epoch50.pth'  # PATH OF THE BASE MODEL # Note: 2021.8.27: pt50 is better 100 is overfitted.


# preprocess(join(cfg.data_path, 'frames'), cfg.image_size)
train_net(cfg)
# test_net(cfg)