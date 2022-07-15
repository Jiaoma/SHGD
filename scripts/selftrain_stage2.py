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

cfg.model_name='PANDA_stage2_V3Net'
cfg.train_test_split='91'
cfg.core_task='Group'
cfg.extra='wS_rebuttal_draw'

cfg.num_workers=6
cfg.device_list = "0"
cfg.use_multi_gpu = False
cfg.training_stage = 2
cfg.distill=False

cfg.train_backbone = True
cfg.test_before_train = True
cfg.test_interval_epoch=10
cfg.all_features_path='/home/molijuly/fast/PANDA/train_all_features.pth.tar'
cfg.group_interaction_path='/home/molijuly/fast/PANDA/train_group_interaction_features.pth.tar'
cfg.interaction_path='/home/molijuly/fast/PANDA/train_interaction_features.pth.tar'
cfg.T=8 # each segment has 8 frames.
cfg.action_features=16*2
cfg.num_posf=32
cfg.num_graph=8

cfg.num_interactions=6+1
cfg.num_avoids=4+1
cfg.num_groups=1

cfg.num_features_relation=cfg.num_features_gcn=256
cfg.num_features_nodes=cfg.action_features+2

cfg.batch_size = 1
cfg.test_batch_size = 1

cfg.train_learning_rate = 7e-3
cfg.train_dropout_prob = 0
cfg.weight_decay = 0
cfg.lr_plan = {}
cfg.max_epoch = 100
cfg.stage2_model_path='/home/molijuly/stage2_epoch20_53.15%.pth'#'/home/molijuly/fast/PANDA/[PANDA_stage2_V3Net+91+Group+ours_stage2]<2022-04-27_09-40-33>/stage2_epoch10_51.13%.pth'#'/home/molijuly/fast/PANDA/[PANDA_stage2_V3Net+91+Group+new_stage2]<2021-12-28_20-30-15>/stage2_epoch10_48.06%.pth'#'/home/molijuly/fast/PANDA/[PANDA_stage2_V1Net+91+Group+onlyGroup_stage2]<2021-10-15_03-52-51>/stage2_epoch100_36.51%.pth'#'/home/molijuly/fast/Virtual_small/[Virtual_stage2_V1Net+11+Group+ver1_stage2]<2021-11-11_19-57-35>/stage2_epoch100_50.48%.pth'#'/home/molijuly/fast/PANDA/carefully/split91/distill_epoch200.pth'
# cfg.save_result_path='/home/molijuly/fast/PANDA/[PANDA_stage2_V1Net_stage2]<2021-09-02_22-30-52>/result_epoch100.pt'
# cfg.save_result_path='/home/molijuly/fast/rebuttal/panda/result_epoch10.pt'
cfg.ablation=False
cfg.iscontinue=False
cfg.continue_path='/home/molijuly/fast/PANDA/[PANDA_stage2_V1Net+91+Group+onlyGroup_stage2]<2021-10-15_03-52-51>/stage2_epoch100_36.51%.pth'#'/home/molijuly/fast/PANDA/[PANDA_stage2_V1Net+91+Group_stage2]<2021-10-02_18-02-35>/stage2_epoch100_2.17%.pth'

cfg.exp_note = '+'.join([cfg.model_name,cfg.train_test_split,cfg.core_task,cfg.extra])
# if cfg.exp_note.endswith('Group'):
cfg.use_unlabeled=True  # The unlabeled data refers to the data without interaction labels.
# else:
    # cfg.use_unlabeled=False # If the main task is interaction or avoidance detection then this option should set to be False.
cfg.stage1_model_path ='/home/molijuly/fast/PANDA/carefully/split%s/stage%d_epoch200.pth'%(cfg.exp_note.split('+')[1],1)
cfg.relation_model_path =''#'/home/molijuly/fast/PANDA/carefully/stage1_epoch200.pth'#'/home/molijuly/fast/PANDA/[PANDA_stage1_V3Net+91+Group+new_stage1]<2022-04-04_23-27-35>/stage1_epoch170.pth'#'/home/molijuly/fast/PANDA/[PANDA_stage1_V3Net+91+Group+new_stage1]<2022-02-26_21-17-46>/stage1_epoch70.pth'#'/home/molijuly/fast/PANDA/carefully/split91/stage1_epoch150.pth'#'/home/molijuly/fast/PANDA/[PANDA_stage1_V3Net+91+Group+new_stage1]<2022-02-27_05-05-18>/stage1_epoch70.pth'#'/home/molijuly/fast/PANDA/[PANDA_stage1_V3Net+91+Group+new_stage1]<2022-02-27_21-46-37>/stage1_epoch150.pth'#'/home/molijuly/fast/PANDA/[PANDA_stage1_V3Net+91+Group+new_stage1]<2022-02-26_21-17-46>/stage1_epoch70.pth'#'/home/molijuly/fast/PANDA/[PANDA_stage1_V3Net+91+Group+new_stage1]<2022-02-27_05-05-18>/stage1_epoch70.pth'#'/home/molijuly/fast/PANDA/[PANDA_stage1_V3Net+91+Group+new_stage1]<2021-12-28_15-43-25>/stage1_epoch50.pth' #'/home/molijuly/fast/PANDA/[PANDA_stage1_V3Net+91+Group+new_stage1]<2022-02-08_02-05-38>/stage1_epoch200.pth'#
# cfg.exp_note = 'PANDA_stage2_V1Net+random'
# preprocess(join(cfg.data_path, 'frames'), cfg.image_size)
train_net(cfg)

# test_net(cfg)

'''
DEBUG
'''
# from pympler.tracker import SummaryTracker
# # tracker = SummaryTracker()
# dataset=PANDADataset(cfg,augment='distill')

# params = {
#         'batch_size': cfg.batch_size,
#         'shuffle': True,
#         'num_workers': 0,
#         'pin_memory': False,
#         'drop_last':False
#     }
# training_loader = torch.utils.data.DataLoader(dataset, **params)
# for i,(aug,feature,who,interaction,group) in enumerate(training_loader):
#     print(aug.shape)
#     # tracker.print_diff()