import time
import os
from utils import readJson
import numpy as np

class Config(object):
    """
    class to save config parameter
    """

    def __init__(self, dataset_name):
        # Global
        self.ori_image_size=1088,1920
        self.image_size = 720, 1280  #input image size
        self.batch_size =  32  #train batch size 
        self.test_batch_size = 8  #test batch size
        self.num_boxes = 12  #max number of bounding boxes in each frame
        self.ablation=False
        self.num_workers=12
        self.num_joints=16

        config_file=readJson("config.json")
        self.config_file=config_file
        
        # Gpu
        self.use_gpu=True
        self.use_multi_gpu=False
        self.device_list="1,"  #id list of gpus used for training
        
        # Dataset
        self.dataset_name=dataset_name
        self.force=False
        
        self.data_path=config_file['Datasets'][dataset_name]['root']
        self.output_path=self.config_file['Datasets'][dataset_name]['output']
        # Backbone 
        self.backbone='inv3' 
        self.crop_size = 5, 5  #crop size of roi align
        self.train_backbone = False  #if freeze the feature extraction part of network, True for stage 1, False for stage 2
        self.out_size = 87, 157  #output feature map size of backbone 
        self.emb_features=288 #185   #output feature map channel of backbone
        self.compress_emb_features=8

        
        # Activity Action
        self.num_actions = 7  #number of action categories
        self.num_activities = 6  #number of activity categories
        self.actions_loss_weight = 1  #weight used to balance action loss and activity loss
        self.actions_weights = None

        # Sample
        self.K=5
        self.num_frames = 3 
        self.num_before = 5
        self.num_after = 4
        self.continuous_frames = 5
        self.interval=3

        # GCN
        self.num_features_pose = 6
        self.num_features_trace = 6
        self.num_features_boxes = 32
        self.num_features_relation=256
        self.num_features_relation_app=256
        self.num_features_relation_mt=128
        self.num_graph=16  #number of graphs
        self.num_features_gcn=self.num_features_boxes
        self.gcn_layers=1  #number of GCN layers
        self.tau_sqrt=False
        self.pos_threshold=0.2  #distance mask threshold in position relation

        # Training Parameters
        self.train_random_seed = 0
        self.train_learning_rate = 2e-4  #initial learning rate 
        self.lr_plan = {41:1e-4, 81:5e-5, 121:1e-5}  #change learning rate in these epochs 
        self.train_dropout_prob = 0.2  #dropout probability
        self.weight_decay = 0  #l2 weight decay
    
        self.max_epoch=150  #max training epoch
        self.test_interval_epoch=10
        
        # Exp
        self.training_stage=1  #specify stage1 or stage2
        self.stage1_model_path=''  #path of the base model, need to be set in stage2
        self.test_before_train=False
        self.exp_note='Latent Relation Discovery'
        self.exp_name=None
        
        
    def init_config(self, need_new_folder=True):
        if self.exp_name is None:
            time_str=time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
            self.exp_name='[%s_stage%d]<%s>'%(self.exp_note,self.training_stage,time_str)
        
        self.result_path=os.path.join(self.output_path,self.exp_name)
        self.log_path=os.path.join(self.output_path,'%s/log.txt'%self.exp_name)
            
        if need_new_folder:
            os.mkdir(self.result_path)
            
            
if __name__=='__main__':
    train_seqs=[101, 106, 111, 115, 120, 125, 130, 135, 140, 145, 149, 154, 159, 164, 169, 174, 179, 391, 397, 402, 407, 412, 417, 421, 426, 431, 436, 441, 446, 451, 455, 460, 465, 470, 475, 480, 485, 85, 91, 96, 103, 109, 114, 118, 123, 132, 137, 142, 147, 151, 153, 157, 162, 167, 172, 177, 182, 394, 400, 405, 410, 415, 420, 424, 429, 434, 439, 444, 449, 454, 458, 462, 467, 472, 477, 482, 487, 87, 93, 98, 11, 15, 19, 23, 27, 310, 317, 321, 325, 329, 333, 344, 350, 358, 362, 366, 372, 38, 386, 42, 46, 52, 56, 60, 621, 625, 629, 633, 637, 64, 650, 656, 66, 664, 668, 672, 678, 686, 70, 80, 10, 16, 22, 312, 318, 322, 328, 34, 345, 349, 359, 363, 369, 373, 377, 381, 385, 39, 43, 53, 57, 61, 622, 626, 63, 634, 646, 651, 655, 665, 669, 673, 677, 681, 685, 689, 691, 71, 75, 79, 184, 187, 190, 193, 197, 202, 205, 208, 211, 213, 216, 219, 224, 229, 232, 235, 240, 243, 246, 249, 490, 493, 496, 499, 503, 508, 511, 514, 517, 519, 522, 525, 530, 535, 538, 541, 546, 549, 552, 555, 252, 256, 259, 261, 263, 267, 270, 273, 276, 279, 282, 284, 286, 289, 291, 294, 298, 300, 302, 304, 558, 562, 565, 567, 569, 573, 576, 579, 582, 585, 588, 590, 592, 595, 597, 600, 604, 606, 608, 610]
    test_seqs=[102, 107, 112, 116, 121, 126, 131, 136, 141, 146, 150, 155, 160, 165, 170, 175, 180, 392, 398, 403, 408, 413, 418, 422, 427, 432, 437, 442, 447, 452, 456, 461, 466, 471, 476, 481, 486, 86, 92, 97, 108, 113, 117, 122, 128, 133, 138, 143, 148, 152, 156, 161, 166, 171, 176, 181, 393, 399, 404, 409, 414, 419, 423, 428, 433, 438, 443, 448, 453, 457, 459, 463, 468, 473, 478, 483, 488, 88, 94, 99, 13, 17, 21, 25, 31, 315, 319, 323, 327, 331, 337, 348, 354, 360, 364, 370, 376, 380, 4, 44, 48, 54, 58, 616, 623, 627, 631, 635, 639, 643, 654, 658, 660, 666, 670, 676, 682, 692, 74, 9, 14, 20, 24, 316, 320, 326, 330, 340, 347, 351, 361, 367, 371, 375, 379, 383, 387, 41, 45, 55, 6, 618, 624, 628, 632, 636, 65, 653, 657, 667, 67, 675, 679, 683, 687, 69, 693, 73, 77, 81, 186, 188, 191, 194, 201, 203, 206, 209, 212, 214, 217, 220, 228, 231, 234, 238, 241, 244, 247, 250, 492, 494, 497, 500, 507, 509, 512, 515, 518, 520, 523, 526, 534, 537, 540, 544, 547, 550, 553, 556, 254, 257, 260, 262, 265, 268, 272, 274, 277, 280, 283, 285, 287, 290, 292, 296, 299, 301, 303, 305, 560, 563, 566, 568, 571, 574, 578, 580, 583, 586, 589, 591, 593, 596, 598, 602, 605, 607, 609, 611]
    from random import shuffle
    from functools import reduce
    action_seqs={0:[],1:[],2:[],3:[],4:[],5:[]} 
    for i in range(240): 
        a=i//40 
        action_seqs[a].append(train_seqs[i]) 
        action_seqs[a].append(test_seqs[i]) 
    
    action_same_seqs={0:[],1:[],2:[],3:[],4:[],5:[]}
    for ac in action_seqs.keys():
        action_seqs[ac].sort()
    
    for ac in action_same_seqs.keys():
        select=[]
        selected=[]
        for seq in action_seqs[ac]:
            if seq not in selected:
                select.append(seq)
                selected.append(seq)
                if seq+306 in action_seqs[ac] and seq+306 not in selected:
                    select.append(seq+306)
                    selected.append(seq+306)
                if seq+306*2 in action_seqs[ac] and seq+306*2 not in selected:
                    select.append(seq+306*2)
                    selected.append(seq+306*2)
                action_same_seqs[ac].append(select)
        shuffle(action_same_seqs[ac])
        action_same_seqs[ac]=reduce(lambda x,y:x+y,action_same_seqs[ac])
    # print(action_same_seqs[5])
    # print(action_same_seqs[4])
    import numpy as np 
    new_train_seqs,new_test_seqs=[],[]
    # for key in action_same_seqs.keys(): 
    for key in [2,3]: 
        actions=action_same_seqs[key] 
        indexs=np.random.choice(80,80,replace=False) 
        for i in indexs[:40]: 
            new_train_seqs.append(actions[i]) 
        for i in indexs[40:]: 
            new_test_seqs.append(actions[i])
    new_train_seqs.sort()
    new_test_seqs.sort()
    print(new_train_seqs)
    print(new_test_seqs)