# -*-coding:UTF-8-*-
import argparse
import time
import torch.optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
import sys
import numpy as np
import cv2
import math
sys.path.append("..")
from UniPose.utils.utils import get_model_summary
from UniPose.utils.utils import adjust_learning_rate as adjust_learning_rate
from UniPose.utils.utils import save_checkpoint      as save_checkpoint
from UniPose.utils.utils import printAccuracies      as printAccuracies
from UniPose.utils.utils import guassian_kernel      as guassian_kernel
from UniPose.utils.utils import get_parameters       as get_parameters
from UniPose.utils       import Mytransforms         as  Mytransforms 
from UniPose.utils.utils import getDataloader        as getDataloader
from UniPose.utils.utils import getOutImages         as getOutImages
from UniPose.utils.utils import AverageMeter         as AverageMeter
from UniPose.utils.utils import draw_paint           as draw_paint
from UniPose.utils       import evaluate             as evaluate
from UniPose.utils.utils import get_kpts             as get_kpts

from UniPose.model.unipose import unipose

from tqdm import tqdm

import torch.nn.functional as F
from collections import OrderedDict
from torchsummary import summary

from PIL import Image


class Trainer(object):
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--pretrained', default=None,type=str, dest='pretrained')
        parser.add_argument('--dataset', type=str, dest='dataset', default='MPII')
        parser.add_argument('--train_dir', default='/PATH/TO/TRAIN',type=str, dest='train_dir')
        parser.add_argument('--val_dir', type=str, dest='val_dir', default='/PATH/TO/LSP/VAL')
        parser.add_argument('--model_name', default=None, type=str)
        parser.add_argument('--model_arch', default='unipose', type=str)

        args = parser.parse_args()

        if args.dataset == 'LSP':
            args.train_dir  = '/PATH/TO/LSP/TRAIN'
            args.val_dir    = '/PATH/TO/LSP/VAL'
            args.pretrained = './UniPose/UniPose_LSP.tar'
        elif args.dataset == 'MPII':
            args.train_dir  = '/PATH/TO/MPIII/TRAIN'
            args.val_dir    = '/PATH/TO/MPIII/VAL'
            args.pretrained = './UniPose/UniPose_MPII.pth'
        self.args         = args
        self.train_dir    = args.train_dir
        self.val_dir      = args.val_dir
        self.model_arch   = args.model_arch
        self.dataset      = args.dataset


        self.workers      = 1
        self.weight_decay = 0.0005
        self.momentum     = 0.9
        self.batch_size   = 8
        self.lr           = 0.0001
        self.gamma        = 0.333
        self.step_size    = 13275
        self.sigma        = 3
        self.stride       = 8

        cudnn.benchmark   = True

        if self.dataset   ==  "LSP":
            self.numClasses  = 14
        elif self.dataset == "MPII":
            self.numClasses  = 15


        model = unipose(self.dataset, num_classes=self.numClasses,backbone='resnet',output_stride=16,sync_bn=True,freeze_bn=False, stride=self.stride)

        self.model       = model.cuda()


        if self.args.pretrained is not None:
            checkpoint = torch.load(self.args.pretrained)
            try:
                p = checkpoint['state_dict']
            except:
                p=checkpoint

            state_dict = self.model.state_dict()
            model_dict = {}

            for k,v in p.items():
                if k in state_dict:
                    model_dict[k] = v

            state_dict.update(model_dict)
            self.model.load_state_dict(state_dict)
        self.model.share_memory()
        # Print model summary and metrics
        dump_input = torch.rand((1, 3, 368, 368))
        print(get_model_summary(self.model, dump_input.cuda()))

    def test(self,oimg:np.ndarray,idx,vis=None,fid=None,pid=None):
        '''
        img: HxWx3
        '''
        self.model.eval()

        center   = [184, 184]

        oimg  = np.array(cv2.resize(oimg,(368,368)))
        img  = oimg.transpose(2, 0, 1).astype(np.float32)
        img  = torch.from_numpy(img)
        mean = [128.0, 128.0, 128.0]
        std  = [256.0, 256.0, 256.0]
        for t, m, s in zip(img, mean, std):
            t.sub_(m).div_(s)

        img       = torch.unsqueeze(img, 0)

        # self.model.eval()

        input_var   = img.cuda()

        heat = self.model(input_var)

        heat = F.interpolate(heat, size=input_var.size()[2:], mode='bilinear', align_corners=True)

        kpts = get_kpts(heat, img_h=368.0, img_w=368.0)
        if vis is not None:
            draw=draw_paint(oimg, kpts, idx, 0, self.model_arch, self.dataset) #TODO: DEBUG from this sentence
            
            font = cv2.FONT_HERSHEY_SIMPLEX

            # org
            org = (50, 50)
            
            # fontScale
            fontScale = 1
            
            # Blue color in BGR
            color = (255, 0, 0)
            
            # Line thickness of 2 px
            thickness = 2
            
            # Using cv2.putText() method
            image = cv2.putText(draw, '%d_%d'%(fid,pid), org, font, 
                            fontScale, color, thickness, cv2.LINE_AA)
            vis.image(image.transpose(2,0,1))
            
            heat = heat.detach().cpu().numpy()

            heat = heat[0].transpose(1,2,0)


            for i in range(heat.shape[0]):
                for j in range(heat.shape[1]):
                    for k in range(heat.shape[2]):
                        if heat[i,j,k] < 0:
                            heat[i,j,k] = 0
                        

            # im       = cv2.resize(oimg,(368,368))

            # heatmap = []
            # for i in range(self.numClasses+1):
            #     heatmap = cv2.applyColorMap(np.uint8(255*heat[:,:,i]), cv2.COLORMAP_JET)
            #     im_heat  = cv2.addWeighted(im, 0.6, heatmap, 0.4, 0)
            #     cv2.imwrite('samples/heat/unipose'+str(i)+'.png', im_heat)

            #end
        return torch.from_numpy(np.asarray(kpts).reshape(1,32)).float()
    


        



if __name__=='__main__':
    img=cv2.imread('/home/molijuly/download.jpg')
    train=Trainer()
    kpts=train.test(img,0) # 15,2 -> 1,30
    print(len(kpts))