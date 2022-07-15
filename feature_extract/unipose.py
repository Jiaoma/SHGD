# -*-coding:UTF-8-*-
import argparse
import torch.optim
import torch.backends.cudnn as cudnn
import numpy as np
import cv2
# sys.path.append("..")
from UniPose.utils.utils import get_model_summary
from UniPose.utils.utils import draw_paint           as draw_paint
from UniPose.utils.utils import get_kpts             as get_kpts

from UniPose.model.unipose import unipose


import torch.nn.functional as F


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
            args.pretrained = 'semantic/references/Skeleton/UniPose/UniPose_LSP.tar'
        elif args.dataset == 'MPII':
            args.train_dir  = '/PATH/TO/MPIII/TRAIN'
            args.val_dir    = '/PATH/TO/MPIII/VAL'
            args.pretrained = 'semantic/references/Skeleton/UniPose/UniPose_MPII.pth'
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
    # @profile
    def test(self,oimg:np.ndarray):
        '''
        img: HxWx3
        '''
        with torch.no_grad():
            self.model.eval()

            oimg  = np.array(oimg)
            h,w=oimg.shape[:2]
            img  = oimg.transpose(2, 0, 1).astype(np.float32)
            img  = torch.from_numpy(img)
            mean = [128.0, 128.0, 128.0]
            std  = [256.0, 256.0, 256.0]
            # for t, m, s in zip(img, mean, std):
                # t.sub_(m).div_(s)
            img.sub_(128.0).div_(256.0)

            img       = torch.unsqueeze(img, 0)

            # self.model.eval()

            input_var   = img.cuda()
            input_var   = F.interpolate(input_var, size=(368,368), mode='bilinear', align_corners=True)
            heat = self.model(input_var)

            heat = F.interpolate(heat, size=input_var.size()[2:], mode='bilinear', align_corners=True)

            kpts = get_kpts(heat, img_h=h, img_w=w)
            draw=draw_paint(oimg, kpts, self.dataset)
            return draw
    def __call__(self, data:np.ndarray) -> np.ndarray:
        return self.test(data)

        



if __name__=='__main__':
    img=cv2.imread('/home/molijuly/download.jpeg')
    train=Trainer()
    draw=train.test(img) # 15,2 -> 1,30
    print(draw.shape)
    print(draw.dtype)