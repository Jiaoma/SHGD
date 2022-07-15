from numpy.lib.type_check import imag
import torch
import numpy as np
from torchvision.ops import roi_align
import os
from tqdm import tqdm
from config import YOURPATH,SAVEPATH
from unipose import Trainer

def check_state_dict(load_dict, new_dict):
    # check loaded parameters and created model parameters
    for k in new_dict:
        if k in load_dict:
            if new_dict[k].shape != load_dict[k].shape:
                print('Skip loading parameter {}, required shape{}, '
                      'loaded shape{}.'.format(
                          k, load_dict[k].shape, new_dict[k].shape))
                new_dict[k] = load_dict[k]
        else:
            print('Drop parameter {}.'.format(k))
    for k in load_dict:
        if not (k in new_dict):
            print('No param {}.'.format(k))
            new_dict[k] = load_dict[k]


def load_model(model, model_path, optimizer=None, lr=None, ucf_pretrain=False):
    start_epoch = 0
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    print('loaded {}, epoch {}'.format(model_path, checkpoint['epoch']))
    state_dict_ = checkpoint['state_dict']
    state_dict = {}

    # convert data_parallal to model
    for k in state_dict_:
        if ucf_pretrain:
            if k.startswith('branch.hm') or k.startswith('branch.mov'):
                continue
        if k.startswith('module') and not k.startswith('module_list'):
            state_dict[k[7:]] = state_dict_[k]
        else:
            state_dict[k] = state_dict_[k]
    check_state_dict(model.state_dict(), state_dict)
    model.load_state_dict(state_dict, strict=False)

    # resume optimizer parameters
    if optimizer is not None:
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            print('Resumed optimizer with start lr', lr)
        else:
            print('No optimizer parameters in checkpoint.')

    if 'best' in checkpoint:
        best = checkpoint['best']
    else:
        best = 100
    if optimizer is not None:
        return model, optimizer, start_epoch, best
    else:
        return model


def extract_PANDA_skeleton_features():
    model = Trainer()

    from visdom import Visdom
    import time
    vis=Visdom()
    group_vid=['07_University_Campus.json', '05_Basketball_Court.json', '10_Huaqiangbei.json', '09_Xili_Street_2.json', '01_University_Canteen.json', '08_Xili_Street_1.json', '03_Xili_Crossroad.json', '02_OCT_Habour.json', '04_Primary_School.json', '06_Xinzhongguan.json']
    group_vid.sort()
    group_vid=[int(i.split('.')[0].split('_')[0]) for i in group_vid]
    interaction_vid=['07_University_Campus.json', '10_Huaqiangbei.json', '09_Xili_Street_2.json', '01_University_Canteen.json', '08_Xili_Street_1.json', '03_Xili_Crossroad.json', '02_OCT_Habour.json']
    interaction_vid.sort()
    interaction_vid=[int(i.split('.')[0].split('_')[0]) for i in interaction_vid]

    # 7 x 256 x 100 x 1024 channels == 8 GB data.
    print('Start to process data')
    try:
        from panda import PANDA
    except:
        from .panda import PANDA
    # Change the following path before you run the py!!

    dataset=PANDA(os.path.join(YOURPATH,'PANDA'),['01_University_Canteen', '02_OCT_Habour', '03_Xili_Crossroad', '04_Primary_School', '05_Basketball_Court', '06_Xinzhongguan', '07_University_Campus', '08_Xili_Street_1', '09_Xili_Street_2', '10_Huaqiangbei'])
    big_dict_vfp_bboxfeatures={}
    count=0
    for vid in tqdm(dataset.target_vids):
        for key,value in dataset.read_one_video(vid):
            # key: vid,fid,pid
            # value: image, bbox
            image,bbox=value
            image=image.astype(np.float32)
            # image=np.transpose(image,(2,0,1))
            # image=(image/255.0-mean)/std
            # image=torch.from_numpy(image).reshape(1,1,3,H,W).cuda()
            boxes_in_flat=torch.Tensor(bbox).reshape(-1,4).cuda()
            # images_in_flat = torch.reshape(images_in, (B * T*F, 3, H, W))  # B*T, 3, H, W
            # boxes_in_flat = boxes_in.permute(0,1,4,2,3).reshape(B*T*F*MAX_N,4)
            # boxes_in_flat=torch.split(boxes_in_flat,MAX_N,dim=0)
            boxes_in_flat=(boxes_in_flat,)
            
            # Use backbone to extract features of images_in
            # Pre-precess first
            with torch.no_grad():
                # features = model(image)
                if count%100==0:
                    features=model.test(image,0,vis,key[1],key[2]) # 16,2 -> 1,32  
                else:
                    features=model.test(image,0) # 16,2 -> 1,32  
                # features = features.reshape(1,64,72,72)  # B*T*F, D, OH, OW
                # pose_features = roi_align(features,boxes_in_flat,crop_size)
                # pose_features = pose_features.reshape(-1,64,R,R).reshape(-1,64*R*R)
                pose_features=features
                # time.sleep(1)
            big_dict_vfp_bboxfeatures[key]=(bbox,pose_features.detach().cpu()) # 1, 2306
            count+=1
            if count%100==0:
                print('Already collect %d vectors, video %d'%(count,vid))

    torch.save(big_dict_vfp_bboxfeatures,os.path.join(SAVEPATH,'PANDA','train_all_features.pth.tar'))
    print('Collect %d samples for all'%len(big_dict_vfp_bboxfeatures.keys()))
    remove_keys_none=[]
    for key in big_dict_vfp_bboxfeatures.keys():
        if key[0] not in group_vid:
            remove_keys_none.append(key)
            
    for k in remove_keys_none: del big_dict_vfp_bboxfeatures[key]

    torch.save(big_dict_vfp_bboxfeatures,os.path.join(SAVEPATH,'PANDA','train_group_interaction_features.pth.tar'))
    print('Collect %d samples for group and interaction'%len(big_dict_vfp_bboxfeatures.keys()))
    # big_dict_vfp_bboxfeatures=torch.load(os.path.join(SAVEPATH,'PANDA','train_all_features.pth.tar'))
    remove_keys_interaction=[]
    for key in big_dict_vfp_bboxfeatures.keys():
        if key[0] not in interaction_vid:
            remove_keys_interaction.append(key)
            
    for k in remove_keys_interaction: del big_dict_vfp_bboxfeatures[k]

    torch.save(big_dict_vfp_bboxfeatures,os.path.join(SAVEPATH,'PANDA','train_interaction_features.pth.tar'))
    print('Collect %d samples for interaction'%len(big_dict_vfp_bboxfeatures.keys()))



if __name__=='__main__':
    extract_PANDA_skeleton_features()
