from shutil import ExecError
from typing_extensions import Annotated
import numpy as np
import os
import sys
from os.path import join
from os import listdir
import torch
import cv2
import json

'''
{
        'A': body touch; 
        'B': talk; 
        'C': pose change & body language; 
        'D': location change; 
        'E': other interaction; 
        'F': unobvious interaction
    }
{
    	'A': front avoiding;
        'B': side or back avoiding;
        'C': stop or slow down;
        'D': other avoidance
	}
'''
GROUP_TYPES=['acquaintance','family','business']#['acquaintance','family','business']
INTERACTION_TYPES=['A','B','C','D','E','F']
AVOID_TYPES=['A','B','C','D']


class PANDA:
    def __init__(self,root:str,target_videos:list) -> None:
        self.root=root
        self.img_path=join(root,'PANDA_IMAGE')
        self.group_anno_path=join(root,'PANDA_Video_groups_and_interactions/train/group_annotation/')
        self.interaction_anno_path=join(root,'PANDA_Video_groups_and_interactions/train/interaction_annotation/')
        self.video_anno_path=join(root,'video_annos')
        self.video_test_path=join(root,'video_test')
        self.video_train_path=join(root,'video_train')
        # Since they only provide annotations for part of the data, we need a subset index list.
        self.group_interaction_indexes=['07_University_Campus', '10_Huaqiangbei', '09_Xili_Street_2', '01_University_Canteen', '08_Xili_Street_1', '03_Xili_Crossroad', '02_OCT_Habour']
        # Extract features of all the videos and save their features depend on the key of vid_fid_pid, e.g., (1,3,3)
        # We only extract features for the training set, since we need tracks.json.
        # The features include (x,y) and an action feature vector.
        # self.target_videos=listdir(self.video_train_path) # We need to specify the videos.
        self.target_videos=target_videos
        self.target_vids=[int(i.split('_')[0]) for i in self.target_videos]
        self.annotation_dict=self.read_all_video_anno()
        self.convert_track_dict_to_frameid_dict()
        
    def convert_track_dict_to_frameid_dict(self):
        '''
        Will add a new attribute frame_dict to self.
        '''
        frame_dict={} # (vid,fid):
        for vid in self.annotation_dict.keys():
            track_list=self.annotation_dict[vid]['track']
            for pdict in track_list:
                pid=pdict['track id']
                frames=pdict['frames']
                for fdict in frames:
                    fid=fdict['frame id']
                    informations={}
                    for _k in fdict.keys():
                        if _k!='frame id':
                            informations[_k]=fdict[_k]
                    if (vid,fid) not in frame_dict:
                        frame_dict[(vid,fid)]={}
                    # if pid in frame_dict[(vid,fid)].keys():
                    #     print('WARNING! Detected one person id existed in the same frame for more than once. Vid:%d, fid:%d, pid:%d'%(vid,fid,pid))
                    frame_dict[(vid,fid)][pid]=informations
        self.frame_dict=frame_dict
    
    def read_one_video(self,vid:int):
        vpath=join(self.video_train_path,self.target_videos[self.target_vids.index(vid)])
        fpaths=listdir(vpath)
        fpaths.sort()
        getfid=lambda x:int(x.split('.')[0].split('_')[2]) # 'seqLength': 234, 'imWidth': 26753, 'imHeight': 15052,
        L=self.annotation_dict[vid]['info']['seqLength']
        imH,imW=self.annotation_dict[vid]['info']['imHeight'],self.annotation_dict[vid]['info']['imWidth']
        already_dict={}
        # need info
        for f in fpaths:
            fid=getfid(f)
            img=cv2.imread(join(vpath,f)) # H,W,3
            # need pid, track pid.
            for pid in self.frame_dict[(vid,fid)].keys():
                if (vid,fid,pid) in already_dict:
                    print('Found duplicated vid,fid,pid, and ignore it.',(vid,fid,pid))
                    continue
                bbox=self.frame_dict[(vid,fid)][pid]['rect'] #{'tl': {'y': 0.927410368, 'x': -0.0297653534}, 'br': {'y': 1.0916774972, 'x': 0.0076305695}}
                bbox= [bbox['tl']['x'],bbox['tl']['y'],bbox['br']['x'],bbox['br']['y']]  # xtl,ytl,xbr,ybr
                # given bbox, info, image, decide the crop area and resize size (224,224). give it to the network. return image.
                reg=lambda x: max(min(x,1.0),0.0)
                bbox= [reg(i) for i in bbox]
                xtl,ytl,xbr,ybr=bbox
                xtl,xbr=xtl*imW,xbr*imW
                ytl,ybr=ytl*imH,ybr*imH
                p_H=ybr-ytl
                p_W=xbr-xtl
                R=max(p_H,p_W)
                if (xtl-(R-p_W)/2)<0:
                    cropxtl=0
                    cropxbr=R
                else:
                    cropxtl=(xtl-(R-p_W)/2)
                    cropxbr=cropxtl+R
                if cropxbr>=imW:
                    cropxbr=imW
                    cropxtl=cropxbr-R
                if (ytl-(R-p_H)/2)<0:
                    cropytl=0
                    cropybr=R
                else:
                    cropytl=(ytl-(R-p_H)/2)
                    cropybr=cropytl+R
                if cropybr>=imH:
                    cropybr=imH
                    cropytl=cropybr-R
                crop_image=img[int(cropytl):int(cropybr),int(cropxtl):int(cropxbr),:]
                try:
                    crop_image=cv2.resize(crop_image,(288,288))
                except:
                    print('Find empty image again! skip. Vid,fid,pid:',(vid,fid,pid))
                    continue
                pbbox=[(xtl-cropxtl)/R,(ytl-cropytl)/R,(xbr-cropxtl)/R,(ybr-cropytl)/R]
                if (pbbox[3]-pbbox[1])==0 or (pbbox[2]-pbbox[0])==0:
                    print('WARNING! Detect empty bboxes and is skipped. Vid, fid, pid:%d,%d,%d'%(vid,fid,pid))
                    continue
                already_dict[(vid,fid,pid)]=None
                yield (vid,fid,pid),(crop_image,pbbox) # key, value
    
    def read_all_video_anno(self):
        res_dict={}
        for i,(vid_key,vid) in enumerate(zip(self.target_videos,self.target_vids)):
            tp=join(self.video_anno_path,vid_key,'tracks.json')
            ifp=join(self.video_anno_path,vid_key,'seqinfo.json')
            gp=join(self.group_anno_path,'annotator_1',vid_key+'.json')
            itp=join(self.interaction_anno_path,'annotator_1',vid_key+'.json')
            res_dict[vid]=self.read_one_video_anno(gp,itp,tp,ifp)
        return res_dict
            
    @staticmethod
    def read_one_video_anno(gpath:str,itpath:str,tpath:str,seq_info_path:str):
        if os.path.exists(gpath):
            with open(gpath,'r') as load_g:
                group_anno=json.load(load_g)
        else:
            group_anno=None
        if os.path.exists(itpath):
            with open(itpath,'r') as load_it:
                interaction_anno=json.load(load_it)
        else:
            interaction_anno=None
        with open(tpath,'r') as load_t:
            track=json.load(load_t)
        with open(seq_info_path,'r') as load_i:
            info=json.load(load_i)
        print('One video read.')
        return {'info':info, 'group_anno':group_anno,'interaction_anno':interaction_anno,'track':track}
    
    
        
if __name__=='__main__':
    tp='/home/lijiacheng/hdd2/PANDA/video_annos/01_University_Canteen/tracks.json'
    ifp='/home/lijiacheng/hdd2/PANDA/video_annos/01_University_Canteen/seqinfo.json'
    gp='/home/lijiacheng/hdd2/PANDA/PANDA_Video_groups_and_interactions/train/group_annotation/annotator_1/01_University_Canteen.json'
    itp='/home/lijiacheng/hdd2/PANDA/PANDA_Video_groups_and_interactions/train/interaction_annotation/annotator_1/01_University_Canteen.json'
    PANDA.read_one_video_anno(gp,itp,tp,ifp)