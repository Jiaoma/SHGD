import os
from os.path import join
from os import mkdir, listdir

from collections import OrderedDict

import cv2
import PIL
import numpy as np

import math

import xml.etree.ElementTree as ET

import random
from PIL import Image

import torch

import numpy as np

from collections import Counter
import os

from pathlib import Path

from os.path import exists

from copy import deepcopy
import json
import shutil

import tqdm

from typing import Dict

'''
fname = '001.jpg'
img = cv2.imread(fname)
# 画矩形框
cv2.rectangle(img, (212,317), (290,436), (0,255,0), 4)
# 标注文本
font = cv2.FONT_HERSHEY_SUPLEX
text = '001'
cv2.putText(img, text, (212, 310), font, 2, (0,0,255), 1)
cv2.imwrite('001_new.jpg', img)
'''
def drawBoundingBox(imgPath:str,bboxes:dict,savePath:str):
    img=cv2.imread(imgPath)
    img=cv2.UMat(img).get()
    font = cv2.FONT_HERSHEY_SIMPLEX
    for key in bboxes.keys():
        bbox=bboxes[key]
        bbox = [int(i) for i in bbox]
        text=str(key)
        f=lambda x:(int(x[0]),int(x[1]))
        cv2.rectangle(img,f(bbox[:2]),f(bbox[2:]), (0, 255, 0), 4)
        cv2.putText(img, text, f(bbox[:2]), font, 2, (0, 0, 255), 1)
    cv2.imwrite(savePath, img)

def safeMkdir(path: str):
    if exists(path):
        return
    else:
        mkdir(path)

# root='/home/molijuly/dep'
def saveDict(src: dict, save_path: str):
    # Notice that, here the dict is not OrderDict
    with open(save_path, 'w') as f:
        json.dump(src, f)


def readDict(path: str,key_is_int=False):
    with open(path, 'r') as f:
        d = json.load(f)
    if key_is_int:
        res = {}
        for key in d.keys():
            res[int(key)] = d[key]
        return res
    else:
        return d

def are_same(bbox_a:tuple,bbox_b_Dict:Dict[int,tuple],thred:float) -> int:
    '''
    Which one is corresponding to bbox_a? Return its key in bbox_b_dict. Thred
    is used for threthold.
    '''
    # f=lambda x:((x[0]+x[2])/2,(x[1]+x[3])/2)
    # c_a=f(bbox_a)
    # min_dis=thred*2 # Meaningless, just want a big float
    # min_index=-1
    # d=lambda x,y:math.sqrt((x[0]-y[0])**2+(x[1]-y[1])**2)
    for i,bbox_b in bbox_b_Dict.items():
        iou=get_iou(bbox_a,bbox_b)
        if iou>0.5:
            return i
    return -1

def get_iou(bb1, bb2):
    # bb1,bb2: y1,x1,y2,x2
    # determine the coordinates of the intersection rectangle
    x_left = max(bb1[1], bb2[1])
    y_top = max(bb1[0], bb2[0])
    x_right = min(bb1[3], bb2[3])
    y_bottom = min(bb1[2], bb2[2])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1[3] - bb1[1]) * (bb1[2] - bb1[0])
    bb2_area = (bb2[3] - bb2[1]) * (bb2[2] - bb2[0])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    return iou

def interpolateBboxes(latter_b: tuple, next_b: tuple, rate: float):
    la=np.asarray(latter_b)
    ne=np.asarray(next_b)
    ret=tuple(((ne-la)*rate+la).tolist())
    return ret

# Read xml in pickout
def readBboxesFromXml(xml_file):
    if type(xml_file) == str:
        xml_file = Path(xml_file)
    elif type(xml_file) == Path:
        pass
    else:
        raise TypeError
    assert xml_file.exists(), '{} does not exist.'.format(xml_file)

    tree = ET.parse(xml_file.open())
    img_name = tree.find("filename").text
    bboxes = {}
    objects = tree.findall("object")
    for object in objects:
        person_label = object.find('name').text.lower().strip()
        bbox = object.find('rectangle')
        xmin = round(float(bbox.find('xmin').text), 2)
        ymin = round(float(bbox.find('ymin').text), 2)
        xmax = round(float(bbox.find('xmax').text), 2)
        ymax = round(float(bbox.find('ymax').text), 2)
        bboxes[person_label] = (xmin, ymin, xmax, ymax)
    return bboxes

def generate_actors(v:torch.Tensor):
    # v: (...,C)
    v=v.float()
    b=torch.diag_embed(v,dim1=-2,dim2=-1) #T,N,N
    T,N,_=b.shape
    only1=b.reshape(T,-1).sum(-1)[:,None,None]
    only1=(only1==1).float()
    # c=torch.diag_embed(torch.ones_like(v),dim1=-2,dim2=-1)
    # return torch.matmul(v[...,:,None],v[...,None,:])+c-2*b
    
    return torch.matmul(v[...,:,None],v[...,None,:])-b+b*only1

def generate_actors_from_action(v:torch.Tensor):
    # v: B,T,MAX_N
    v=v.float()
    b=torch.diag_embed(v,dim1=-2,dim2=-1) #B,T,N,N
    T,N,_=b.shape
    only1=b.reshape(T,-1).sum(-1)[:,None,None]
    only1=(only1==1).float()
    # c=torch.diag_embed(torch.ones_like(v),dim1=-2,dim2=-1)
    # return torch.matmul(v[...,:,None],v[...,None,:])+c-2*b
    
    return torch.matmul(v[...,:,None],v[...,None,:])-b+b*only1