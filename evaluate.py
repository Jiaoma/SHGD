from os.path import join
from os import mkdir, listdir
import cv2
import torch

from os.path import exists

import json

from config import *
from utils import print_log


from feature_extract.panda import GROUP_TYPES

from scipy.optimize import linear_sum_assignment as linear_assignment
from reference.cdp.source import graph


def saveDict(src: dict, save_path: str):
    # Notice that, here the dict is not OrderDict
    with open(save_path, 'w') as f:
        json.dump(src, f)


def safeMkdir(path: str):
    if exists(path):
        return
    else:
        mkdir(path)


def drawBoundingBox(imgPath: str, bboxes: dict, savePath: str):
    img = cv2.imread(imgPath)
    img = cv2.UMat(img).get()
    font = cv2.FONT_HERSHEY_SIMPLEX
    for key in bboxes.keys():
        bbox = bboxes[key]
        bbox = [int(i) for i in bbox]
        text = str(key)
        def f(x): return (int(x[0]), int(x[1]))
        cv2.rectangle(img, f(bbox[:2]), f(bbox[2:]), (0, 255, 0), 4)
        cv2.putText(img, text, f(bbox[:2]), font, 2, (0, 0, 255), 1)
    cv2.imwrite(savePath, img)


def plot_confusion_matrix(cm,
                          target_names,
                          save_path='./result.svg',
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True,
                          ):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    save_path:    path to save image.

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    plot_confusion_matrix(cm           = np.array([[ 1098,  1934,   807],
                                              [  604,  4392,  6233],
                                              [  162,  2362, 31760]]),
                      normalize    = False,
                      target_names = ['high', 'medium', 'low'],
                      title        = "Confusion Matrix")

    """
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools
    matplotlib.use('Agg')
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(
        accuracy, misclass))
    plt.savefig(save_path)
    plt.close()

def group_eval(GROUP, GT, crit='half'):
        if not GROUP:
            return 0,0,0,0,0,0
        if not GT:
            return 0,0,0,0,0,0

        TP = 0

        for gt in GT:
            gt_set = set(gt)
            gt_card = len(gt)
            for group in GROUP:
                group_set = set(group)
                group_card = len(group)
                if crit == 'half':
                    inters = list(gt_set & group_set)
                    inters_card = len(inters)
                    if group_card == 2 and gt_card == 2:
                        if not len(gt_set - group_set):
                            TP += 1
                    elif inters_card / max(gt_card, group_card) > 1/2:
                        TP += 1
                elif crit == 'card':
                    inters = list(gt_set & group_set)
                    inters_card = len(inters)
                    if group_card == 2 and gt_card == 2:
                        if not len(gt_set - group_set):
                            TP += 1
                    elif inters_card / max(gt_card, group_card) > 2/3:
                        TP += 1
                elif crit == 'dpmm':
                    inters = list(gt_set & group_set)
                    inters_card = len(inters)
                    if group_card == 2 and gt_card == 2:
                        if not len(gt_set - group_set):
                            TP += 1
                    elif inters_card / max(gt_card, group_card) > 0.6:
                        TP += 1
                elif crit == 'all':
                    if not len(gt_set - group_set):
                        TP += 1
        FP = len(GROUP) - TP
        FN = len(GT) - TP

        if TP + FP == 0:
            precision = 0
        else:
            precision = TP / (TP + FP)

        if TP + FN == 0:
            recall = 0
        else:
            recall = TP / (TP + FN)

        if (precision+recall)==0:
            f1=0
        else:
            f1 = (2 * precision * recall) / (precision + recall)

        return precision, recall, f1, TP, FP, FN

def thred_group_eval(GROUP, GT, thred):
    # check group and gt
    if not GROUP:
        return 0,0,0,0,0,0
    if not GT:
        return 0,0,0,0,0,0

    TP = 0

    for gt in GT:
        gt_set = set(gt)
        gt_card = len(gt)
        for group in GROUP:
            group_set = set(group)
            group_card = len(group)
            inters = list(gt_set & group_set)
            inters_card = len(inters)
            if inters_card / max(gt_card, group_card) > thred:
                TP += 1
                continue

    FP = len(GROUP) - TP
    FN = len(GT) - TP

    if TP + FP == 0:
        precision = 1
    else:
        precision = TP / (TP + FP)

    if TP + FN == 0:
        recall = 1
    else:
        recall = TP / (TP + FN)

    if (precision+recall)==0:
        f1=0
    else:
        f1 = (2 * precision * recall) / (precision + recall)

    return precision, recall, f1, TP, FP, FN

import matplotlib.pyplot as plt
def AUC_IoU(GROUPs,GTs,KEYs,thred=(0.5,0.6,0.7,0.8,0.9,1.0)):
    thred=[i/100 for i in range(50,101,1)]
    # prec=[]
    # rec=[]
    pr=[]
    for t in thred:
        TP=0
        FP=0
        FN=0
        for k in KEYs:
            GROUP=GROUPs[k]['members']
            GT=GTs[k]['gtmembers']
            # GROUP=GROUPs[k]
            # GT=GTs[k]
            precision, recall, f1, tp, fp, fn=thred_group_eval(GROUP,GT,t)
            if precision>1 or recall>1:
                print('debug')
            TP+=tp
            FP+=fp
            FN+=fn
        if TP + FP == 0:
            precision = 1
        else:
            precision = TP / (TP + FP)

        if TP + FN == 0:
            recall = 1
        else:
            recall = TP / (TP + FN)

        if (precision+recall)==0:
            f1=0
        else:
            f1 = (2 * precision * recall) / (precision + recall)
        pr.append((f1,t))
    pr.sort(key=lambda x:x[1])    
    num=len(thred)
    auc=0
    for i in range(num):
        if i==0:
            auc+=pr[i][0]*(pr[i][1]-0.5)
        else:
            auc+=pr[i][0]*(pr[i][1]-pr[i-1][1])
    return auc/(thred[-1]-thred[0])

def Ngroup_eval(N, GROUP, GTo, crit='half'):
    # check group and gt
    if not GROUP:
        # print('group is empty!')
        return 0,0,0,0,0,0
    
    GT=[]
    for gt in GTo:
        if (N<6 and len(gt)==N) or (N>=6 and len(gt)>=6):
            GT.append(gt)
            # if N==4:
            #     print(4)
    if not GT:
        # print('gt is empty!')
        return 0,0,0,0,0,0

    TP = 0

    for gt in GT:
        gt_set = set(gt)
        gt_card = len(gt)
        for group in GROUP:
            group_set = set(group)
            group_card = len(group)
            if crit == 'half':
                inters = list(gt_set & group_set)
                inters_card = len(inters)
                if group_card == 2 and gt_card == 2:
                    if not len(gt_set - group_set):
                        TP += 1
                elif inters_card / max(gt_card, group_card) > 1/2:
                    TP += 1
            elif crit == 'card':
                inters = list(gt_set & group_set)
                inters_card = len(inters)
                if group_card == 2 and gt_card == 2:
                    if not len(gt_set - group_set):
                        TP += 1
                elif inters_card / max(gt_card, group_card) > 2/3:
                    TP += 1
            elif crit == 'dpmm':
                inters = list(gt_set & group_set)
                inters_card = len(inters)
                if group_card == 2 and gt_card == 2:
                    if not len(gt_set - group_set):
                        TP += 1
                elif inters_card / max(gt_card, group_card) > 0.6:
                    TP += 1
            elif crit == 'all':
                if not len(gt_set - group_set):
                    TP += 1

    FP = len(GROUP) - TP
    FN = len(GT) - TP

    if TP + FP == 0:
        precision = 0
    else:
        precision = TP / (TP + FP)

    if TP + FN == 0:
        recall = 0
    else:
        recall = TP / (TP + FN)

    if (precision+recall)==0:
        f1=0
    else:
        f1 = (2 * precision * recall) / (precision + recall)

    return precision, recall, f1, TP, FP, FN # However, only recall makes sense.

def mat_group_eval(GROUP, GT):
    # check group and gt
    if not GROUP:
        # print('group is empty!')
        return 0,0
    if not GT:
        # print('gt is empty!')
        return 0,0
    def get_max(data):
        m=-1
        for i in data:
            for j in i:
                if j>m:
                    m=j
        return m
    maxn=max(get_max(GROUP),get_max(GT))+1
    GTR=np.zeros((maxn,maxn))
    GPR=np.zeros((maxn,maxn))
    for gt in GT:
        for a in gt:
            for b in gt:
                if b==a:
                    continue
                GTR[a,b]=1
                GTR[b,a]=1
    for gr in GROUP:
        for a in gr:
            for b in gr:
                if b==a:
                    continue
                GPR[a,b]=1
                GPR[b,a]=1
    TP=(GTR*GPR).sum()
    AL=GTR.sum()+GPR.sum()-TP
    # print(TP)
    # print(GTR.sum(),GPR.sum())
    # print(AL)
    return TP, AL


def evaluatePANDA(cfg, result, fast=False,binary=True):
    group_preds = []
    group_gts = []

    def iou(a: list, b: list):
        sameNum = 0
        if len(b) == 0:
            return 0
        for i in a:
            if i in b:
                sameNum += 1
        return sameNum/len(b)
    tp=0
    fp=0
    fn=0
    count=0
    vids_f1={}
    safed=lambda x,y:0 if y==0 else x/y
    for vid in result['group'].keys():  # vid is video id.
        _,_,_,TP,FP,FN=group_eval(result['group'][vid]['members'],result['group'][vid]['gtmembers'])
        count+=len(result['group'][vid]['gtmembers'])
        def local_f1():
            prec=safed(TP,(TP+FP))        
            recall=safed(TP,(TP+FN))
            return safed(2*prec*recall,(prec+recall))
        vids_f1[vid]=local_f1()
        tp+=TP
        fn+=FN
        fp+=FP
    # print('GT: numer', count)
    print("vidfid",vids_f1)
    nrec={}
    for n in range(2,7):
        averec=[]
        for vid in result['group'].keys():
            _,rec,f1,_,_,_=Ngroup_eval(n,result['group'][vid]['members'],result['group'][vid]['gtmembers'])
            averec.append(rec)
        nrec[n]=sum(averec)/len(averec) if len(averec)!=0 else 0
    
    prec=safed(tp,(tp+fp))        
    recall=safed(tp,(tp+fn))
    f1=safed(2*prec*recall,(prec+recall))

    report = {}
    for k,v in nrec.items():
        report[str(k)]=v
        
    report['AUC_IoU']=AUC_IoU(result['group'],result['group'],list(result['group'].keys()))

    mfz=0
    mfm=0
    
    for vid in result['group'].keys():  # vid is video id.
        TP,AL=mat_group_eval(result['group'][vid]['members'],result['group'][vid]['gtmembers'])
        mfz+=TP
        mfm+=AL
    report['mat_IoU']=mfz/mfm if mfm!=0 else 0
    

    report['group_f1'] = f1 #each_group_f1.mean().tolist()
    report['group_prec'] = prec #each_group_prec.mean().tolist()
    report['group_recall'] = recall #each_group_recall.mean().tolist()

    safeMkdir(cfg.result_path)
    print_log(cfg.log_path, report)
    saveDict(report, join(cfg.result_path, 'report.json'))
    # since the 0 represents background.
    return report['group_f1']


def acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

def evaluateJRDB_asPANDA(cfg, result, fast=False,binary=True):
    '''
    TODO:
    1. interaction/avoids: prec,f1,recall (each type and overall)
    2. group detection: first (preserve the IoU>0.5, then cal their prec, recall, f1. The IoU<=0.5 will be considered in the future.)
    3. Draw confusion matrixs.
    '''
    GROUP_NUM = len(GROUP_TYPES)
    # Confuse matrix
    # group_num_matrix = np.zeros((GROUP_NUM, GROUP_NUM))


    # delete bad group
    group_preds = []
    group_gts = []

    def iou(a: list, b: list):
        sameNum = 0
        if len(b) == 0:
            return 0
        for i in a:
            if i in b:
                sameNum += 1
        return sameNum/len(b)
    tp=0
    fp=0
    fn=0
    
    '''
    remove noisy
    '''
    vids_f1={}
    
    
    
    for vfid in result['group'].keys():  # vfid is video id.
        # pred_ck=[0,]*len(result['group'][vfid]['members'])
        # gt_ck=[0,]*len(result['group'][vfid]['gtmembers'])
        # for i, g in enumerate(result['group'][vfid]['members']):
        #     gts = result['group'][vfid]['gtmembers']
        #     for j, ggt in enumerate(gts):
        #         if iou(g, ggt)>=0.5:
        #             group_preds.append(result['group'][vfid]['relation'][i])
        #             group_gts.append(result['group'][vfid]['gtrelation'][j])
        #             pred_ck[i]=1
        #             gt_ck[j]=1
        #             continue # To ensure that one predicted group only belongs to one group in ground-truth.
        _,_,_,TP,FP,FN=group_eval(result['group'][vfid]['members'],result['group'][vfid]['gtmembers'])
        safed=lambda x,y:0 if y==0 else x/y
        prec=safed(TP,(TP+FP))        
        recall=safed(TP,(TP+FN))
        f1=safed(2*prec*recall,(prec+recall))
        vids_f1[vfid[0]]=f1
        tp+=TP
        fn+=FN
        fp+=FP
    nrec={}
    for n in range(2,7):
        averec=[]
        for vid in result['group'].keys():
            _,rec,f1,_,_,_=Ngroup_eval(n,result['group'][vid]['members'],result['group'][vid]['gtmembers'])
            averec.append(rec)
        nrec[n]=sum(averec)/len(averec) if len(averec)!=0 else 0
    safed=lambda x,y:0 if y==0 else x/y
    prec=safed(tp,(tp+fp))        
    recall=safed(tp,(tp+fn))
    f1=safed(2*prec*recall,(prec+recall))
    print(vids_f1)
    # Here, we exclude the negative class 0.
    # group_labels = [i for i in range(1, GROUP_NUM+1)]

    # group_gts = torch.Tensor(group_gts)
    # group_preds = torch.Tensor(group_preds)

    # each_group_prec, each_group_recall, each_group_f1, _ = precision_recall_fscore_support(
    #     group_gts.flatten().numpy(), group_preds.flatten().numpy(), average='binary')

    report = {}
    
    for k,v in nrec.items():
        report[str(k)]=v
        
    report['AUC_IoU']=AUC_IoU(result['group'],result['group'],list(result['group'].keys()))


    mfz=0
    mfm=0
    
    for vid in result['group'].keys():  # vid is video id.
        TP,AL=mat_group_eval(result['group'][vid]['members'],result['group'][vid]['gtmembers'])
        mfz+=TP
        mfm+=AL
    report['mat_IoU']=mfz/mfm if mfm!=0 else 0

    report['group_f1'] = f1 #each_group_f1.mean().tolist()
    report['group_prec'] = prec #each_group_prec.mean().tolist()
    report['group_recall'] = recall #each_group_recall.mean().tolist()
    # report['each_group_f1'] = each_group_f1.tolist()
    # report['each_group_prec'] = each_group_prec.tolist()
    # report['each_group_recall'] = each_group_recall.tolist()

    # report['group_acc'] = (group_gts == group_preds).float().mean().tolist()

    safeMkdir(cfg.result_path)
    print_log(cfg.log_path, report)
    saveDict(report, join(cfg.result_path, 'report.json'))
    # since the 0 represents background.
    group_labels = [1]

    # for group_a in group_labels:
    #     for group_b in group_labels:
    #         group_num_matrix[group_labels.index(group_a), group_labels.index(group_b)] = (
    #             (group_gts == group_a) & (group_preds == group_b)).sum()

    # if not fast:
    #     plot_confusion_matrix(group_num_matrix, save_path=join(cfg.result_path, 'group.svg'),
    #                           normalize=True,
    #                           target_names=['acquaintance', 'family', 'business'], title='Confusion Matrix')
    return report['group_f1']

def evaluate_from_file(path, dataset = "PANDA"):
    a=torch.load(path)
    class mycfg:
        result_path='./'
        log_path='./file.txt'
    if dataset=="PANDA":
        print(evaluatePANDA(mycfg(),a))
    elif dataset=="JRDB":
        print(evaluateJRDB_asPANDA(mycfg(),a))
    else:
        print("dataset not supported")        

if __name__ == '__main__':
    evaluate_from_file('/home/molijuly/fast/sambashare/SHGD_result/PANDA_features/model_ckpts/Refer2/result.pt', dataset="PANDA")
    evaluate_from_file('/home/molijuly/fast/sambashare/SHGD_result/JRDB_features/result_ws.pt', dataset="JRDB")