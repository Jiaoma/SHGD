import torch
import time
import pickle
from display import Visdom_E
import cv2
import numpy as np
import json
import copy


def readJson(p: str):
    with open(p, 'r') as f:
        config = json.load(f)
    return config


def writeJson(data, p: str):
    with open(p, 'w') as f:
        json.dump(data, f)


def get_kpts(maps, img_h=368.0, img_w=368.0):
    # maps (1,15,46,46)
    maps = maps.clone().cpu().data.numpy()
    map_6 = maps[0]

    kpts = []
    for m in map_6[0:]:
        h, w = np.unravel_index(m.argmax(), m.shape)
        x = int(w * img_w / m.shape[1])
        y = int(h * img_h / m.shape[0])
        kpts.append([x, y])
    return kpts


def drawSkeleton(kpts: np.ndarray, im: np.ndarray):
    '''
    kpts: Lx2 (X,Y)
    img: HxWx3
    '''

    #       RED           GREEN           RED          YELLOW          YELLOW          PINK          GREEN
    colors = [[000, 000, 255], [000, 255, 000], [000, 000, 255], [255, 255, 000], [255, 255, 000], [255, 000, 255],
              [000, 255, 000], \
              [255, 000, 000], [255, 255, 000], [255, 000, 255], [000, 255, 000], [000, 255, 000], [000, 000, 255],
              [255, 255, 000], [255, 000, 000]]
    #       BLUE          YELLOW          PINK          GREEN          GREEN           RED          YELLOW           BLUE

    limbSeq = [[8, 9], [7, 12], [12, 11], [11, 10], [7, 13], [13, 14], [7, 6], [6, 2], [2, 1], [1, 0], [6, 3], [3, 4],
               [4, 5], [7, 8]]
    kpts = kpts.astype(np.int)
    # im = cv2.resize(cv2.imread(img_path),(368,368))
    # draw points
    for k in kpts:
        x = k[0]
        y = k[1]
        cv2.circle(im, (x, y), radius=3, thickness=-1, color=(0, 0, 255))

    # draw lines
    for i in range(len(limbSeq)):
        cur_im = im.copy()
        limb = limbSeq[i]
        [Y0, X0] = kpts[limb[0]]
        [Y1, X1] = kpts[limb[1]]

        if X0 != 0 and Y0 != 0 and X1 != 0 and Y1 != 0:
            if i < len(limbSeq) - 4:
                cv2.line(cur_im, (Y0, X0), (Y1, X1), colors[i], 5)
            else:
                cv2.line(cur_im, (Y0, X0), (Y1, X1), [0, 0, 255], 5)

        im = cv2.addWeighted(im, 0.2, cur_im, 0.8, 0)
    return im


def drawRelation(bbox: list, relation_mat: np.ndarray, im: np.ndarray):
    '''
    img: HxWx3
    '''

    #       RED           GREEN           RED          YELLOW          YELLOW          PINK          GREEN
    colors = [[000, 000, 255], [000, 255, 000], [000, 000, 255], [255, 255, 000], [255, 255, 000], [255, 000, 255],
              [000, 255, 000], \
              [255, 000, 000], [255, 255, 000], [255, 000, 255], [000, 255, 000], [000, 255, 000], [000, 000, 255],
              [255, 255, 000], [255, 000, 000]]
    #       BLUE          YELLOW          PINK          GREEN          GREEN           RED          YELLOW           BLUE

    limbSeq = [[8, 9], [7, 12], [12, 11], [11, 10], [7, 13], [13, 14], [7, 6], [6, 2], [2, 1], [1, 0], [6, 3], [3, 4],
               [4, 5], [7, 8]]

    # im = cv2.resize(cv2.imread(img_path),(368,368))
    # draw points
    p_num = len(bbox)
    # draw lines
    for i in range(p_num):
        for j in range(p_num):
            if j == i:
                continue
            if relation_mat[i, j] < 0.05:
                continue
            [X0, Y0] = bbox[i]
            [X1, Y1] = bbox[j]

            if X0 != 0 and Y0 != 0 and X1 != 0 and Y1 != 0:
                cv2.line(im, (Y0, X0), (Y1, X1), [0, 0, int(255 * relation_mat[i, j])], 5)
    return im


def plotBoxes(images: torch.Tensor, boxes: torch.Tensor, OW, OH, color):
    images = ((images - torch.min(images)) / (torch.max(images) - torch.min(images)) * 255).int()
    C, H, W = images.shape
    MAX_N, _ = boxes.shape
    img = images.permute(1, 2, 0).detach().cpu().numpy().copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    for key in range(boxes.shape[0]):
        bbox = boxes[key]
        bbox = [int(i) for i in bbox]
        text = str(key)
        f = lambda x: (int(x[0] * OW), int(x[1] * OH))
        cv2.rectangle(img, f(bbox[:2]), f(bbox[2:]), color, 1)
        cv2.putText(img, text, f(bbox[:2]), font, 1, color, 1)
    img = torch.Tensor(img).permute(2, 0, 1)
    return img


# Replace operation
def axis2axis(xy, ori_axis, target_axis):
    '''
    xy: x,y
    ori_axis: xstart,xend,ystart,yend
    target_axis: xstart,xend,ystart,yend
    '''
    # input two axises, output the new axis in the other one.
    xy_ = [0, 0]
    xy_[0] = (xy[0] - ori_axis[0]) * (target_axis[1] - target_axis[0]) / (ori_axis[1] - ori_axis[0])
    xy_[1] = (xy[1] - ori_axis[2]) * (target_axis[3] - target_axis[2]) / (ori_axis[3] - ori_axis[2])
    return xy_


def xreverse(xy, axis):
    xy_ = copy.deepcopy(xy)
    xy_[0] = axis[1] - xy[0]
    return xy_


def yreverse(xy, axis):
    xy_ = copy.deepcopy(xy)
    xy_[1] = axis[3] - xy[1]
    return xy_


def readlines(file_path, key_index, preserve_index):
    res = {}
    with open(file_path) as f:
        for l in f:
            l = l.strip().split()
            key = int(l[key_index])
            if key not in res:
                res[key] = []
            res[key].append([float(l[i]) for i in preserve_index])
    return res


def drawPoint(img, xy):
    f = lambda x: (int(x[0]), int(x[1]))
    f1 = lambda x: (int(x[0]) + 10, int(x[1]) + 10)
    cv2.rectangle(img, f(xy), f1(xy), (255, 0, 0), 5)
    return img


def drawBox(img, x1y1x2y2):
    bbox = x1y1x2y2
    bbox = [int(i) for i in bbox]
    cv2.rectangle(img, bbox[:2], bbox[2:], 'red', 1)
    return img


class result_real_time_show:
    # online
    def __init__(self, cfg, name, vis: Visdom_E):
        self.name = name
        self.cfg = cfg
        self.vis = vis
        self.vis.set(self.name + '_img0')
        self.vis.set(self.name + '_img1')
        self.vis.set(self.name + '_img2')
        self.vis.set(self.name + '_text')
        self.vis.set(self.name + '_gt_actor')
        self.vis.set(self.name + '_p_actors')

    def input_image_gt(self, imgs, boxes_in, interaction, actors, actions, interaction_p, actors_p, actions_p):
        # img: T,C,H,W
        self.imgs = imgs
        self.T, self.C, self.H, self.W = imgs.shape
        # boxes_in: T,N,4
        self.boxes_in = boxes_in
        self.interaction = interaction
        self.actors = actors
        self.data = {'interaction': interaction, 'actors': actors, 'actions': actions, 'interaction_p': interaction_p,
                     'actors_p': actors_p, 'actions_p': actions_p}
        self.plotImage_text()

    def plotImage_text(self, color=(0, 255, 0)):
        self.vis.imageE(plotBoxes(self.imgs[0], self.boxes_in[0], self.cfg.image_size[1] / self.cfg.out_size[1],
                                  self.cfg.image_size[0] / self.cfg.out_size[0], color), self.name + '_img0')
        self.vis.imageE(plotBoxes(self.imgs[1], self.boxes_in[1], self.cfg.image_size[1] / self.cfg.out_size[1],
                                  self.cfg.image_size[0] / self.cfg.out_size[0], color), self.name + '_img1')
        self.vis.imageE(plotBoxes(self.imgs[2], self.boxes_in[2], self.cfg.image_size[1] / self.cfg.out_size[1],
                                  self.cfg.image_size[0] / self.cfg.out_size[0], color), self.name + '_img2')
        self.vis.textE(self.data, self.name + '_text')

    def plotActorMatrix(self, gt_actor_matrix, pred_actor_matrix):
        # NxN
        _, N, _ = gt_actor_matrix.shape
        self.vis.plot_confusion_matrixs(self.name + '_gt_actor', gt_actor_matrix.detach().cpu().numpy())
        self.vis.plot_confusion_matrixs(self.name + '_p_actors', pred_actor_matrix.detach().cpu().numpy())


def prep_images(images):
    """
    preprocess images
    Args:
        images: pytorch tensor
    """
    images = images.div(255.0)

    images = torch.sub(images, 0.5)
    images = torch.mul(images, 2.0)

    return images


def calc_pairwise_distance(X, Y):
    """
    computes pairwise distance between each element
    Args: 
        X: [N,D]
        Y: [M,D]
    Returns:
        dist: [N,M] matrix of euclidean distances
    """
    rx = X.pow(2).sum(dim=1).reshape((-1, 1))
    ry = Y.pow(2).sum(dim=1).reshape((-1, 1))
    dist = rx - 2.0 * X.matmul(Y.t()) + ry.t()
    return torch.sqrt(dist)


def calc_pairwise_distance_3d(X, Y):
    """
    computes pairwise distance between each element
    Args: 
        X: [B,N,D]
        Y: [B,M,D]
    Returns:
        dist: [B,N,M] matrix of euclidean distances
    """
    B = X.shape[0]

    rx = X.pow(2).sum(dim=2).reshape((B, -1, 1))
    ry = Y.pow(2).sum(dim=2).reshape((B, -1, 1))

    dist = rx - 2.0 * X.matmul(Y.transpose(1, 2)) + ry.transpose(1, 2)

    return torch.sqrt(dist)


def sincos_encoding_2d(positions, d_emb):
    """
    Args:
        positions: [N,2]
    Returns:
        positions high-dimensional representation: [N,d_emb]
    """

    N = positions.shape[0]

    d = d_emb // 2

    idxs = [np.power(1000, 2 * (idx // 2) / d) for idx in range(d)]
    idxs = torch.FloatTensor(idxs).to(device=positions.device)

    idxs = idxs.repeat(N, 2)  # N, d_emb

    pos = torch.cat([positions[:, 0].reshape(-1, 1).repeat(1, d), positions[:, 1].reshape(-1, 1).repeat(1, d)], dim=1)

    embeddings = pos / idxs

    embeddings[:, 0::2] = torch.sin(embeddings[:, 0::2])  # dim 2i
    embeddings[:, 1::2] = torch.cos(embeddings[:, 1::2])  # dim 2i+1

    return embeddings


def print_log(file_path, *args):
    print(*args)
    if file_path is not None:
        with open(file_path, 'a') as f:
            print(*args, file=f)


def show_config(cfg):
    print_log(cfg.log_path, '=====================Config=====================')
    for k, v in cfg.__dict__.items():
        print_log(cfg.log_path, k, ': ', v)
    print_log(cfg.log_path, '======================End=======================')


def show_epoch_info(phase, log_path, info):
    print_log(log_path, '')
    if phase == 'Test':
        print_log(log_path, '====> %s at epoch #%d' % (phase, info['epoch']))
        print_log(log_path, 'Group Activity Accuracy: %.2f%%, Loss: %.5f, Using %.1f seconds' % (
            info['activities_acc'], info['loss'], info['time']))
    else:
        print_log(log_path, '%s at epoch #%d' % (phase, info['epoch']))
        print_log(log_path,
                  'Group Activity Accuracy: %.2f%%, action: %.2f%%, actor: %.2f%%  Loss: %.5f, Using %.1f seconds' % (
                      info['activities_acc'], info['actions_acc'], info['actors_acc'], info['loss'], info['time']))


def log_final_exp_result(log_path, data_path, exp_result):
    no_display_cfg = ['num_workers', 'use_gpu', 'use_multi_gpu', 'device_list',
                      'batch_size_test', 'test_interval_epoch', 'train_random_seed',
                      'result_path', 'log_path', 'device']

    with open(log_path, 'a') as f:
        print('', file=f)
        print('', file=f)
        print('', file=f)
        print('=====================Config=====================', file=f)

        for k, v in exp_result['cfg'].__dict__.items():
            if k not in no_display_cfg:
                print(k, ': ', v, file=f)

        print('=====================Result======================', file=f)

        print('Best result:', file=f)
        print(exp_result['best_result'], file=f)

        print('Cost total %.4f hours.' % (exp_result['total_time']), file=f)

        print('======================End=======================', file=f)

    data_dict = pickle.load(open(data_path, 'rb'))
    data_dict[exp_result['cfg'].exp_name] = exp_result
    pickle.dump(data_dict, open(data_path, 'wb'))


class AverageMeter(object):
    """
    Computes the average value
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Timer(object):
    """
    class to do timekeeping
    """

    def __init__(self):
        self.last_time = time.time()

    def timeit(self):
        old_time = self.last_time
        self.last_time = time.time()
        return self.last_time - old_time
