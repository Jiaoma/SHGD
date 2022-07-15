from os.path import join
from typing import OrderedDict
import warnings

warnings.filterwarnings('ignore')

import torch.optim as optim
from torch.nn.functional import cosine_similarity
import torch.nn as nn
import torch.nn.functional as F

import random

from torch.utils import data

from config import *
from dataset import *
from base_model import RBSModel2
from utils import *
from evaluate import evaluatePANDA
from reference.cdp.source import graph

note_net_map = {
    'PANDA_stage1_V3Net': RBSModel2,
    'PANDA_stage2_V3Net': RBSModel2,
}


def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()


def set_drop_eval(m):
    classname = m.__class__.__name__
    if classname.find('Drop') != -1:
        m.eval()


@torch.no_grad()
def get_wd_params(model: nn.Module):
    # Parameters must have a defined order. 
    # No sets or dictionary iterations.
    # See https://pytorch.org/docs/stable/optim.html#base-class
    # Parameters for weight decay.
    decay = list()
    no_decay = list()
    for name, param in model.named_parameters():
        print('checking {}'.format(name))
        if hasattr(param, 'requires_grad') and not param.requires_grad:
            continue
        if 'weight' in name and 'norm' not in name and 'bn' not in name:
            decay.append(param)
        else:
            no_decay.append(param)
    return decay, no_decay


def try_to(ts, device):
    if ts is not None:
        return ts.to(device)
    else:
        return None


def adjust_lr(optimizer, new_lr):
    print('change learning rate:', new_lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


import copy


class detector:
    def __init__(self, initdict):
        self.initdict = copy.deepcopy(initdict)
        for k in self.initdict.keys():
            self.initdict[k] = self.initdict[k].detach().cpu()

    def new_dict(self, ndict: OrderedDict):
        res = {}
        for k, v in ndict.items():
            if k in self.initdict:
                res[k] = torch.abs(self.initdict[k] - v.detach().cpu()).sum()
                print('Key:{}, var:{}'.format(k, res[k].item()))


def train_net(cfg, training_set=None, validation_set=None):
    """
    training gcn net
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.device_list

    # Show config parameters
    cfg.init_config()
    show_config(cfg)
    # vis=Visdom_E()
    # vis.set('tsne',env='hit')

    # Reading dataset
    if (training_set is None) or (validation_set is None):
        training_set, validation_set = return_dataset(cfg)
    g = torch.Generator()
    g.manual_seed(cfg.train_random_seed)
    params = {
        'batch_size': cfg.batch_size,
        'shuffle': True,
        'num_workers': cfg.num_workers,
        'pin_memory': False,
        'drop_last': True,
        'worker_init_fn': seed_worker,
        'generator': g,
    }

    params_test = {
        'batch_size': cfg.test_batch_size,
        'shuffle': False,
        'num_workers': cfg.num_workers,
        'pin_memory': False,
        'worker_init_fn': seed_worker,
        'generator': g,
    }

    training_loader = data.DataLoader(training_set, **params)

    params['batch_size'] = cfg.test_batch_size
    validation_loader = data.DataLoader(validation_set, **params_test)

    # Set random seed
    np.random.seed(cfg.train_random_seed)
    torch.manual_seed(cfg.train_random_seed)
    torch.cuda.manual_seed_all(cfg.train_random_seed)
    random.seed(cfg.train_random_seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)

    # Set data position
    if cfg.use_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    Net = note_net_map[cfg.exp_note.split('+')[0]]
    # Build model and optimizer
    basenet_list = {'PANDA': Net, }
    gcnnet_list = {'PANDA': Net, }
    params_net = {
        'cfg': cfg,
        'n_stgcnn': 3,
        'n_txpcnn': 3,
        'input_feat1': cfg.action_features,
        'input_feat2': cfg.num_posf,
        'output_feat': 32,
        'seq_len': cfg.T,
        'pred_seq_len': cfg.T,
        'kernel_size': 3,
        'mode': cfg.training_stage
    }
    if cfg.training_stage == 1:
        Basenet = basenet_list[cfg.dataset_name]
        model = Basenet(**params_net)

    elif cfg.training_stage == 2:
        model = gcnnet_list[cfg.dataset_name](**params_net)
        # Load backbone
        if ('stage' in cfg.exp_note) and cfg.relation_model_path != '':
            model.loadmodel(cfg.relation_model_path, pretrain=True)
        if cfg.stage2_model_path != '':  # this is for test
            model.loadmodel(cfg.stage2_model_path)
        # if cfg.stage2_model_path!='': # this is for test
        # model.loadmodel(cfg.stage2_model_path)
    else:
        assert (False)

    if cfg.iscontinue:
        assert cfg.continue_path != ''
        model.loadmodel(cfg.continue_path)

    if cfg.use_multi_gpu:
        model = nn.DataParallel(model)

    model = model.to(device=device)

    model.train()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters(
    )), lr=cfg.train_learning_rate, weight_decay=cfg.weight_decay)
    start_epoch = 1
    if cfg.iscontinue:
        state = torch.load(cfg.continue_path)
        start_epoch = state['epoch']

    if cfg.training_stage == 1:
        train_list = {'PANDA': train_stage1, }
    else:
        train_list = {'PANDA': train_panda_stage2, }
    test_list = {'PANDA': test_panda, }
    train = train_list[cfg.dataset_name]
    test = test_list[cfg.dataset_name]

    if cfg.test_before_train and cfg.training_stage == 2:
        test_info = test(validation_loader, model, device, 0, cfg)  # ,d)
        print(test_info)
        torch.cuda.empty_cache()

    # Training iteration
    best_result = {'epoch': 0, 'activities_acc': 0}

    for epoch in range(start_epoch, start_epoch + cfg.max_epoch):

        if epoch in cfg.lr_plan:
            adjust_lr(optimizer, cfg.lr_plan[epoch])
        train_info = train(training_loader, model,
                           device, optimizer, epoch, cfg)
        show_epoch_info('Train', cfg.log_path, train_info)

        # Test
        if epoch % cfg.test_interval_epoch == 0:
            if cfg.training_stage == 2:
                test_info = test(validation_loader, model, device, epoch, cfg)  # ,d)
                torch.cuda.empty_cache()
                show_epoch_info('Test', cfg.log_path, test_info)

                if test_info['activities_acc'] >= best_result['activities_acc']:
                    best_result = test_info
                print_log(cfg.log_path,
                          'Best group activity accuracy: %.2f%% at epoch #%d.' % (
                              best_result['activities_acc'], best_result['epoch']))

                # Save model STAGE1_MODEL
            if cfg.training_stage == 2:
                state = {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
                filepath = cfg.result_path + \
                           '/stage%d_epoch%d_%.2f%%.pth' % (
                               cfg.training_stage, epoch,
                               test_info['activities_acc'])
                torch.save(state, filepath)
                print('model saved to:', filepath)
            elif cfg.training_stage == 1:
                state = {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
                filepath = cfg.result_path + \
                           '/stage%d_epoch%d.pth' % (
                               cfg.training_stage, epoch)
                torch.save(state, filepath)
                print('model saved to:', filepath)
            else:
                assert False


def test_net(cfg, training_set=None, validation_set=None):
    """
    training gcn net
    """
    if cfg.training_stage == 1:
        return
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.device_list

    # Show config parametersx
    cfg.init_config()
    show_config(cfg)

    # Reading dataset
    if (training_set is None) or (validation_set is None):
        training_set, validation_set = return_dataset(cfg)

    g = torch.Generator()
    g.manual_seed(cfg.train_random_seed)

    params = {
        'batch_size': cfg.test_batch_size,
        'shuffle': False,
        'num_workers': cfg.num_workers,
        'pin_memory': False,
        'worker_init_fn': seed_worker,
        'generator': g,
        'drop_last': False
    }

    params['batch_size'] = cfg.test_batch_size
    validation_loader = data.DataLoader(validation_set, **params)

    # Set random seed
    np.random.seed(cfg.train_random_seed)
    torch.manual_seed(cfg.train_random_seed)
    torch.cuda.manual_seed_all(cfg.train_random_seed)
    random.seed(cfg.train_random_seed)
    # torch.backends.cudnn.deterministic = True

    # Set data position
    if cfg.use_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    Net = note_net_map[cfg.exp_note.split('+')[0]]

    # Build model and optimizer
    basenet_list = {'PANDA': Net, }
    gcnnet_list = {'PANDA': Net, }
    params_net = {
        'cfg': cfg,
        'n_stgcnn': 3,
        'n_txpcnn': 3,
        'input_feat1': cfg.action_features,
        'input_feat2': cfg.num_posf,
        'output_feat': 32,
        'seq_len': cfg.T,
        'pred_seq_len': cfg.T,
        'kernel_size': 3,
        'mode': cfg.training_stage
    }
    if cfg.training_stage == 1:
        Basenet = basenet_list[cfg.dataset_name]
        model = Basenet(**params_net)

    elif cfg.training_stage == 2:
        model = gcnnet_list[cfg.dataset_name](**params_net)
        # Load backbone
        if ('stage' in cfg.exp_note) and cfg.relation_model_path != '':
            model.loadmodel(cfg.relation_model_path, pretrain=True)
        if cfg.stage2_model_path != '':  # this is for test
            model.loadmodel(cfg.stage2_model_path)
    else:
        assert (False)

    model = model.to(device=device)

    model.train()

    test_list = {'PANDA': test_panda, }
    test = test_list[cfg.dataset_name]

    test_info = test(validation_loader, model, device, 0, cfg)
    print_log(cfg.log_path, test_info)


def bce(source, target, positive_weight=10, attentions=1):
    return (-1 * attentions * (target * torch.log(source + 1e-6) * positive_weight + (1 - target) * torch.log(
        1 - source + 1e-6)))  # .mean()


def test_panda(data_loader, model, device, epoch, cfg):
    model.train()
    model.apply(set_drop_eval)

    epoch_timer = Timer()

    if not hasattr(cfg, 'save_result_path'):
        # Class correct num
        group_results_seq_collection = {'prediction': {}, 'gt': {}, 'divide': {}}  # vid:{(pid,pid):[r1,r2,...]}
        group_final_collection = {}

        with torch.no_grad():
            for batch_data in data_loader:
                # prepare batch data
                feature_mat, A, interaction_gt_mat, avoid_gt_mat, group_gt_mat, index_mat, _ = batch_data
                # B,T,N,C  B,T,N,N  B,T,N         B,T,N    B,N,N           B,T,N,3
                feature_mat = try_to(feature_mat, device=device)
                A = try_to(A, device=device)
                batch_size = batch_data[0].shape[0]

                binary_A_mask = A > 0.02
                interaction_scores, avoid_scores, group_scores = model(feature_mat.permute(0, 3, 1, 2), A)
                # Note, the first two scores are useless in this work, but these two tasks could be explored in the future.
                nn_mask = (binary_A_mask[0].sum(dim=0) > 1)  # N,N # Remove people all alone
                group_scores = group_scores.detach().cpu()
                binary_A_mask, A, nn_mask = binary_A_mask.detach().cpu(), A.detach().cpu(), nn_mask.detach().cpu()
                vfpd = index_mat[0, 0, 0]  # Only consider batch_size==1, it's a tensor([vid,fid,pid])
                for b in range(batch_size):
                    for nnpair in nn_mask.nonzero():
                        vid = vfpd[0].int().item()
                        index_a, index_b = nnpair.int().tolist()
                        if index_a == index_b:
                            continue
                        target_dict = group_results_seq_collection['prediction']
                        divide_dict = group_results_seq_collection['divide']
                        gt_dict = group_results_seq_collection['gt']
                        if vid not in target_dict:
                            target_dict[vid] = {}
                            gt_dict[vid] = {}
                            divide_dict[vid] = {}
                        pid_a, pid_b = index_mat[b, 0, index_a, 2].int().item(), index_mat[
                            b, 0, index_b, 2].int().item()
                        # how close
                        pp_group_score = group_scores[b, index_a, index_b, 0]
                        pp_group_gt = group_gt_mat[b, index_a, index_b]
                        if (pid_a, pid_b) not in target_dict[vid]:
                            target_dict[vid][(pid_a, pid_b)] = []
                            gt_dict[vid][(pid_a, pid_b)] = []
                            divide_dict[vid][(pid_a, pid_b)] = []
                        target_dict[vid][(pid_a, pid_b)].append(pp_group_score)
                        gt_dict[vid][(pid_a, pid_b)].append(pp_group_gt)  # this gt is useless.
                        divide_dict[vid][(pid_a, pid_b)].append(pp_group_score.item())

        print('data collection okay')
        # Post-process: Average
        for _v in group_results_seq_collection['prediction'].keys():
            v_group_results_seq_collection = group_results_seq_collection['prediction'][_v]
            for p in v_group_results_seq_collection.keys():
                ave = sum(v_group_results_seq_collection[p]) / len((v_group_results_seq_collection[p]))
                v_group_results_seq_collection[p] = ave  # torch.argmax(ave).item()
        for _v in group_results_seq_collection['gt'].keys():
            v_group_results_seq_collection_gt = group_results_seq_collection['gt'][_v]
            for p in v_group_results_seq_collection_gt.keys():
                ave = sum(v_group_results_seq_collection_gt[p]) / len((v_group_results_seq_collection_gt[p]))
                v_group_results_seq_collection_gt[p] = ave.item()
        for _v in group_results_seq_collection['divide'].keys():
            v_group_results_seq_collection_div = group_results_seq_collection['divide'][_v]
            for p in v_group_results_seq_collection_div.keys():
                ave = sum(v_group_results_seq_collection_div[p]) / len(v_group_results_seq_collection_div[p])
                v_group_results_seq_collection_div[p] = ave

        for vid in group_results_seq_collection['divide'].keys():
            pairs = np.asarray(list(group_results_seq_collection['divide'][vid].keys()))  # N,2
            scores = np.asarray([group_results_seq_collection['divide'][vid][k] for k in
                                 group_results_seq_collection['divide'][
                                     vid].keys()])  # Only in python >3.5/6(maybe) the keies has order.
            # vid: [group1 group2 group3 ...]
            components = graph.graph_propagation(pairs, scores, 5, 0.1, 100)
            cluster = [[n.name for n in c] for c in components]  # [[],[]]
            # groupi: member, relation type.
            if vid not in group_final_collection:
                group_final_collection[vid] = {'members': [], 'relation': [], 'gtmembers': [], 'gtrelation': []}
            clstypes = []
            newcluster = []
            for clst in cluster:
                clst.sort()
                ave_type_vector = []
                for mema in clst:
                    for memb in clst:
                        if (mema, memb) in group_results_seq_collection['prediction'][vid]:
                            ave_type_vector.append(group_results_seq_collection['prediction'][vid][(mema, memb)])
                if len(ave_type_vector) == 0:
                    continue
                newcluster.append(clst)
                ave_type_vector = sum(ave_type_vector) / len(ave_type_vector)
                clst_type = (ave_type_vector >= 0.5).int().item()  # torch.argmax(torch.tensor(ave_type_vector)).item()
                clstypes.append(clst_type)
            group_final_collection[vid]['relation'] = clstypes
            group_final_collection[vid]['members'] = newcluster
            group_final_collection[vid]['gtmembers'] = [x['members'] for x in data_loader.dataset.vid_group_dict[vid]]
            group_final_collection[vid]['gtrelation'] = [x['relation'] for x in data_loader.dataset.vid_group_dict[vid]]

        final_result = {'group': group_final_collection}
        torch.save(final_result, join(cfg.result_path, 'result_epoch%d.pt' % epoch))
        print('Saved to %s' % (join(cfg.result_path, 'result_epoch%d.pt' % epoch)))
    else:
        final_result = torch.load(cfg.save_result_path)
    activities_acc = evaluatePANDA(cfg, final_result, fast=False)  # Instead, its MHIA
    test_info = {
        'time': epoch_timer.timeit(),
        'epoch': epoch,
        'loss': -1,
        'activities_acc': activities_acc * 100,
    }
    return test_info


def train_stage1(data_loader, model, device, optimizer, epoch, cfg):
    loss_meter = AverageMeter()
    epoch_timer = Timer()
    weighted = torch.ones(34).to(device)
    weighted[32] = 100
    weighted[33] = 100
    weighted = weighted[None, None, None, :].float()
    loss = nn.BCELoss(reduction='sum')
    for batch_data in data_loader:
        model.train()
        optimizer.zero_grad()
        # prepare batch data
        batch_data = [try_to(b, device) for b in batch_data]
        batch_size, num_frames, bboxes_num, channels = batch_data[0].shape
        recover_features, a = model(batch_data[0][:, :, :, :-1].permute(0, 3, 1, 2), batch_data[-2][
            0])  # Here, we only could use batch_size as 1, since different batch has different A matrix.
        recover_loss1 = (
            F.mse_loss(recover_features, batch_data[1][:, :, :, cfg.action_features:cfg.action_features + cfg.num_posf],
                       reduction='sum'))
        # Total loss
        total_loss = recover_loss1
        loss_meter.update(total_loss.item(), batch_size)

        total_loss.backward()
        optimizer.step()
    # Save the model parameters. We need all the parameters saved.

    train_info = {
        'time': epoch_timer.timeit(),
        'epoch': epoch,
        'loss': loss_meter.avg,
        'activities_acc': 0,
        'actions_acc': 0,
        'actors_acc': 0
    }
    return train_info


def bce(source, target, positive_weight=10, attentions=1):
    return (-1 * attentions * (target * torch.log(source + 1e-6) * positive_weight + (1 - target) * torch.log(
        1 - source + 1e-6))).mean()


def cosine(source, target):
    return (1 - cosine_similarity(source, target)).mean()


def train_panda_stage2(data_loader, student_model, device, optimizer, epoch, cfg):
    loss_meter = AverageMeter()
    epoch_timer = Timer()
    for batch_data in data_loader:
        student_model.train()
        optimizer.zero_grad()
        # prepare batch data
        batch_data = [try_to(b, device) for b in batch_data]
        batch_size, num_frames, bboxes_num, channels = batch_data[0].shape
        binary_A_mask = batch_data[1]  # B,T,N,N
        interaction_scores, avoid_scores, group_scores = student_model(batch_data[0].permute(0, 3, 1, 2),
                                                                       binary_A_mask)  # B,T,N,N
        group_scores = torch.sigmoid(group_scores)
        group_loss = cosine(group_scores.reshape(batch_size, -1), batch_data[4].float().reshape(batch_size, -1)) + bce(
            group_scores.reshape(batch_size, -1), batch_data[4].float().reshape(batch_size, -1)) * 0.001
        total_loss = group_loss
        loss_meter.update(total_loss.item(), batch_size)

        total_loss.backward()
        optimizer.step()

    train_info = {
        'time': epoch_timer.timeit(),
        'epoch': epoch,
        'loss': loss_meter.avg,
        'activities_acc': 0,
        'actions_acc': 0,
        'actors_acc': 0
    }
    return train_info
