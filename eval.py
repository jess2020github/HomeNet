import os
import argparse
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import PanoDataset
from utils import StatisticDict
from pano import get_ini_cor
from pano_opt import optimize_cor_id
from utils_eval import eval_PE, eval_3diou, augment, augment_undo

from pspnet import PSPNet

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# Model related arguments
parser.add_argument('--ckpt', default='ckpt/pre',
                    help='Checkpoint path to load model.')
parser.add_argument('--device', default='cuda:0',
                    help='device to run models.')
# Dataset related arguments
parser.add_argument('--root_dir', default='data/test',
                    help='root directory to construct dataloader.')
parser.add_argument('--input_cat', default=['img', 'line'], nargs='+',
                    help='input channels subdirectories')
parser.add_argument('--input_channels', default=6, type=int,
                    help='numbers of input channels')
parser.add_argument('--d1', default=21, type=int,
                    help='Post-processing parameter.')
parser.add_argument('--d2', default=3, type=int,
                    help='Post-processing parameter.')
# Augmentation related
parser.add_argument('--flip', action='store_true',
                    help='whether to perfome left-right flip. '
                         '# of input x2.')
parser.add_argument('--rotate', nargs='*', default=[], type=float,
                    help='whether to perfome horizontal rotate. '
                         'each elements indicate fraction of image width. '
                         '# of input xlen(rotate).')
parser.add_argument('--post_optimization', action='store_true',
                    help='whether to performe post gd optimization')
parser.add_argument('--backend', default='resnet34', type=str,
                    help='Feature Extractor')
args = parser.parse_args()
device = torch.device(args.device)

models = {
    'squeezenet': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='squeezenet'),
    'densenet': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=1024, deep_features_size=512, backend='densenet'),
    'resnet18': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet18'),
    'resnet34': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet34'),
    'resnet50': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet50'),
    'resnet101': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet101'),
    'resnet152': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet152')
}

# Create dataloader
dataset = PanoDataset(root_dir=args.root_dir,
                      cat_list=[*args.input_cat, 'edge', 'cor'],
                      flip=False, rotate=False,
                      gamma=False,
                      return_filenames=True)


# Prepare model
backend = args.backend.lower()
net = models[backend]()
net = nn.DataParallel(net).to(device)
net.load_state_dict(torch.load(args.ckpt))


# Start evaluation
test_losses = StatisticDict()
test_pano_losses = StatisticDict()
test_2d3d_losses = StatisticDict()
for ith, datas in enumerate(dataset):
    print('processed %d batches out of %d' % (ith, len(dataset)), end='\r', flush=True)

    x = torch.cat([datas[i] for i in range(len(args.input_cat))], dim=0).numpy()
    x_augmented, aug_type = augment(x, args.flip, args.rotate)

    with torch.no_grad():
        x_augmented = torch.FloatTensor(x_augmented).to(device)
        de_list = net(x_augmented)
        edg_tensor = torch.sigmoid(de_list[:,:-1])
        cor_tensor = torch.sigmoid(de_list[:,-1:])

        # Recover the effect from augmentation
        edg_img = augment_undo(edg_tensor.cpu().numpy(), aug_type)
        cor_img = augment_undo(cor_tensor.cpu().numpy(), aug_type)

        # Merge all results from augmentation
        edg_img = edg_img.transpose([0, 2, 3, 1]).mean(0)
        cor_img = cor_img.transpose([0, 2, 3, 1]).mean(0)[..., 0]

    # Load ground truth corner label
    k = datas[-1][:-4]
    path = os.path.join(args.root_dir, 'label_cor', '%s.txt' % k)
    with open(path) as f:
        gt = np.array([line.strip().split() for line in f], np.float64)

    # Construct corner label from predicted corner map
    cor_id = get_ini_cor(cor_img, args.d1, args.d2)

    # Gradient descent optimization
    if args.post_optimization:
        cor_id = optimize_cor_id(cor_id, edg_img, cor_img,
                                 num_iters=100, verbose=False)

    # Compute normalized corner error
    cor_error = ((gt - cor_id) ** 2).sum(1) ** 0.5
    cor_error /= np.sqrt(cor_img.shape[0] ** 2 + cor_img.shape[1] ** 2)
    pe_error = eval_PE(cor_id[0::2], cor_id[1::2], gt[0::2], gt[1::2])
    iou3d = eval_3diou(cor_id[1::2], cor_id[0::2], gt[1::2], gt[0::2])
    test_losses.update('CE(%)', cor_error.mean() * 100)
    test_losses.update('PE(%)', pe_error * 100)
    test_losses.update('3DIoU', iou3d)

    if k.startswith('pano'):
        test_pano_losses.update('CE(%)', cor_error.mean() * 100)
        test_pano_losses.update('PE(%)', pe_error * 100)
        test_pano_losses.update('3DIoU', iou3d)
    else:
        test_2d3d_losses.update('CE(%)', cor_error.mean() * 100)
        test_2d3d_losses.update('PE(%)', pe_error * 100)
        test_2d3d_losses.update('3DIoU', iou3d)

print('[RESULT overall     ] %s' % (test_losses), flush=True)
print('[RESULT panocontext ] %s' % (test_pano_losses), flush=True)
print('[RESULT stanford2d3d] %s' % (test_2d3d_losses), flush=True)
