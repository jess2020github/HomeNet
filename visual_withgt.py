import os
import glob
import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from utils_eval import augment, augment_undo
from pano import get_ini_cor, draw_boundary_from_cor_id, draw_gt_from_cor_id 
from pano_opt import optimize_cor_id

from pspnet import PSPNet

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# Model related arguments
parser.add_argument('--ckpt', default='ckpt/pre',
                    help='path to load model.')
parser.add_argument('--device', default='cuda:0',
                    help='device to run models.')
# I/O related arguments
parser.add_argument('--img_glob', required=True,
                    help='NOTE: Remeber to quote your glob path.')
parser.add_argument('--line_glob', required=True,
                    help='shold have the same number of files as img_glob. '
                         'two list with same index are load as input channels. '
                         'NOTE: Remeber to quote your glob path.')
parser.add_argument('--layout_gt', required=True,
                    help='NOTE: Remeber to quote your glob path.')
parser.add_argument('--output_dir', required=True)
# Data augmented arguments (to improve output quality)
parser.add_argument('--flip', action='store_true',
                    help='whether to perfome left-right flip. '
                         '# of input x2.')
parser.add_argument('--rotate', nargs='*', default=[], type=float,
                    help='whether to perfome horizontal rotate. '
                         'each elements indicate fraction of image width. '
                         '# of input xlen(rotate).')
# Post porcessing related arguments
parser.add_argument('--d1', default=21, type=int,
                    help='Post-processing parameter.')
parser.add_argument('--d2', default=3, type=int,
                    help='Post-processing parameter.')
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

# Check input arguments validation
for path in glob.glob(args.img_glob):
    assert os.path.isfile(path), '%s not found' % path
for path in glob.glob(args.line_glob):
    assert os.path.isfile(path), '%s not found' % path
assert os.path.isdir(args.output_dir), '%s is not a directory' % args.output_dir
for rotate in args.rotate:
    assert 0 <= rotate and rotate <= 1, 'elements in --rotate should in [0, 1]'


# Prepare model
backend = args.backend.lower()
net = models[backend]()
net = nn.DataParallel(net).to(device)
net.load_state_dict(torch.load(args.ckpt))

# Load path to visualization
img_paths = sorted(glob.glob(args.img_glob))
line_paths = sorted(glob.glob(args.line_glob))
assert len(img_paths) == len(line_paths), '# of input mismatch for each channels'


# Process each input
for i_path, l_path in zip(img_paths, line_paths):
    print('img  path:', i_path)
    print('line path:', l_path)

    # Load and cat input images
    i_img = np.array(Image.open(i_path), np.float32) / 255
    l_img = np.array(Image.open(l_path), np.float32) / 255
    x_img = np.concatenate([
        i_img.transpose([2, 0, 1])], axis=0)

    # Augment data
    x_imgs_augmented, aug_type = augment(x_img, args.flip, args.rotate)

    # Feedforward and extract output images
    with torch.no_grad():
        x = torch.FloatTensor(x_imgs_augmented).to(device)
        de_list = net(x)

        edg_tensor = torch.sigmoid(de_list[:,:-1])
        cor_tensor = torch.sigmoid(de_list[:,-1:])

        # Recover the effect from augmentation
        edg_img = augment_undo(edg_tensor.cpu().numpy(), aug_type)
        cor_img = augment_undo(cor_tensor.cpu().numpy(), aug_type)

    # Merge all results from augmentation
    edgmap = edg_img.transpose([0, 2, 3, 1]).mean(0).copy()
    cormap = cor_img.transpose([0, 2, 3, 1]).mean(0)[..., 0].copy()

    with open(args.layout_gt) as f:
        cor_id_gt = np.array([line.split() for line in f], np.float32)

    # Post processing to extract layout
    cor_id = get_ini_cor(cormap, args.d1, args.d2)
    if args.post_optimization:
        cor_id = optimize_cor_id(cor_id, edgmap, cormap,
                                 num_iters=100, verbose=False)

    # Draw extracted layout on source image
    bon_img = draw_boundary_from_cor_id(cor_id.copy(), i_img * 255)
    bon_img = draw_gt_from_cor_id(cor_id_gt.copy(), bon_img)

    # Composite all result in one image
    all_in_one = 0.3 * edgmap + 0.3 * cormap[..., None] + 0.4 * i_img
    all_in_one = draw_boundary_from_cor_id(cor_id.copy(), all_in_one * 255)

    # Dump results
    basename = os.path.splitext(os.path.basename(i_path))[0]
    path_edg = os.path.join(args.output_dir, '%s_edg.png' % basename)
    path_cor = os.path.join(args.output_dir, '%s_cor.png' % basename)
    path_bon = os.path.join(args.output_dir, '%s_bon.png' % basename)
    path_all_in_one = os.path.join(args.output_dir, '%s_all.png' % basename)
    path_cor_id = os.path.join(args.output_dir, '%s_cor_id.txt' % basename)

    Image.fromarray((edgmap * 255).astype(np.uint8)).save(path_edg)
    Image.fromarray((cormap * 255).astype(np.uint8)).save(path_cor)
    Image.fromarray(bon_img).save(path_bon)
    Image.fromarray(all_in_one).save(path_all_in_one)
    with open(path_cor_id, 'w') as f:
        for x, y in cor_id:
            f.write('%.6f %.6f\n' % (x, y))
