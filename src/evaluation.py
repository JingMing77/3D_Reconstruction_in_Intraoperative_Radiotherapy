import os, sys

import numpy as np

sys.path.append('..')
import torch

from utils.io_utils import *
from utils.evaluation_utils import *
from options import config_parser


def load_txt(image_list, txt_dir, H, W):
    depths = []
    for image_name in image_list:
        txt_name = image_name.replace('.jpg', '.txt')
        data = np.loadtxt(os.path.join(txt_dir, txt_name))
        disp0 = data[:, 2]
        disp0 = np.reshape(disp0, (H, W))
        disp = 1.0 / (disp0 + 1e-6)
        disp[disp0 == 0] = 0

        depths.append(disp)

    return np.stack(depths)


def main(args):
    image_list = load_img_list(args.datadir, load_test=False)
    # image_list = load_img_list(args.datadir, load_test=True)

    prior_path = os.path.join(args.basedir, args.expname, 'depth_priors', 'results')
    nerf_path = os.path.join(args.basedir, args.expname, 'nerf', 'results')
    filter_path = os.path.join(args.basedir, args.expname, 'filter')

    gt_depths, _ = load_gt_depths(image_list, os.path.join(args.datadir, 'l'), blender=args.blender)
    tissue_masks = load_masks(os.path.join(args.datadir, 'mask_l'), image_list)
    print("prior depth evaluation:")
    prior_depths = load_depths(image_list, prior_path)
    # prior_depths = load_txt(image_list, os.path.join(args.datadir, 'ad'), gt_depths.shape[1], gt_depths.shape[2])
    # prior_depths = load_depths(image_list,  os.path.join(args.datadir, 'colmap'))
    depth_evaluation(gt_depths, prior_depths, gt_masks=tissue_masks, savedir=prior_path, blender=args.blender)

    print("nerf depth evaluation:")
    nerf_depths = load_depths(image_list, nerf_path)
    depth_evaluation(gt_depths, nerf_depths, gt_masks=tissue_masks, savedir=nerf_path, blender=args.blender)

    print("filter depth evaluation:")
    filter_depths = load_depths(image_list, filter_path)
    depth_evaluation(gt_depths, filter_depths, gt_masks=tissue_masks, savedir=filter_path, blender=args.blender)

    image_list_all = load_img_list(args.datadir, load_test=True)
    image_list_test = list(set(image_list_all) - set(image_list))
    nerf_rgbs = load_rgbs_np(image_list, nerf_path,
                             use_cv2=False, is_png=True)
    gt_rgbs = load_rgbs_np(image_list,
                           os.path.join(args.datadir, 'l', 'images_{}'.format(args.factor)),
                           use_cv2=False, is_png=True)
    print("nerf novel view synthesis evaluation:")
    with torch.no_grad():
        print("train set rgbs: ")
        rgb_evaluation(gt_rgbs, nerf_rgbs, savedir=nerf_path)
        print("test set rgbs: ")
        nerf_rgbs = load_rgbs_np(image_list_test, nerf_path,
                                 use_cv2=False, is_png=True)
        gt_rgbs = load_rgbs_np(image_list_test,
                               os.path.join(args.datadir, 'l', 'images_{}'.format(args.factor)),
                               use_cv2=False, is_png=True)
        rgb_evaluation(gt_rgbs, nerf_rgbs, savedir=nerf_path)


if __name__ == '__main__':
    parser = config_parser()
    args = parser.parse_args()
    main(args)
