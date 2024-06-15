import os
import cv2
import numpy as np
import torch
import imageio
from natsort import natsorted
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from .colmap_utils import *
import pdb


def load_img_list(datadir, load_test=False, only_test=False):
    with open(os.path.join(datadir, 'train.txt'), 'r') as f:
        lines = f.readlines()
        image_list = [line.strip() for line in lines]

    if load_test:
        with open(os.path.join(datadir, 'test.txt'), 'r') as f:
            lines = f.readlines()
            image_list += [line.strip() for line in lines]

    if only_test:
        with open(os.path.join(datadir, 'test.txt'), 'r') as f:
            lines = f.readlines()
            image_list = [line.strip() for line in lines]
    return image_list


def augment(depth, mask, focal_x, baseline, factor, depth_gt_mean,
            disp_mean, padding=False, blender=False, mask_path=None):
    """
    apply augmentation and find occluded pixels
    """
    median = np.median(depth[depth != 0])
    occ_mask = (depth > (4 * median))  # obvious noise
    occ_mask[depth < (0.1 * median)] = True
    depth[occ_mask] = 0
    mask[occ_mask] = 0

    if not blender:
        # align scale
        sc = depth_gt_mean / median

        fx = focal_x / factor
        disp = (fx * baseline / (depth * sc + 1e-6))
        disp[occ_mask] = 0
        w = disp.shape[-1]
        if padding:
            invalid_disp = 0
            disp_mean = disp_mean + w // 2
        else:
            invalid_disp = -w

        # set large/small values to be 0
        max_disp = np.percentile(disp, 98)
        min_disp = np.percentile(disp, 2)
        disp[disp > max_disp] = invalid_disp
        disp[disp < min_disp] = invalid_disp

        # real world disp 2 image
        delta_d = np.median(disp[disp != invalid_disp]) - disp_mean
        disp = disp - delta_d
        disp[disp < 2 * invalid_disp] = invalid_disp
        disp[disp > 2 * w] = invalid_disp

        disp = np.ascontiguousarray(disp, dtype=np.float32)
        mask_ = (np.abs(disp - invalid_disp) < 1e-3)

        # set occ pixels 0
        depth[mask_] = 0
        mask[mask_] = 0

    else: # blender
        assert mask_path is not None, 'provide mask_l_file for blender data!'
        mask_img = cv2.imread(mask_path, 0)
        if mask_img.shape != depth.shape:
            mask_img = cv2.resize(mask_img, (depth.shape[1], depth.shape[0]))
        inf_val = np.max(depth) * 1.5
        depth[mask_img == 0] = inf_val  # background
        mask[mask_img == 0] = 1

    # return image
    return depth, mask


def load_colmap(image_list, datadir, H=None, W=None, doAugment=False, logdir=None, load_right=False,
                focal_x=None, baseline=None, depth_gt_mean=None, masks_file=None, save_path=None):
    depths = []
    masks = []

    ply_path = os.path.join(datadir, 'l', 'dense', 'fused.ply')
    ply_masks = read_ply_mask(ply_path)

    N_views = len(os.listdir(os.path.join(datadir, 'l', 'images')))
    if load_right:
        depth_path = os.path.join(datadir, 'l', 'dense/stereo/depth_maps')
        if len(os.listdir(depth_path)) == 2 * N_views:
            import warnings
            warnings.warn("only left imgs {} used in colmap".format(N_views))
            print()
            print("load left depths and masks for depth training")
        else:
            nums = [int(name.split('.')[0]) for name in image_list]
            image_list.extend([f"{i + N_views}.jpg" for i in nums])

    for image_name in image_list:
        depth_path = os.path.join(datadir, 'l', 'dense/stereo/depth_maps', image_name + '.geometric.bin')
        depth = read_array(depth_path)
        mask = ply_masks[image_name]

        factor = 1.0
        if H is not None:
            factor = depth.shape[1] / W
            depth = cv2.resize(depth, (W, H))
            mask = cv2.resize(mask, (W, H))
        if doAugment:
            disp_mean_id = (int(image_name[:-4]) - 1) % N_views
            disp_mean = np.load(os.path.join(logdir, 'matches', 'disp_mean.npy'))[disp_mean_id]
            mask_path = None
            blender_dataset = False
            if masks_file is not None:
                mask_path = os.path.join(masks_file, image_name[:-4] + '.png')
                blender_dataset = True
            depth, mask = \
                augment(depth, mask, focal_x, baseline, factor, depth_gt_mean, disp_mean,
                        blender=blender_dataset, mask_path=mask_path)
        else:
            median = np.median(depth[depth != 0])
            occ_mask = (depth > (4 * median))  # noise
            occ_mask[depth < (0.1 * median)] = True
            depth[occ_mask] = 0
            mask[occ_mask] = 0
            depth = cv2.medianBlur(depth, 3)

        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)
            np.save(os.path.join(save_path, image_name.replace('.jpg', '.npy')), depth)
            depth_ = 1. / (depth + 1e-6)
            depth_[depth == 0] = 0
            d_sc = (depth_ - np.min(depth_)) / (np.max(depth_) - np.min(depth_))
            cv2.imwrite(os.path.join(save_path, "{}".format(image_name)),
                        cv2.applyColorMap(np.uint8(d_sc * 255), cv2.COLORMAP_MAGMA))
            cv2.imwrite(os.path.join(save_path, "mask_{}".format(image_name)), np.uint8(mask * 255))

        depths.append(depth)
        masks.append(mask > 0.5)

    return np.stack(depths), np.stack(masks)


def load_gt_depths(image_list, datadir, H=None, W=None, blender=False):
    depths = []
    masks = []

    for image_name in image_list:
        frame_id = image_name.split('.')[0]
        is_disp = False
        if os.path.exists(os.path.join(datadir, 'depth')):
            depth_path = os.path.join(datadir, 'depth', '{}.jpg'.format(frame_id))
        else:
            depth_path = os.path.join(datadir, 'disparity', '{}.jpg'.format(frame_id))
            is_disp = True

        depth = cv2.imread(depth_path, 0)
        if blender:
            inf_val = depth[0, 0]
            depth[depth == inf_val] = 0
        depth = depth.astype(np.float32) / 1000
        if is_disp:
            mask = (depth != 0)
            depth = (1. / (depth + 1e-6)) * mask

        if H is not None:
            mask = (depth > 0).astype(np.uint8)
            depth_resize = cv2.resize(depth, (W, H), interpolation=cv2.INTER_NEAREST)
            mask_resize = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
            depths.append(depth_resize)
            masks.append(mask_resize > 0.5)
        else:
            depths.append(depth)
            masks.append(depth > 0)

    return np.stack(depths), np.stack(masks)


def load_depths(image_list, datadir, H=None, W=None, is_disp=False, load_right=False, focal_x=None, baseline=None):
    depths = []

    for image_name in image_list:
        frame_id = image_name.split('.')[0]
        depth_path = os.path.join(datadir, '{}_depth.npy'.format(frame_id))
        if not os.path.exists(depth_path):
            depth_path = os.path.join(datadir, '{}.npy'.format(frame_id))
        depth = np.load(depth_path)

        if H is not None:
            depth_resize = cv2.resize(depth, (W, H))
            depths.append(depth_resize)
        else:
            depths.append(depth)

    if load_right:
        n_img = len(os.listdir(datadir)) // 4
        for image_name in image_list:
            frame_id = image_name.split('.')[0]
            depth_path = os.path.join(datadir, '{}_depth.npy'.format(int(frame_id) + n_img))
            if not os.path.exists(depth_path):
                depth_path = os.path.join(datadir, '{}.npy'.format(int(frame_id) + n_img))
            depth = np.load(depth_path)

            if H is not None:
                depth_resize = cv2.resize(depth, (W, H))
                depths.append(depth_resize)
            else:
                depths.append(depth)

    depths_np = np.stack(depths)

    if is_disp:        # disparity 2 depth
        if focal_x is None or baseline is None:
            raise ValueError(f' need focal_x and baseline '
                             f'information to compute depthMap from disparity')
        else:
            factor = 1.0
            if W is not None:
                depth_frame_id = image_list[0].split('.')[0]
                depth = np.load(os.path.join(datadir, depth_frame_id + '_depth.npy'))
                factor = depth.shape[1] / W
            fx = focal_x / factor
            depths_np = fx * baseline / (depths_np + 1e-6)

    return depths_np


def load_masks(mask_dir, img_list, H=None, W=None):
    print("loading masks from " + mask_dir)
    masks = []
    img_list = natsorted(img_list)
    mask_list = natsorted(f for f in os.listdir(mask_dir) if f.endswith('.png'))
    pad = mask_list[0].split('_')[-1][:-4]
    for id, img_name in enumerate(img_list):
        mask_name = img_name.replace('.jpg', '.png')
        mask = cv2.imread(os.path.join(mask_dir, mask_name), 0)
        if H is not None:
            mask = cv2.resize(mask, (W, H))
        mask[mask > 0] = 1
        masks.append(mask)

    return np.stack(masks)


def pil_loader(path):
    from PIL import Image
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def load_rgbs(image_list, datadir, H=None, W=None, is_png=False):
    from PIL import Image
    to_tensor = transforms.ToTensor()
    # resize = transforms.Resize((H, W), interpolation=Image.ANTIALIAS)
    resize = transforms.Resize((H, W), interpolation=InterpolationMode.LANCZOS)
    rgbs = []

    for image_name in image_list:
        if is_png:
            image_name = image_name.replace('.jpg', '.png')
        rgb_path = os.path.join(datadir, image_name)
        rgb = pil_loader(rgb_path)
        if H is not None:
            rgb = resize(rgb)

        rgbs.append(to_tensor(rgb))

    return torch.stack(rgbs)


def load_rgbs_np(image_list, datadir, H=None, W=None, is_png=False, use_cv2=True):
    rgbs = []

    for image_name in image_list:
        if is_png:
            image_name = image_name.replace('.jpg', '.png')
        rgb_path = os.path.join(datadir, image_name)
        if use_cv2:
            rgb = cv2.imread(rgb_path)
        else:
            rgb = imageio.imread(rgb_path)[..., :3] / 255.0

        if H is not None:
            if use_cv2:
                rgb = cv2.resize(rgb, (W, H))
            else:
                rgb = resize(rgb, (W, H))

        rgbs.append(rgb)

    return np.stack(rgbs)


def visualize_depth(depth, mask=None, depth_min=None, depth_max=None, direct=False):
    """Visualize the depth map with colormap.
       Rescales the values so that depth_min and depth_max map to 0 and 1,
       respectively.
    """
    depth0 = depth.copy()
    if not direct:
        depth = 1.0 / (depth + 1e-6)  # Points with larger depth values should be darker
        depth[depth0 == 0] = 0
    invalid_mask = np.logical_or(np.isnan(depth), np.logical_not(np.isfinite(depth)))
    if mask is not None:
        invalid_mask += np.logical_not(mask)
    if depth_min is None:
        depth_min = np.percentile(depth[np.logical_not(invalid_mask)], 5)
    if depth_max is None:
        depth_max = np.percentile(depth[np.logical_not(invalid_mask)], 95)
    depth[depth < depth_min] = depth_min
    depth[depth > depth_max] = depth_max
    depth[invalid_mask] = depth_max

    depth_scaled = (depth - depth_min) / (depth_max - depth_min)
    depth_scaled_uint8 = np.uint8(depth_scaled * 255)
    depth_color = cv2.applyColorMap(depth_scaled_uint8, cv2.COLORMAP_MAGMA)
    depth_color[invalid_mask, :] = 0

    return depth_color

