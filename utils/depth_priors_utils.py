import os

import cv2
import imageio
import numpy as np
import torch
from scipy.interpolate import griddata

from models.nerf.run_nerf_helpers import to8b
from utils.io_utils import visualize_depth


def compute_depth_loss(depth_pred, depth_gt, mask_gt, inv_depth=1):
    loss_list = []
    for pred, gt, mask in zip(depth_pred, depth_gt, mask_gt):
        log_pred = torch.log(pred[mask])
        log_target = inv_depth * torch.log(gt[mask])
        alpha = (log_target - log_pred).sum() / mask.sum()
        log_diff = torch.abs((log_pred - log_target + alpha))
        d = 0.05 * 0.2 * (log_diff.sum() / mask.sum())
        loss_list.append(d)

    return torch.stack(loss_list, 0).mean()


def generate_imgc(imgs_l, imgs_r, depths_lr, poses_lrc, K,
                  generate_from='l', method='linear', save_dir=None):
    print("start: generate imgc, depths_c and masks_lrc ...")

    imgs_c = []
    depths_c = []
    masks_c = []
    N_imgs, _, H, W = imgs_l.shape  # N, 3, H, W
    depths_l = depths_lr[:N_imgs, ...]
    depths_r = depths_lr[N_imgs:, ...]
    poses_l, poses_r, poses_c = \
        poses_lrc[:N_imgs, :, :4], poses_lrc[N_imgs:2 * N_imgs, :, :4], poses_lrc[2 * N_imgs:, :, :4]
    for i in range(N_imgs):
        # Compute pointcloudl and pointcloudr from uv
        depthl = depths_l[i, ...]
        depthr = depths_r[i, ...]
        imgl, imgr = imgs_l[i, ...], imgs_r[i, ...]
        imgl, imgr = np.reshape(imgl, (3, H * W)), np.reshape(imgr, (3, H * W))
        posel, poser, posec = poses_l[i, ...], poses_r[i, ...], poses_c[i, ...]
        posel = np.vstack((posel, [0, 0, 0, 1]))
        poser = np.vstack((poser, [0, 0, 0, 1]))
        posec = np.vstack((posec, [0, 0, 0, 1]))

        pointsl = np.zeros((3, H * W), dtype=np.float32)
        pointsr = np.zeros((3, H * W), dtype=np.float32)
        x, y = np.meshgrid(np.arange(W), np.arange(H))
        x = x.flatten()
        y = y.flatten()
        pointsl[0, :] = (x - K[0, 2]) * depthl.flatten() / K[0, 0]
        pointsl[1, :] = (y - K[1, 2]) * depthl.flatten() / K[1, 1]
        pointsl[2, :] = depthl.flatten()
        pointsr[0, :] = (x - K[0, 2]) * depthr.flatten() / K[0, 0]
        pointsr[1, :] = (y - K[1, 2]) * depthr.flatten() / K[1, 1]
        pointsr[2, :] = depthr.flatten()

        # depth c from triangle Ol_P_Or
        dist_pointsl = np.linalg.norm(pointsl, axis=0)  # Ol_P
        dist_pointsr = np.linalg.norm(pointsr, axis=0)  # Or_P
        dist_camlr = np.linalg.norm(posel[:3, 3] - poser[:3, 3])  # Ol_Or: baseline
        dist_camlr = np.full(dist_pointsr.shape, dist_camlr)
        s = (dist_pointsl + dist_pointsr + dist_camlr) / 2
        area = np.sqrt(s * (s - dist_pointsl) * (s - dist_pointsr) * (s - dist_camlr))
        depthc = 2 * area / dist_camlr  # Zc
        depthc = np.nan_to_num(depthc, nan=0, posinf=0, neginf=0)
        depthc = np.reshape(depthc, (H, W))
        depths_c.append(depthc)
        mask_c = depthc == 0

        # points in world coordinates
        pointsl = np.vstack((pointsl, np.ones((1, pointsl.shape[1]))))
        pointsr = np.vstack((pointsr, np.ones((1, pointsr.shape[1]))))
        points_world_l = np.dot(np.linalg.inv(posel), pointsl)
        points_world_r = np.dot(np.linalg.inv(poser), pointsr)
        pointsc_l = posec @ points_world_l
        pointsc_r = posec @ points_world_r

        # cv2.imshow("mask_c", np.uint8(mask_c) * 255)
        # cv2.imshow("depth_c", depthc)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # continue

        # points in center_cam coordinates
        pointsc = np.zeros((3, H * W), dtype=np.float32)
        pointsc[0, :] = (x - K[0, 2]) * depthc.flatten() / K[0, 0]
        pointsc[1, :] = (y - K[1, 2]) * depthc.flatten() / K[1, 1]
        pointsc[2, :] = depthc.flatten()
        pointsc = np.vstack((pointsc, np.ones((1, pointsc.shape[1]))))

        # mask_c for invalid ps ( abs(p_c_form_uv - p_c_from_l/r) > th)
        points_c_diff = np.abs(pointsc - pointsc_l)
        points_c_diff = np.reshape(points_c_diff[:3, :].T, (H, W, 3))
        th = np.median(points_c_diff) * 3 * 3
        maskct = (np.sum(points_c_diff, axis=2) > th)
        mask_c = np.logical_or(mask_c, maskct)
        points_c_diff = np.abs(pointsc - pointsc_r)
        points_c_diff = np.reshape(points_c_diff[:3, :].T, (H, W, 3))
        th = np.median(points_c_diff) * 3 * 3
        maskct = (np.sum(points_c_diff, axis=2) > th)
        mask_c = np.logical_or(mask_c, maskct)

        # Project valid pointcloud to imgl_c
        points_world_c = np.dot(np.linalg.inv(posec), pointsc)
        if generate_from == 'l':
            uv_l = np.dot(K, posel[:3, :]) @ points_world_c
            uv_l = uv_l[:2, :] / uv_l[2, :]
            img_c = griddata(np.array((x, y)).transpose(), np.array(imgl).transpose(), uv_l.transpose(),
                             method=method)
        elif generate_from == 'r':
            uv_r = np.dot(K, poser[:3, :]) @ points_world_c
            uv_r = uv_r[:2, :] / uv_r[2, :]
            img_c = griddata(np.array((x, y)).transpose(), np.array(imgr).transpose(), uv_r.transpose(),
                             method=method)
        else:
            raise ValueError("Can not generate_from: {}".format(generate_from))
        img_c = np.nan_to_num(img_c, nan=0, posinf=0, neginf=0)
        imgs_c.append(np.reshape(img_c.transpose(), (3, H, W)))
        maskct = img_c == 0
        maskct = np.reshape(maskct[:, 0], (H, W))
        mask_c = np.logical_or(mask_c, maskct)
        masks_c.append(mask_c)

        # cv2.imshow("img_c", np.reshape(img_c, (H, W, 3)))
        # cv2.imshow("mask_c", np.uint8(mask_c) * 255)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        if save_dir is not None:
            rgb8 = to8b(np.reshape(img_c, (H, W, 3)))
            frame_name = os.path.join(save_dir, '{}_cen.png'.format(i + 1))
            imageio.imwrite(frame_name, rgb8)
            frame_name = os.path.join(save_dir, '{}_cen_d.png'.format(i + 1))
            depthc_vis = visualize_depth(depthc)
            cv2.imwrite(frame_name, depthc_vis)

    masks_c = np.logical_not(masks_c)
    return np.stack(imgs_c), np.stack(depths_c), np.stack(masks_c)


def load_nerf_result(res_dir, H=None, W=None):
    lrc_poses = np.load(os.path.join(res_dir, 'poses_lrc_bounds.npy'))
    lrc_poses = lrc_poses.astype(np.float32)
    N_imgs = lrc_poses.shape[0] // 3
    lr_depths = []
    for i in range(N_imgs * 2):
        depth_path = os.path.join(res_dir, '{}_depth.npy'.format(i + 1))
        depth_i = np.load(depth_path)
        if H is not None:
            depth_i = cv2.resize(depth_i, (W, H))
        lr_depths.append(depth_i)
    return lrc_poses, np.stack(lr_depths)
