import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.nerf.importance_sampling import importance_sampling_coords


def refine_area(masks, depths, imgs=None, block_size=16, save_path=None):
    N, H, W = masks.shape
    masks_res = []
    kernel = np.ones((15, 15), np.uint8)
    for i in range(N):
        img = imgs[i, ...] if imgs is not None else None
        depth = depths[i, ...]
        mask = masks[i, ...].astype(np.float32)  # 1 or 0
        mask_ = cv2.erode(mask, kernel, iterations=1)
        grad_x = cv2.Sobel(depth, cv2.CV_32F, 1, 0)
        grad_y = cv2.Sobel(depth, cv2.CV_32F, 0, 1)
        Scale_absX, Scale_absY = abs(grad_x), abs(grad_y)
        grad = cv2.addWeighted(Scale_absX, 0.5, Scale_absY, 0.5, 0)

        grad = grad * mask_
        invalid = (depth < 1)

        for ii in range(0, grad.shape[0], block_size):
            for jj in range(0, grad.shape[1], block_size):
                grad_block = grad[ii: ii + block_size, jj: jj + block_size]
                grad_th = np.percentile(grad_block, 80)
                invalid[ii: ii + block_size, jj: jj + block_size] = (grad_block > grad_th)
        invalid = invalid.astype(np.float32)

        mask_r = mask * invalid
        mask_r = cv2.dilate(mask_r, kernel, iterations=1)
        # only reflect
        if img is not None:
            maskt = (img > 0.4).all(axis=2)
            mask_r = mask_r * maskt * mask_
        masks_res.append(mask_r)

        if save_path is not None:
            cv2.imwrite(os.path.join(save_path, 'mask_refine_{}.jpg'.format(i + 1)), masks_res[-1] * 255)

    return np.stack(masks_res)



def apply_grad_conf(masks, imgs, depth_confidences, w_grad=1, w_mask=1, w_reflect=0.3, vis=False):
    N, H, W = masks.shape
    masks_res = []
    for i in range(N):
        depth_confidence = depth_confidences[i, ...]
        img = imgs[i, ...]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # mask
        mask = masks[i, ...]  # 1 or 0
        mask = np.logical_not(mask)  # 1 for background 0 for tissue
        mask = mask.astype(np.float32)
        # grad
        grad_x = cv2.Sobel(img, cv2.CV_32F, 1, 0)
        grad_y = cv2.Sobel(img, cv2.CV_32F, 0, 1)
        Scale_absX = abs(grad_x)
        Scale_absY = abs(grad_y)
        grad = cv2.addWeighted(Scale_absX, 0.5, Scale_absY, 0.5, 0)
        grad = cv2.normalize(grad, None, 0, 1, cv2.NORM_MINMAX)  # 0 - 1
        # reflect areas
        reflect = (img > 240).astype(np.float32)  # 0 - 1
        # result: 0 - 1
        res = -((mask * w_mask) + (grad * w_grad) + (reflect * w_reflect))
        res_min = np.min(res)
        res_max = np.max(res)
        res = (res - res_min) / (res_max - res_min)
        # depth_confidence
        err_bound = np.percentile(depth_confidence, 95)
        res[depth_confidence > err_bound] /= 2.
        res[depth_confidence == 0] = 0
        masks_res.append(res)
        # cv2.imshow("1", res)
        # cv2.waitKey(0)
        if vis and i == 0:
            cv2.imwrite("grad.png", np.uint8(grad * 255))
            cv2.imwrite("reflect.png", np.uint8(reflect * 255))
            mask = np.uint8(masks_res[i] * 255)
            # mask = cv2.applyColorMap(mask, cv2.COLORMAP_MAGMA)
            cv2.imwrite("mask_grad_reflect.png", mask)

    return np.stack(masks_res)


def align_scales(depth_priors, colmap_depths, colmap_masks, poses, sc, i_train, i_test):
    ratio_priors = []
    N_images = depth_priors.shape[0]
    # depth_priors = 1 / depth_priors
    for i in range(N_images // 2):
        ratio_priors.append(np.median(colmap_depths[i][colmap_masks[i]]) / np.median(depth_priors[i][colmap_masks[i]]))
    ratio_priors += ratio_priors
    ratio_priors = np.stack(ratio_priors)
    ratio_priors = ratio_priors[:, np.newaxis, np.newaxis]

    # if len(i_test) > 0:
    #     neighbor_idx = cal_neighbor_idx(poses, i_train, i_test)
    #     depth_priors_test = depth_priors[i_train][neighbor_idx]
    #     ratio_priors_test = ratio_priors[i_train][neighbor_idx]
    #     depth_priors = np.concatenate([depth_priors, depth_priors_test], axis=0)
    #     ratio_priors = np.concatenate([ratio_priors, ratio_priors_test], axis=0)

    depth_priors = depth_priors * sc * ratio_priors  # align scales
    return depth_priors


def cal_colmap_confidences(depths, T, K, topk=4, save_dir=None):
    _, H, W = depths.shape
    view_num = depths.shape[0]
    invK = torch.inverse(K)
    batch_K = torch.unsqueeze(K, 0).repeat(view_num, 1, 1)
    batch_invK = torch.unsqueeze(invK, 0).repeat(view_num, 1, 1)
    invT = torch.inverse(T)
    pix_coords = calculate_coords(W, H)
    cam_points = BackprojectDepth(depths, batch_invK, pix_coords)
    depth_confidences = []

    for i in range(depths.shape[0]):
        cam_points_i = cam_points[i:i + 1].repeat(view_num, 1, 1)
        T_i = torch.matmul(invT, T[i:i + 1].repeat(view_num, 1, 1))
        pix_coords_ref = Project3D(cam_points_i, batch_K, T_i, H, W)
        depths_ = Project3D_depth(cam_points_i, batch_K, T_i, H, W)
        depths_proj = F.grid_sample(depths.unsqueeze(1), pix_coords_ref, padding_mode="zeros").squeeze()
        error = torch.abs(depths_proj - depths_) / (depths_ + 1e-7)
        depth_confidence, _ = error.topk(k=topk, dim=0, largest=False)
        depth_confidence = depth_confidence.mean(0).cpu().numpy()
        depth_confidence[depth_confidence < 0] = 1
        depth_confidence[(depths[i, ...]).cpu().numpy() == 0] = 0
        depth_confidences.append(depth_confidence)
    if save_dir is not None:
        print('save depth_confidences in ' + save_dir)
        for idx, depth_confidence_save in enumerate(depth_confidences):
            d_min = np.percentile(depth_confidence_save, 5)
            d_max = np.percentile(depth_confidence_save, 95)
            depth_confidence_save[depth_confidence_save < d_min] = d_min
            depth_confidence_save[depth_confidence_save > d_max] = d_max
            d_min, d_max = np.min(depth_confidence_save), np.max(depth_confidence_save)
            depth_confidence_save = (depth_confidence_save - d_min) / (d_max - d_min)
            depth_confidence_save_int8 = np.uint8(depth_confidence_save * 255)
            depth_confidence_save_color = cv2.applyColorMap(depth_confidence_save_int8, cv2.COLORMAP_MAGMA)
            cv2.imwrite(os.path.join(save_dir, "{}_depth_confidence.png".format(idx + 1)), depth_confidence_save_color)
    return np.stack(depth_confidences, 0)


def cal_depth_confidences(depths, T, K, i_train, topk=4, save_dir=None):
    _, H, W = depths.shape
    view_num = len(i_train)
    invK = torch.inverse(K)
    batch_K = torch.unsqueeze(K, 0).repeat(view_num, 1, 1)
    batch_invK = torch.unsqueeze(invK, 0).repeat(depths.shape[0], 1, 1)
    T_train = T[i_train]
    invT = torch.inverse(T_train)
    pix_coords = calculate_coords(W, H)
    cam_points = BackprojectDepth(depths, batch_invK, pix_coords)
    depth_confidences = []

    for i in range(depths.shape[0]):
        cam_points_i = cam_points[i:i + 1].repeat(view_num, 1, 1)
        T_i = torch.matmul(invT, T[i:i + 1].repeat(view_num, 1, 1))
        pix_coords_ref = Project3D(cam_points_i, batch_K, T_i, H, W)
        depths_ = Project3D_depth(cam_points_i, batch_K, T_i, H, W)
        depths_proj = F.grid_sample(depths[i_train].unsqueeze(1), pix_coords_ref,
                                    padding_mode="zeros").squeeze()
        error = torch.abs(depths_proj - depths_) / (depths_ + 1e-7)
        depth_confidence, _ = error.topk(k=topk, dim=0, largest=False)
        depth_confidence = depth_confidence.mean(0).cpu().numpy()
        depth_confidence[depth_confidence < 0] = 0
        depth_confidences.append(depth_confidence)
    if save_dir is not None:
        print('save depth_confidences in ' + save_dir)
        for idx, depth_confidence_save in enumerate(depth_confidences):
            d_min = np.percentile(depth_confidence_save, 5)
            d_max = np.percentile(depth_confidence_save, 95)
            depth_confidence_save[depth_confidence_save < d_min] = d_min
            depth_confidence_save[depth_confidence_save > d_max] = d_max
            d_min, d_max = np.min(depth_confidence_save), np.max(depth_confidence_save)
            depth_confidence_save = (depth_confidence_save - d_min) / (d_max - d_min)
            depth_confidence_save_int8 = np.uint8(depth_confidence_save * 255)
            depth_confidence_save_color = cv2.applyColorMap(depth_confidence_save_int8, cv2.COLORMAP_MAGMA)
            cv2.imwrite(os.path.join(save_dir, "{}_depth_confidence.png".format(idx + 1)), depth_confidence_save_color)
    return np.stack(depth_confidences, 0)


def calculate_coords(W, H):
    meshgrid = np.meshgrid(range(W), range(H), indexing='xy')
    id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
    id_coords = torch.from_numpy(id_coords)
    pix_coords = torch.stack(
        [id_coords[0].view(-1), id_coords[1].view(-1)], 0)
    ones = torch.ones(1, H * W)
    pix_coords = pix_coords.to(ones.device)
    pix_coords = torch.cat([pix_coords, ones], 0)
    return pix_coords


def BackprojectDepth(depth, invK, pix_coords):
    batch_size, H, W = depth.shape
    ones = torch.ones(batch_size, 1, H * W)
    cam_points = torch.matmul(invK[:, :3, :3], pix_coords)
    cam_points = depth.view(batch_size, 1, -1) * cam_points
    cam_points = torch.cat([cam_points, ones], 1)
    return cam_points


def Project3D(points, K, T, H, W, eps=1e-7):
    batch_size = points.shape[0]
    P = torch.matmul(K, T)[:, :3, :]

    cam_points = torch.matmul(P, points)

    pix_coords = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1) + eps)
    pix_coords = pix_coords.view(batch_size, 2, H, W)
    pix_coords = pix_coords.permute(0, 2, 3, 1)
    pix_coords[..., 0] /= W - 1
    pix_coords[..., 1] /= H - 1
    pix_coords = (pix_coords - 0.5) * 2
    return pix_coords


def Project3D_depth(points, K, T, H, W, eps=1e-7):
    batch_size = points.shape[0]
    P = torch.matmul(K, T)[:, :3, :]

    cam_points = torch.matmul(P, points)
    return cam_points[:, 2, :].view(batch_size, H, W)


def cal_neighbor_idx(poses, i_train, i_test):
    angles = []
    trans = []
    for i in range(poses.shape[0]):
        angles.append(vec_from_R(poses[i].copy()))
        trans.append(poses[i][:3, 3].copy())
    angles, trans = np.stack(angles), np.stack(trans)
    angle_dis = angles[i_test][:, None] - angles[i_train][None, :]
    tran_dis = trans[i_test][:, None] - trans[i_train][None, :]
    angle_dis = (angle_dis ** 2).sum(-1)
    angle_sort = np.argsort(angle_dis, axis=1)
    tran_dis = (tran_dis ** 2).sum(-1)
    tran_sort = np.argsort(tran_dis, axis=1)
    x_range = np.arange(len(i_test))[:, None].repeat(len(i_train), axis=1)
    y_range = np.arange(len(i_train))[None].repeat(len(i_test), axis=0)
    angle_dis[x_range, angle_sort] = y_range
    tran_dis[x_range, tran_sort] = y_range
    final_score = 100 * (angle_dis + tran_dis) + angle_dis
    neighbor_idx = np.argmin(final_score, axis=1)
    return neighbor_idx


def vec_from_R(rot):
    temp = (rot[:3, :3] - rot[:3, :3].transpose(1, 0)) / 2
    angle_vec = np.stack([temp[2][1], -temp[2][0], temp[1][0]])
    angle = np.linalg.norm(angle_vec)
    axis = angle_vec / angle
    return np.arcsin(angle) * axis


def importance_sampling(rays_rgb, H, W, tissue_masks, device, sample_rate):
    rays_rgb0 = np.reshape(rays_rgb, (rays_rgb.shape[0], H * W, 4, 3))  # [N, H * W, ro+rd+rgb(d)+prior, 3]
    ray_importance_maps = torch.Tensor(tissue_masks).to(device)
    coords = torch.stack(torch.meshgrid(torch.linspace(0, H - 1, H),
                                        torch.linspace(0, W - 1, W)), -1)  # (H, W, 2)
    coords = torch.reshape(coords, [-1, 2])  # (H * W, 2)
    sample_per_img = int(sample_rate * H * W)
    rays_rgb_t = np.zeros((rays_rgb0.shape[0], sample_per_img, 4, 3))
    for i in range(tissue_masks.shape[0]):
        ray_importance_map = ray_importance_maps[i]
        select_inds, _, cdf = importance_sampling_coords(
            ray_importance_map[coords[:, 0].long(), coords[:, 1].long()].unsqueeze(0), sample_per_img)
        select_inds = torch.max(torch.zeros_like(select_inds), select_inds)
        select_inds = torch.min((coords.shape[0] - 1) * torch.ones_like(select_inds), select_inds)
        select_inds = select_inds.squeeze(0)
        select_inds = select_inds.cpu().numpy()
        rays_rgb_t[i] = rays_rgb0[i, select_inds, ...]

        # sample_img = np.zeros(H * W)
        # sample_img[select_inds] = 255
        # sample_img = sample_img.reshape((H, W))
        # cv2.imshow("1", sample_img)
        # cv2.waitKey(0)

    return rays_rgb_t
