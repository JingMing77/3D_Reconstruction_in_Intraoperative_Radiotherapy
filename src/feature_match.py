#! /usr/bin/env python3
#
# %BANNER_BEGIN%
# ---------------------------------------------------------------------
# %COPYRIGHT_BEGIN%
#
#  Magic Leap, Inc. ("COMPANY") CONFIDENTIAL
#
#  Unpublished Copyright (c) 2020
#  Magic Leap, Inc., All Rights Reserved.
#
# NOTICE:  All information contained herein is, and remains the property
# of COMPANY. The intellectual and technical concepts contained herein
# are proprietary to COMPANY and may be covered by U.S. and Foreign
# Patents, patents in process, and are protected by trade secret or
# copyright law.  Dissemination of this information or reproduction of
# this material is strictly forbidden unless prior written permission is
# obtained from COMPANY.  Access to the source code contained herein is
# hereby forbidden to anyone except current COMPANY employees, managers
# or contractors who have executed Confidentiality and Non-disclosure
# agreements explicitly covering such access.
#
# The copyright notice above does not evidence any actual or intended
# publication or disclosure  of  this source code, which includes
# information that is confidential and/or proprietary, and is a trade
# secret, of  COMPANY.   ANY REPRODUCTION, MODIFICATION, DISTRIBUTION,
# PUBLIC  PERFORMANCE, OR PUBLIC DISPLAY OF OR THROUGH USE  OF THIS
# SOURCE CODE  WITHOUT THE EXPRESS WRITTEN CONSENT OF COMPANY IS
# STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE LAWS AND
# INTERNATIONAL TREATIES.  THE RECEIPT OR POSSESSION OF  THIS SOURCE
# CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS
# TO REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE,
# USE, OR SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.
#
# %COPYRIGHT_END%
# ----------------------------------------------------------------------
# %AUTHORS_BEGIN%
#
#  Originating Authors: Paul-Edouard Sarlin
#                       Daniel DeTone
#                       Tomasz Malisiewicz
#
# %AUTHORS_END%
# --------------------------------------------------------------------*/
# %BANNER_END%
import os

import cv2
import numpy as np
import matplotlib.cm as cm
import torch
from natsort import natsorted

from models.spsg.matching import Matching
from models.spsg.utils import (make_matching_plot, AverageTimer, read_image)

def sparse_disp(output_dir, img_dir, resize, size0, img_list,
                th_conf=0.7, th_y=20, expand_r=5, search_disp=3, ncc_win=5):
    if len(resize) == 1 and resize[0] < 0:
        resize = (size0[1], size0[0])
    input_dir = output_dir
    height, width = resize[1], resize[0]
    height0, width0 = size0[0], size0[1]
    factor = width / width0

    for id, img_name in enumerate(img_list):

        # load .npz
        npz_name = img_name.replace('.jpg', '_matches.npz')
        match_data = np.load(os.path.join(input_dir, npz_name))
        # print(match_data.files)  # ['keypoints0', 'keypoints1', 'matches', 'match_confidence']

        kp0 = match_data['keypoints0']
        kp1 = match_data['keypoints1']
        matches = match_data['matches']
        match_confidence = match_data['match_confidence']
        valid = (matches > -1)
        mkpts0 = kp0[valid]
        mkpts1 = kp1[matches[valid]]
        mconf = match_confidence[valid]

        # load left and right img
        img0 = cv2.imread(os.path.join(img_dir, 'l', 'rectify', img_name))
        img1 = cv2.imread(os.path.join(img_dir, 'r', 'rectify', img_name))
        disp = np.zeros(shape=(height, width))
        Nmatches = 0
        for p_i, point0 in enumerate(mkpts0):
            point1 = mkpts1[p_i]
            dx = point0[0] - point1[0]
            dy = abs(point0[1] - point1[1])
            if dy < th_y and mconf[p_i] > th_conf:
                # correct match points
                y_min = min(point0[1], point1[1])
                disp[int(y_min): int(y_min + dy), int(point0[0])] = dx
                Nmatches += 1
        print("find {} match points in image {}".format(Nmatches, img_name))

        # Calculate disparity in the field of matching points
        img0_resize = cv2.cvtColor(cv2.resize(img0, (width, height)), cv2.COLOR_BGR2GRAY)
        img1_resize = cv2.cvtColor(cv2.resize(img1, (width, height)), cv2.COLOR_BGR2GRAY)
        for p_i, point0 in enumerate(mkpts0):
            point1 = mkpts1[p_i]
            disp_cen = point0[0] - point1[0]
            # point0's neighbor: [expand_r * expand_r]
            for hh in range(-expand_r, expand_r + 1):
                for ww in range(-expand_r, expand_r + 1):
                    pix0_h = int(point0[1] + hh)
                    pix0_w = int(point0[0] + ww)
                    if pix0_h - ncc_win < 0 or pix0_h + ncc_win + 1 >= height \
                            or pix0_w - ncc_win < 0 or pix0_w + ncc_win + 1 >= width:
                        # out of range
                        continue
                    if disp[pix0_h, pix0_w] != 0:
                        # already have value
                        continue
                    pix0_win = img0_resize[(pix0_h - ncc_win): (pix0_h + ncc_win + 1),
                               (pix0_w - ncc_win): (pix0_w + ncc_win + 1)]  # (ncc_win*2+1, ncc_win*2+1)
                    pix0_win_mean = np.mean(pix0_win)
                    v0_ = pix0_win - pix0_win_mean
                    v0_2 = np.sum(v0_ ** 2)

                    ncc_max = -1
                    pix0_disp = disp_cen
                    for di in range(-search_disp, search_disp + 1):
                        pix1_h = pix0_h
                        pix1_w = int(pix0_w - (disp_cen + di))
                        if pix1_h - ncc_win < 0 or + ncc_win + 1 >= height \
                                or pix1_w - ncc_win < 0 or pix1_w + ncc_win + 1 >= width:
                            # out of range
                            continue
                        # pix1 = img1_resize[pix1_h, pix1_w]  # (1,)
                        pix1_win = img1_resize[(pix1_h - ncc_win): (pix1_h + ncc_win + 1),
                                   (pix1_w - ncc_win): (pix1_w + ncc_win + 1)]  # (ncc_win*2+1, ncc_win*2+1)
                        pix1_win_mean = np.mean(pix1_win)
                        v1_ = pix1_win - pix1_win_mean
                        v1_2 = np.sum(v1_ ** 2)
                        ncc = np.sum(v0_ * v1_) / np.sqrt(v0_2 * v1_2 + 1e-8)
                        if ncc > ncc_max:
                            ncc_max = ncc
                            pix0_disp = disp_cen + di
                    disp[pix0_h, pix0_w] = pix0_disp

        # resize
        disp_save = cv2.resize(disp, (width0, height0))
        disp_save = disp_save / 2
        cv2.imwrite(os.path.join(output_dir, '{}_sparse_disp.png'.format(id + 1)), disp_save)


def disp_supervise(output_dir, img_dir, resize, size0, img_list,
                   th_y=20, th_conf=0.7, ncc_win=5, search_disp=3):
    if len(resize) == 1 and resize[0] < 0:
        resize = (size0[1], size0[0])
    input_dir = output_dir
    height, width = resize[1], resize[0]
    height0, width0 = size0[0], size0[1]
    factor = width / width0

    for id, img_name in enumerate(img_list):

        # load .npz
        npz_name = img_name.replace('.jpg', '_matches.npz')
        match_data = np.load(os.path.join(input_dir, npz_name))
        # print(match_data.files)  # ['keypoints0', 'keypoints1', 'matches', 'match_confidence']

        kp0 = match_data['keypoints0']
        kp1 = match_data['keypoints1']
        matches = match_data['matches']
        match_confidence = match_data['match_confidence']
        valid = (matches > -1)
        mkpts0 = kp0[valid]
        mkpts1 = kp1[matches[valid]]
        mconf = match_confidence[valid]

        # load left and right img
        img0 = cv2.imread(os.path.join(img_dir, 'l', 'rectify', img_name))
        img1 = cv2.imread(os.path.join(img_dir, 'r', 'rectify', img_name))
        img0_resize = cv2.cvtColor(cv2.resize(img0, (width, height)), cv2.COLOR_BGR2GRAY)
        img1_resize = cv2.cvtColor(cv2.resize(img1, (width, height)), cv2.COLOR_BGR2GRAY)
        # mask
        mask_name = img_name.replace('jpg', 'png')
        mask0 = cv2.imread(os.path.join(img_dir, 'mask_l', mask_name), 0)
        mask0 = cv2.resize(mask0, (width, height))

        # markers
        markers = np.zeros(shape=(height, width))
        disp = np.zeros(shape=(height, width))
        disp_ref = []

        Nmatches = 0
        for p_i, point0 in enumerate(mkpts0):
            point1 = mkpts1[p_i]
            dx = point0[0] - point1[0]
            dy = abs(point0[1] - point1[1])
            if dy < th_y and mconf[p_i] > th_conf:
                # correct match points
                Nmatches += 1
                y_min = min(point0[1], point1[1])
                disp[int(y_min): int(y_min + dy), int(point0[0])] = dx
                markers[int(y_min): int(y_min + dy), int(point0[0])] = Nmatches
                disp_ref.append(dx)

        print("find {} match points in image {}".format(Nmatches, img_name))

        # define background and watershed
        markers[mask0 == 0] = 0
        markers = markers.astype(np.int32)
        img0_ = cv2.resize(img0, (width, height))
        markers = cv2.watershed(img0_, markers)

        # Calculate disparity in the field of matching points
        for marker_id in range(Nmatches):
            disp_cen = disp_ref[marker_id]
            coords = np.array(np.where(markers == marker_id))  # (2, n)
            for idx in range(coords.shape[1]):
                pix0_h = coords[0, idx]
                pix0_w = coords[1, idx]
                if pix0_h - ncc_win < 0 or pix0_h + ncc_win + 1 >= height \
                        or pix0_w - ncc_win < 0 or pix0_w + ncc_win + 1 >= width:
                    # out of range
                    continue
                if disp[pix0_h, pix0_w] != 0:
                    # already have value
                    continue
                pix0_win = img0_resize[(pix0_h - ncc_win): (pix0_h + ncc_win + 1),
                           (pix0_w - ncc_win): (pix0_w + ncc_win + 1)]  # (ncc_win*2+1, ncc_win*2+1)
                pix0_win_mean = np.mean(pix0_win)
                v0_ = pix0_win - pix0_win_mean
                v0_2 = np.sum(v0_ ** 2)

                ncc_max = -1
                pix0_disp = disp_cen
                # search disp range
                for di in range(-search_disp, search_disp + 1):
                    pix1_h = pix0_h
                    pix1_w = int(pix0_w - (disp_cen + di))
                    if pix1_h - ncc_win < 0 or + ncc_win + 1 >= height \
                            or pix1_w - ncc_win < 0 or pix1_w + ncc_win + 1 >= width:
                        # out of range
                        continue
                    # pix1 = img1_resize[pix1_h, pix1_w]  # (1,)
                    pix1_win = img1_resize[(pix1_h - ncc_win): (pix1_h + ncc_win + 1),
                               (pix1_w - ncc_win): (pix1_w + ncc_win + 1)]  # (ncc_win*2+1, ncc_win*2+1)
                    pix1_win_mean = np.mean(pix1_win)
                    v1_ = pix1_win - pix1_win_mean
                    v1_2 = np.sum(v1_ ** 2)
                    ncc = np.sum(v0_ * v1_) / np.sqrt(v0_2 * v1_2 + 1e-8)
                    if ncc > ncc_max:
                        ncc_max = ncc
                        pix0_disp = disp_cen + di
                disp[pix0_h, pix0_w] = pix0_disp

        disp[markers == -1] = 0  # bounds
        disp[markers == 0] = 0  # background

        # resize
        disp_save = cv2.resize(disp, (width0, height0))
        disp_save = disp_save / 2
        cv2.imwrite(os.path.join(output_dir, '{}_supervisor.png'.format(id + 1)), disp_save)


def cal_mean_disp(output_dir, resize, size0, img_list, th_y=20, th_conf=0.7):
    if len(resize) == 1 and resize[0] < 0:
        resize = (size0[1], size0[0])
    input_dir = output_dir
    height, width = resize[1], resize[0]
    height0, width0 = size0[0], size0[1]
    factor = width / width0

    means = []
    img_list = natsorted(img_list)
    for id, img_name in enumerate(img_list):
        # load .npz
        npz_name = img_name.replace('.jpg', '_matches.npz')
        match_data = np.load(os.path.join(input_dir, npz_name))
        # print(match_data.files)  # ['keypoints0', 'keypoints1', 'matches', 'match_confidence']

        kp0 = match_data['keypoints0']
        kp1 = match_data['keypoints1']
        matches = match_data['matches']
        match_confidence = match_data['match_confidence']
        valid = (matches > -1)
        mkpts0 = kp0[valid]
        mkpts1 = kp1[matches[valid]]
        mconf = match_confidence[valid]

        # load left and right img
        Nmatches = 0
        sum_d = 0.
        sum_conf = 0.
        for p_i, point0 in enumerate(mkpts0):
            point1 = mkpts1[p_i]
            dx = (point0[0] - point1[0]) / factor
            dy = abs(point0[1] - point1[1])
            if dy < th_y and mconf[p_i] > th_conf:
                # correct match points
                sum_d = sum_d + (mconf[p_i] * dx)
                sum_conf = sum_conf + mconf[p_i]
                Nmatches += 1
        mean_d = sum_d / sum_conf
        means.append(mean_d)
        print("find {} match points in image {}".format(Nmatches, img_name))
        print("the weighted average of disparity is {} in image {}".format(mean_d, img_name))
    means = np.stack(means)
    np.save(os.path.join(output_dir, 'disp_mean.npy'), means)


def run_spsg_match(opt):
    with torch.no_grad():
        print("---------------------------------------------------------------------------------------------")
        print("SuperGlue_Match begins ")
        opt.viz = True
        load_pre = True
        if len(opt.resize) == 2 and opt.resize[1] == -1:
            opt.resize = opt.resize[0:1]
        if len(opt.resize) == 2:
            print('Will resize to {}x{} (WxH)'.format(
                opt.resize[0], opt.resize[1]))
        elif len(opt.resize) == 1 and opt.resize[0] > 0:
            print('Will resize max dimension to {}'.format(opt.resize[0]))
        elif len(opt.resize) == 1:
            print('Will not resize images')
        else:
            raise ValueError('Cannot specify more than two integers for --resize')

        datadir = opt.datadir
        output_dir = os.path.join(opt.basedir, opt.expname, 'matches')
        left_dir = os.path.join(datadir, 'l', 'rectify')
        right_dir = os.path.join(datadir, 'r', 'rectify')
        mask_l_dir = os.path.join(datadir, 'mask_l')
        mask_r_dir = os.path.join(datadir, 'mask_r')
        img_list = natsorted(os.listdir(left_dir))
        mask_list = natsorted(f for f in os.listdir(mask_l_dir) if f.endswith('.png'))
        size0 = cv2.imread(os.path.join(left_dir, img_list[0])).shape[:2]
        # N_imgs = len(img_list)

        # Load the SuperPoint and SuperGlue models.
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('Running match on device \"{}\"'.format(device))
        config = {
            'superpoint': {
                'nms_radius': 4,  # SuperPoint Non Maximum Suppression (NMS) radius (Must be positive)
                'keypoint_threshold': 0.005,  # SuperPoint keypoint detector confidence threshold
                'max_keypoints': -1  # Maximum number of keypoints detected by Superpoint (\'-1\' keeps all keypoints)
            },
            'superglue': {
                'weights': 'indoor',  # SuperGlue weights
                'sinkhorn_iterations': 20,  # Number of Sinkhorn iterations performed by SuperGlue
                'match_threshold': 0.2,  # SuperGlue match threshold
            }
        }
        matching = Matching(config).eval().to(device)

        # Create the output directories if they do not exist already.

        print('Looking for data in directory \"{}\"'.format(datadir))

        print('Will write matches to directory \"{}\"'.format(output_dir))

        if opt.viz:
            print('Will write visualization images to',
                  'directory \"{}\"'.format(output_dir))

        # check previous results
        matches_Nresults = len(os.listdir(output_dir))
        if matches_Nresults >= 2 * len(img_list) and load_pre:
            print("load previous match result from " + output_dir)
            if matches_Nresults == len(img_list) * 3:
                print("load previous sparse disparity maps from " + output_dir)
                cal_mean_disp(output_dir, opt.resize, size0, img_list)
                return
            else:
                sparse_disp(output_dir, datadir, opt.resize, size0, img_list)
                cal_mean_disp(output_dir, opt.resize, size0, img_list)
                return

        timer = AverageTimer(newline=True)

        for i, img_name in enumerate(img_list):
            name0 = name1 = img_name
            mask_name0 = mask_name1 = mask_list[i]
            matches_path = os.path.join(output_dir, name0[:-4] + '_matches.npz')
            viz_path = os.path.join(output_dir, name0[:-4] + '_matches.png')

            do_match = True
            do_viz = opt.viz

            if not (do_match or do_viz):
                timer.print('Finished pair {:5} of {:5}'.format(i, len(img_list)))
                continue

            # If a rotation integer is provided (e.g. from EXIF data), use it:
            rot0, rot1 = 0, 0

            # Load the image pair.
            image0, inp0, scales0 = read_image(
                os.path.join(left_dir, name0), device, opt.resize, rot0, resize_float=False,
                mask_path=os.path.join(mask_l_dir, mask_name0))
            image1, inp1, scales1 = read_image(
                os.path.join(right_dir, name1), device, opt.resize, rot1, resize_float=False,
                mask_path=os.path.join(mask_r_dir, mask_name1))
            if image0 is None or image1 is None:
                print('Problem reading image pair: {} {}'.format(
                    os.path.join(left_dir, name0), os.path.join(right_dir, name1)))
                exit(1)
            timer.update('load_image')

            if do_match:
                # Perform the matching.
                pred = matching({'image0': inp0, 'image1': inp1})
                pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
                kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
                matches, conf = pred['matches0'], pred['matching_scores0']
                timer.update('matcher')

                # Write the matches to disk.
                out_matches = {'keypoints0': kpts0, 'keypoints1': kpts1,
                               'matches': matches, 'match_confidence': conf}
                np.savez(str(matches_path), **out_matches)

            # Keep the matching keypoints.
            valid = matches > -1
            mkpts0 = kpts0[valid]
            mkpts1 = kpts1[matches[valid]]
            mconf = conf[valid]

            if do_viz:
                # Visualize the matches.
                color = cm.jet(mconf)
                text = [
                    'SuperGlue',
                    'Keypoints: {}:{}'.format(len(kpts0), len(kpts1)),
                    'Matches: {}'.format(len(mkpts0)),
                ]
                if rot0 != 0 or rot1 != 0:
                    text.append('Rotation: {}:{}'.format(rot0, rot1))

                # Display extra parameter info.
                k_thresh = matching.superpoint.config['keypoint_threshold']
                m_thresh = matching.superglue.config['match_threshold']
                small_text = [
                    'Keypoint Threshold: {:.4f}'.format(k_thresh),
                    'Match Threshold: {:.2f}'.format(m_thresh),
                    'Image Pair: {}:{}'.format(i + 1, i + 1),
                ]

                make_matching_plot(
                    image0, image1, kpts0, kpts1, mkpts0, mkpts1, color,
                    text, viz_path, show_keypoints=False,
                    fast_viz=False, opencv_display=False, opencv_title='Matches', small_text=small_text)

                timer.update('viz_match')

            timer.print('Finished pair {:5} of {:5}'.format(i + 1, len(img_list)))

        print("computing disparity maps from match results ...")

        sparse_disp(output_dir, datadir, opt.resize, size0, img_list)
        # disp_supervise(output_dir, datadir, opt.resize, size0, img_list)
        cal_mean_disp(output_dir, opt.resize, size0, img_list)
        print("disparity maps saved in {} ...".format(output_dir))  # logs/exp/matches/xx_disp.png


def vis_sparse_pc(rgb_path=None, disp_path=None, fx=None, fy=None, cx=None, cy=None, baseline=None):
    factor = 5
    import open3d as o3d
    if rgb_path is None or disp_path is None:
        raise RuntimeError(f'need to input rgb_path and depth_path')

    print("Read Redwood dataset")
    color_raw = o3d.io.read_image(rgb_path)
    delta_d = 0
    t = rgb_path.split('/')[-1][:-4]
    if len(t.split('_')) > 1:
        delta_d = int(t.split('_')[-1])

    disparity_map = cv2.imread(disp_path, 0)
    depth_map = np.zeros_like(disparity_map)
    depth_map = ((fx/factor * baseline) / (disparity_map + delta_d)).astype(np.uint8)
    depth_map[disparity_map == 0] = 512
    # Create Open3D depth image
    depth_image = o3d.geometry.Image(depth_map)

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_raw, depth_image)
    # print(rgbd_image)

    # plt.subplot(1, 2, 1)
    # plt.title('grayscale image')
    # plt.imshow(rgbd_image.color)
    # plt.subplot(1, 2, 2)
    # plt.title('depth image')
    # plt.imshow(rgbd_image.depth)
    # plt.show()
    Cam_Intrinsic = o3d.camera.PinholeCameraIntrinsic()
    Cam_Intrinsic.set_intrinsics(
        width=depth_map.shape[1], height=depth_map.shape[0], fx=fx, fy=fy, cx=cx, cy=cy)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image, Cam_Intrinsic)

    # Flip it, otherwise the pointcloud will be upside down
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    # o3d.io.write_point_cloud("test.pcd", pcd)

    o3d.visualization.draw_geometries([pcd])


if __name__ == '__main__':
    # parser = config_parser()
    # args = parser.parse_args()
    #
    # run_spsg_match(args)

    vis_sparse_pc('../heart/l/rectify/1_655.jpg', '../logs/heart/matches/1_sparse_disp.png',
                  fx=942.05, fy=942.05, cx=181.44, cy=156.68, baseline=435)

    # f5:
    # fx = 391.656525, fy = 426.835144, cx = 165.964371, cy = 154.498138, baseline = 5
    # data:
    # fx = 942.05, fy = 942.05, cx = 181.44, cy = 156.68, baseline = 435

