import cv2
import numpy as np
import torch
import os, imageio

from natsort import natsorted

from utils.io_utils import load_img_list


# ######### Slightly modified version of LLFF data loading code
# #########  see https://github.com/Fyusion/LLFF for original

def _minify(basedir, l_r='l', factor=None):
    sfx = '_{}'.format(factor)
    imgdir = os.path.join(basedir, l_r, 'images' + sfx)
    if not os.path.exists(imgdir):
        print(imgdir, 'does not exist')
        os.makedirs(imgdir)

    with open(os.path.join(basedir, 'train.txt'), 'r') as f_list:
        lines = f_list.readlines()
    if os.path.exists(os.path.join(basedir, 'test.txt')):
        with open(os.path.join(basedir, 'test.txt'), 'r') as f_list:
            lines += f_list.readlines()

    img0names = natsorted(os.listdir(os.path.join(basedir, l_r, 'images')))
    imgfiles = [os.path.join(imgdir, f.strip().replace('.jpg', '.png')) for f in lines]
    files = os.listdir(imgdir)
    if not len(files) == len(img0names):
        print(imgdir, 'does not have correct images')
        img0files = [os.path.join(basedir, l_r, 'images', f) for f in img0names \
                     if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
        for i, img0file in enumerate(img0files):
            src0 = cv2.imread(img0file)
            src = cv2.resize(src0, (src0.shape[1] // factor, src0.shape[0] // factor))
            cv2.imwrite(imgfiles[i], src)
    return imgfiles


def _cal_right_pose(poses0, R, t, dist=None):
    poses0 = poses0.transpose([2, 0, 1])  # (n, 3, 5)

    dist_poses = np.linalg.norm(poses0[0, :, 3] - poses0[-1, :, 3])

    R_list = R.split(',')
    t_list = t.split(',')
    R_mat = np.mat([[float(R_list[0]), float(R_list[1]), float(R_list[2])],
                    [float(R_list[3]), float(R_list[4]), float(R_list[5])],
                    [float(R_list[6]), float(R_list[7]), float(R_list[8])]])
    t_vec = np.mat([[float(t_list[0]), float(t_list[1]), float(t_list[2])]])
    if dist is not None:
        t_vec = t_vec * (dist_poses / dist)
    bottom = np.mat([[0., 0., 0., 1.]])
    Rt_21 = np.concatenate((R_mat.transpose(), t_vec.transpose()), axis=1)  # (3, 4) [R', t]
    M_21 = np.concatenate((Rt_21, bottom), axis=0)
    poses1 = []
    for i in range(poses0.shape[0]):
        pose0 = poses0[i, :, :]  # (3, 5)
        Rt_1w = np.mat(pose0[:, :4])
        hwf = pose0[:, 4:]
        M_1w = np.concatenate((Rt_1w, bottom), axis=0)  # (4, 4)
        M_2w = M_21 * M_1w
        pose1 = np.concatenate((np.array(M_2w)[:3, :], hwf), axis=1)
        poses1.append(pose1)

    return np.stack(poses1)


def _load_data(basedir, factor=None, width=None, height=None, load_imgs=True, R_mat=None, t_vec=None, dist=None):

    poses_arr0 = np.load(os.path.join(basedir, 'poses_bounds.npy'))

    image_list = load_img_list(basedir, load_test=True)
    list_poses = []
    for idx in range(poses_arr0.shape[0]):
        list_poses.append(poses_arr0[idx, :])
    t_zip = zip(list_poses, image_list)
    sorted_t_zip = natsorted(t_zip, key=lambda x: x[1])
    sorted_t = zip(*sorted_t_zip)
    list_poses, image_list = [list(x) for x in sorted_t]
    poses_arr0 = np.stack(list_poses)

    poses0 = poses_arr0[:, :-2].reshape([-1, 3, 5]).transpose([1, 2, 0])  # (3, 5, n)
    bds0 = poses_arr0[:, -2:].transpose([1, 0])     # (2, n)
    bds1 = bds0
    bds = np.concatenate((bds0, bds1), axis=-1)

    if R_mat is not None and t_vec is not None:
        pose1 = _cal_right_pose(poses0, R_mat, t_vec, dist)  # (n, 3, 5)
        pose1 = pose1.transpose([1, 2, 0])  # (3, 5, n)
        poses = np.concatenate((poses0, pose1), axis=-1)  # (3, 5, 2n)
    else:
        raise ValueError(f"need to input R and t for right poses")

    img0 = [os.path.join(basedir, 'l', 'images', f) for f in sorted(os.listdir(os.path.join(basedir, 'l', 'images'))) \
            if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')][0]
    sh = imageio.imread(img0).shape

    _factor = 1
    if factor is not None:
        _factor = factor
    elif height is not None:
        _factor = sh[0] / float(height)
    elif width is not None:
        _factor = sh[1] / float(width)

    imgfiles0 = natsorted(os.listdir(os.path.join(basedir, 'l', 'images')))
    imgfiles1 = natsorted(os.listdir(os.path.join(basedir, 'r', 'images')))

    if _factor != 1:
        imgfiles0 = _minify(basedir, 'l', _factor)
        imgfiles1 = _minify(basedir, 'r', _factor)

    if poses.shape[-1] != len(imgfiles0) + len(imgfiles1):
        print('Mismatch between imgs {} and poses {} !!!!'.format(len(imgfiles0), poses.shape[-1]))
        return

    # update the hwf in poses
    sh = cv2.imread(imgfiles0[0]).shape
    poses[:2, 4, :] = np.array(sh[:2]).reshape([2, 1])  # h, w
    poses[2, 4, :] = poses[2, 4, :] * 1. / _factor  # f

    if not load_imgs:
        return poses, bds

    def imread(f):
        if f.endswith('png'):
            return imageio.imread(f, ignoregamma=True)
        else:
            return imageio.imread(f)

    rgb_imgs0 = [imread(f)[..., :3] / 255. for f in imgfiles0]
    rgb_imgs0 = np.stack(rgb_imgs0, -1)  # (h, w, 3, n)
    rgb_imgs1 = [imread(f)[..., :3] / 255. for f in imgfiles1]
    rgb_imgs1 = np.stack(rgb_imgs1, -1)
    rgb_imgs = np.concatenate((rgb_imgs0, rgb_imgs1), axis=-1)

    print('Loaded image data', rgb_imgs.shape, poses[:, -1, 0])
    return poses, bds, rgb_imgs


def normalize(x):
    return x / np.linalg.norm(x)


def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m


def ptstocam(pts, c2w):
    tt = np.matmul(c2w[:3, :3].T, (pts - c2w[:3, 3])[..., np.newaxis])[..., 0]
    return tt


def poses_avg(poses):
    hwf = poses[0, :3, -1:]

    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)

    return c2w


def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, rots, N):
    render_poses = []
    rads = np.array(list(rads) + [1.])
    hwf = c2w[:, 4:5]

    for theta in np.linspace(0., 2. * np.pi * rots, N + 1)[:-1]:
        c = np.dot(c2w[:3, :4], np.array([np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.]) * rads)
        z = normalize(c - np.dot(c2w[:3, :4], np.array([0, 0, -focal, 1.])))
        render_poses.append(np.concatenate([viewmatrix(z, up, c), hwf], 1))
    return render_poses


def render_path_fixed(c2w, N):
    render_poses = []
    hwf = c2w[:, 4:5]

    for i in range(N):
        eye_pose = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
        render_poses.append(np.concatenate([eye_pose, hwf], 1))

    return render_poses


def render_path_zoom(c2w, up, delta, N):
    render_poses = []
    hwf = c2w[:, 4:5]
    z_d = c2w[:3, 2]
    c_o = c2w[:3, 3]

    half_N = (N + 1) // 2

    for t in np.linspace(0., delta, half_N):
        c = c_o - t * z_d

        render_poses.append(np.concatenate([viewmatrix(z_d, up, c), hwf], 1))

    for t in np.linspace(0., delta, N + 1 - half_N):
        c = c_o - (delta - t) * z_d

        render_poses.append(np.concatenate([viewmatrix(z_d, up, c), hwf], 1))

    return render_poses


def recenter_poses(poses):
    poses_ = poses + 0
    bottom = np.reshape([0, 0, 0, 1.], [1, 4])
    c2w = poses_avg(poses)
    c2w = np.concatenate([c2w[:3, :4], bottom], -2)
    bottom = np.tile(np.reshape(bottom, [1, 1, 4]), [poses.shape[0], 1, 1])
    poses = np.concatenate([poses[:, :3, :4], bottom], -2)

    poses = np.linalg.inv(c2w) @ poses
    poses_[:, :3, :4] = poses[:, :3, :4]
    poses = poses_
    return poses


#####################


def spherify_poses(poses, bds):
    p34_to_44 = lambda p: np.concatenate([p, np.tile(np.reshape(np.eye(4)[-1, :], [1, 1, 4]), [p.shape[0], 1, 1])], 1)

    rays_d = poses[:, :3, 2:3]
    rays_o = poses[:, :3, 3:4]

    def min_line_dist(rays_o, rays_d):
        A_i = np.eye(3) - rays_d * np.transpose(rays_d, [0, 2, 1])
        b_i = -A_i @ rays_o
        pt_mindist = np.squeeze(-np.linalg.inv((np.transpose(A_i, [0, 2, 1]) @ A_i).mean(0)) @ (b_i).mean(0))
        return pt_mindist

    pt_mindist = min_line_dist(rays_o, rays_d)

    center = pt_mindist
    up = (poses[:, :3, 3] - center).mean(0)

    vec0 = normalize(up)
    vec1 = normalize(np.cross([.1, .2, .3], vec0))
    vec2 = normalize(np.cross(vec0, vec1))
    pos = center
    c2w = np.stack([vec1, vec2, vec0, pos], 1)

    poses_reset = np.linalg.inv(p34_to_44(c2w[None])) @ p34_to_44(poses[:, :3, :4])

    rad = np.sqrt(np.mean(np.sum(np.square(poses_reset[:, :3, 3]), -1)))

    sc = 1. / rad
    poses_reset[:, :3, 3] *= sc
    bds *= sc
    rad *= sc

    centroid = np.mean(poses_reset[:, :3, 3], 0)
    zh = centroid[2]
    radcircle = np.sqrt(rad ** 2 - zh ** 2)
    new_poses = []

    for th in np.linspace(0., 2. * np.pi, 120):
        camorigin = np.array([radcircle * np.cos(th), radcircle * np.sin(th), zh])
        up = np.array([0, 0, -1.])

        vec2 = normalize(camorigin)
        vec0 = normalize(np.cross(vec2, up))
        vec1 = normalize(np.cross(vec2, vec0))
        pos = camorigin
        p = np.stack([vec0, vec1, vec2, pos], 1)

        new_poses.append(p)

    new_poses = np.stack(new_poses, 0)

    new_poses = np.concatenate([new_poses, np.broadcast_to(poses[0, :3, -1:], new_poses[:, :3, -1:].shape)], -1)
    poses_reset = np.concatenate(
        [poses_reset[:, :3, :4], np.broadcast_to(poses[0, :3, -1:], poses_reset[:, :3, -1:].shape)], -1)

    return poses_reset, new_poses, bds


def load_llff_data(basedir, factor=8, recenter=True, bd_factor=.75, spherify=False, path_zflat=False,
                   render_path='spiral', use_depth=False, R_mat=None, t_vec=None, dist=None):
    # No downsampling
    if factor == 1:
        factor = None

    poses, bds, imgs = _load_data(basedir, factor=factor, R_mat=R_mat, t_vec=t_vec, dist=dist)  # factor=8 downsamples original imgs by 8x
    print('Loaded', basedir, bds.min(), bds.max())

    # Correct rotation matrix ordering and move variable dim to axis 0
    poses = np.concatenate([poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1)
    poses = np.moveaxis(poses, -1, 0).astype(np.float32)
    images = np.moveaxis(imgs, -1, 0).astype(np.float32)
    bds = np.moveaxis(bds, -1, 0).astype(np.float32)

    # Rescale if bd_factor is provided
    sc = 1. if bd_factor is None else 1. / (bds.min() * bd_factor)
    poses[:, :3, 3] *= sc
    bds *= sc

    if recenter:
        poses = recenter_poses(poses)

    if spherify:
        poses, render_poses, bds = spherify_poses(poses, bds)

    else:
        c2w = poses_avg(poses)

        print('recentered', c2w.shape)
        print(c2w[:3, :4])

        ## Get spiral
        # Get average pose
        up = normalize(poses[:, :3, 1].sum(0))

        # Find a reasonable "focus depth" for this dataset
        close_depth, inf_depth = bds.min() * .9, bds.max() * 5.
        dt = .75
        mean_dz = 1. / (((1. - dt) / (close_depth + 1e-6) + dt / (inf_depth + 1e-6)) + 1e-6)
        focal = mean_dz

        # Get radii for spiral path
        shrink_factor = .8
        zdelta = close_depth * .2
        tt = poses[:, :3, 3]  # ptstocam(poses[:3,3,:].T, c2w).T
        rads = np.percentile(np.abs(tt), 90, 0)
        c2w_path = c2w
        N_views = 120
        N_rots = 2
        if path_zflat:
            #             zloc = np.percentile(tt, 10, 0)[2]
            zloc = -close_depth * .1
            c2w_path[:3, 3] = c2w_path[:3, 3] + zloc * c2w_path[:3, 2]
            rads[2] = 0.
            N_rots = 1
            N_views /= 2

        if render_path == 'spiral':
            # Generate poses for spiral path
            render_poses = render_path_spiral(c2w_path, up, rads, focal, zdelta, zrate=.5, rots=N_rots, N=N_views)
            print('Render poses: spiral')
        elif render_path == 'fixidentity':
            # Generate poses for fixed path
            render_poses = render_path_fixed(c2w_path, N_views)
            print('Render poses: fix identity')
        elif render_path == 'zoom':
            zoom_dist = rads[2]

            # Generate poses for zoom path
            render_poses = render_path_zoom(c2w_path, up, zoom_dist, N_views)
            print('Render poses: zoom')

    render_poses = np.array(render_poses).astype(np.float32)

    c2w = poses_avg(poses)
    print('Data:')
    print('poses.shape: ', poses.shape, 'images.shape: ', images.shape, 'bds.shape', bds.shape)

    dists = np.sum(np.square(c2w[:3, 3] - poses[:, :3, 3]), -1)

    images = images.astype(np.float32)
    poses = poses.astype(np.float32)

    image_list = natsorted(load_img_list(basedir, load_test=True))
    i_train = []
    i_test = []
    with open(os.path.join(basedir, 'train.txt'), 'r') as f_list:
        lines = f_list.readlines()
        lines = [line.strip("\n") for line in lines]
        s = 0
        for line in lines:
            for id in range(s, len(image_list)):
                if line == image_list[id]:
                    i_train.append(id)
                    s = id
                    break
                if id == len(image_list) - 1:
                    raise RuntimeError(f'wrong in train.txt')
        # i_train = list(np.arange(len(lines)))

    if os.path.exists(os.path.join(basedir, 'test.txt')):
        with open(os.path.join(basedir, 'test.txt'), 'r') as f_list:
            lines = f_list.readlines()
            lines = [line.strip("\n") for line in lines]
            s = 0
            for line in lines:
                for id in range(s, len(image_list)):
                    if line == image_list[id]:
                        i_test.append(id)
                        s = id
                        break
                    if id == len(image_list) - 1:
                        raise RuntimeError(f'wrong in test.txt')
            # i_test = list(np.arange(len(lines)) + len(i_train))
    N_img = poses.shape[0] // 2
    i_train = i_train + [x + N_img for x in i_train]
    i_test = i_test + [x + N_img for x in i_test]
    times = np.linspace(0., 1., poses.shape[0]//2)
    times = np.concatenate((times, times))
    render_times = torch.linspace(0., 1., render_poses.shape[0])

    return images, poses, bds, times, render_poses, render_times, i_train, i_test, sc
