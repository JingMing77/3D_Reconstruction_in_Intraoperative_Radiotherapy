import sys

sys.path.append('..')
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange

from models.nerf.run_nerf_helpers import *
from options import config_parser
from .load_stereo_llff import load_llff_data
from utils.io_utils import *
from utils.nerf_utils import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
DEBUG = False


def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn

    def ret(inputs):
        return torch.cat([fn(inputs[i:i + chunk]) for i in range(0, inputs.shape[0], chunk)], 0)

    return ret


def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024 * 64):
    """Prepares inputs and applies network 'fn'.
    """
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    embedded = embed_fn(inputs_flat)

    if viewdirs is not None:
        input_dirs = viewdirs[:, None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat([embedded, embedded_dirs], -1)

    outputs_flat = batchify(fn, netchunk)(embedded)
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs


def batchify_rays(rays_flat, chunk=1024 * 32, depth_priors=None, depth_confidences=None, **kwargs):
    """Render rays in smaller minibatches to avoid OOM.
    """
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(rays_flat[i:i + chunk], depth_priors=depth_priors[i:i + chunk],
                          depth_confidences=depth_confidences[i:i + chunk], **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k: torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret


def render(H, W, focal, chunk=1024 * 32, rays=None, c2w=None, ndc=True,
           near=0., far=1.,
           use_viewdirs=False, c2w_staticcam=None, depth_priors=None, depth_confidences=None,
           **kwargs):
    """Render rays
    Args:
      H: int. Height of image in pixels.
      W: int. Width of image in pixels.
      focal: float. Focal length of pinhole camera.
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for
        each example in batch.
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
      ndc: bool. If True, represent ray origin, direction in NDC coordinates.
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
      use_viewdirs: bool. If True, use viewing direction of a point in space in model.
      c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for
       camera while using other c2w argument for viewing directions.
    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned by render_rays().
    """
    if c2w is not None:
        # special case to render full image
        rays_o, rays_d = get_rays(H, W, focal, c2w)
    else:
        # use provided ray batch
        rays_o, rays_d = rays

    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays(H, W, focal, c2w_staticcam)
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1, 3]).float()

    sh = rays_d.shape  # [..., 3]
    if ndc:
        # for forward facing scenes
        rays_o_ori = rays_o.clone()
        rays_d_ori = rays_d.clone()
        rays_o, rays_d = ndc_rays(H, W, focal, 1., rays_o, rays_d)

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1, 3]).float()
    rays_d = torch.reshape(rays_d, [-1, 3]).float()
    depth_priors = torch.reshape(depth_priors, [-1]).float()
    depth_confidences = torch.reshape(depth_confidences, [-1]).float()

    near, far = near * torch.ones_like(rays_d[..., :1]), far * torch.ones_like(rays_d[..., :1])
    rays = torch.cat([rays_o, rays_d, near, far], -1)
    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)

    # Render and reshape
    all_ret = batchify_rays(rays, chunk, depth_priors, depth_confidences, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    # if ndc:
    #     all_ret['depth_map'] = -1 / rays_d_ori[..., 2] * (1 / (1 - all_ret['depth_map']) + rays_o_ori[..., 2])

    k_extract = ['rgb_map', 'disp_map', 'acc_map', 'depth_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k: all_ret[k] for k in all_ret if k not in k_extract}

    return ret_list + [ret_dict]


def render_path(render_poses, hwf, chunk, render_kwargs, image_list, sc=1.,
                depth_priors=None, depth_confidences=None, savedir=None, render_factor=0):
    H, W, focal = hwf

    if render_factor != 0:
        # Render downsampled for speed
        H = H // render_factor
        W = W // render_factor
        focal = focal / render_factor

    rgbs = []
    disps = []
    depths = []

    for i, c2w in enumerate(tqdm(render_poses)):
        rgb, disp, acc, depth, _ = render(H, W, focal, depth_priors=depth_priors[i],
                                          depth_confidences=depth_confidences[i], chunk=chunk, c2w=c2w[:3, :4],
                                          **render_kwargs)
        rgbs.append(rgb.cpu().numpy())
        disps.append(disp.cpu().numpy())
        depths.append(depth)
        if i == 0:
            print(rgb.shape, disp.shape)

        if savedir is not None:
            depth_ratio = depth / depth_priors[i]
            min_r, max_r = 0.5, 1.5
            invalid_mask = torch.logical_or((depth_ratio < min_r), (depth_ratio > max_r))
            invalid_mask = invalid_mask.cpu().numpy()
            coords = np.transpose(np.where(invalid_mask == 1))  # [n, 2]
            for coord in coords:
                d_block = depth_priors[i][max(coord[0] - 5, 0): min(coord[0] + 5, W - 1),
                          max(coord[1] - 5, 0): min(coord[1] + 5, W - 1)]
                depth[coord[0], coord[1]] = torch.median(d_block)

            rgb8 = to8b(rgbs[-1])
            if i >= len(image_list):
                frame_id = int(image_list[i - len(image_list)].split('.')[0]) + len(image_list)
                frame_id = str(frame_id)
            else:
                frame_id = image_list[i].split('.')[0]
            filename = os.path.join(savedir, '{}.png'.format(frame_id))
            imageio.imwrite(filename, rgb8)
            filename = os.path.join(savedir, '{}_depth.npy'.format(frame_id))
            np.save(filename, depth.cpu().numpy() / sc)
            disp_visual = visualize_depth(depth.cpu().numpy())
            filename = os.path.join(savedir, '{}_depth.png'.format(frame_id))
            cv2.imwrite(filename, disp_visual)

    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)
    depths = torch.stack(depths, 0)

    return rgbs, disps, depths.cpu().numpy()


def create_nerf(args):
    """Instantiate NeRF's MLP model.
    """
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)

    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)
    output_ch = 4
    skips = [4]
    model = NeRF(D=args.netdepth, W=args.netwidth,
                 input_ch=input_ch, output_ch=output_ch, skips=skips,
                 input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
    grad_vars = list(model.parameters())

    network_query_fn = lambda inputs, viewdirs, network_fn: run_network(inputs, viewdirs, network_fn,
                                                                        embed_fn=embed_fn,
                                                                        embeddirs_fn=embeddirs_fn,
                                                                        netchunk=args.netchunk)

    # Create optimizer
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    start = 0
    basedir = args.basedir
    expname = args.expname

    ##########################

    # Load checkpoints
    if args.ft_path is not None and args.ft_path != 'None':
        ckpts = [args.ft_path]
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        # start = 0
        start = ckpt['global_step']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (start / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate

        # Load model
        model.load_state_dict(ckpt['network_fn_state_dict'])

    else:
        ckpt_path = os.path.join(basedir, expname, 'nerf', 'checkpoints')
        ckpts = [os.path.join(ckpt_path, f) for f in sorted(os.listdir(ckpt_path)) if 'tar' in f]

        print('Found ckpts', ckpts)
        if len(ckpts) > 0 and not args.no_reload:
            ckpt_path = ckpts[-1]
            print('Reloading from', ckpt_path)
            ckpt = torch.load(ckpt_path)

            start = ckpt['global_step']
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])

            # Load model
            model.load_state_dict(ckpt['network_fn_state_dict'])

    ##########################

    render_kwargs_train = {
        'network_query_fn': network_query_fn,
        'perturb': args.perturb,
        'N_samples': args.N_samples,
        'network_fn': model,
        'use_viewdirs': args.use_viewdirs,
        'white_bkgd': args.white_bkgd,
        'raw_noise_std': args.raw_noise_std,
        'near_bound': args.near,
        'far_bound': args.far,
    }

    # NDC only good for LLFF-style forward facing data
    if args.no_ndc:
        print('Not ndc!')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp

    render_kwargs_test = {k: render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer


def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1. - torch.exp(-act_fn(raw) * dists)

    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[..., :1].shape)], -1)  # [N_rays, N_samples]

    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

    rgb = torch.sigmoid(raw[..., :3])  # [N_rays, N_samples, 3]
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[..., 3].shape) * raw_noise_std

        # Overwrite randomly sampled data if pytest
        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(raw[..., 3].shape)) * raw_noise_std
            noise = torch.Tensor(noise)

    alpha = raw2alpha(raw[..., 3] + noise, dists)  # [N_rays, N_samples]
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1. - alpha + 1e-10], -1), -1)[:, :-1]
    rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]

    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1. / torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)
    if white_bkgd:
        rgb_map = rgb_map + (1. - acc_map[..., None])

    return rgb_map, disp_map, acc_map, weights, depth_map, rgb


def render_rays(ray_batch,
                network_fn,
                network_query_fn,
                N_samples,
                depth_priors,
                depth_confidences,
                retraw=False,
                lindisp=False,
                perturb=0.,
                white_bkgd=False,
                raw_noise_std=0.,
                pytest=False,
                near_bound=None,
                far_bound=None,
                no_depth_sampling=False,
                depth_sampling_clamp=False):
    """Volumetric rendering.
    Args:
      ray_batch: array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction.
      network_fn: function. Model for predicting RGB and density at each point
        in space.
      network_query_fn: function used for passing queries to network_fn.
      N_samples: int. Number of different times to sample along each ray.
      retraw: bool. If True, include model's raw, unprocessed predictions.
      lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
      perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
        random points in time.
      white_bkgd: bool. If True, assume a white background.
      raw_noise_std: ...
    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
      disp_map: [num_rays]. Disparity map. 1 / depth.
      acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
      raw: [num_rays, num_samples, 4]. Raw predictions from model.
      rgb0: See rgb_map. Output for coarse model.
      disp0: See disp_map. Output for coarse model.
      acc0: See acc_map. Output for coarse model.
      z_std: [num_rays]. Standard deviation of distances along ray for each
        sample.
    """
    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]  # [N_rays, 3] each
    viewdirs = ray_batch[:, -3:] if ray_batch.shape[-1] > 8 else None
    bounds = torch.reshape(ray_batch[..., 6:8], [-1, 1, 2])
    if no_depth_sampling:
        near = 2.0 * torch.ones(size=(N_rays, 1))
        far = 6.0 * torch.ones(size=(N_rays, 1))
        t_vals = torch.linspace(0., 1., steps=N_samples)
        if not lindisp:
            z_vals = near * (1. - t_vals) + far * t_vals
        else:
            z_vals = 1. / (1. / near * (1. - t_vals) + 1. / far * t_vals)

    elif not no_depth_sampling and not depth_sampling_clamp:  # sample according to Normal Distribution: N(depth, depth_conf^2)
        near = (depth_priors * (1 - torch.clamp(depth_confidences, min=near_bound, max=far_bound))).unsqueeze(
            1)  # [near_bound, far_bound]
        far = (depth_priors * (1 + torch.clamp(depth_confidences, min=near_bound, max=far_bound))).unsqueeze(1)

        depth_c = (depth_confidences - torch.min(depth_confidences)) / (
                torch.max(depth_confidences) - torch.min(depth_confidences))
        depth_c = (depth_c / 10.) + 0.01  # [0.01, 0.11]
        mean = depth_priors.unsqueeze(1).expand([N_rays, N_samples])
        std = depth_c.unsqueeze(1).expand([N_rays, N_samples])
        z_vals, _ = torch.sort(torch.normal(mean, std), dim=1)
        min_val, _ = torch.min(z_vals, dim=1)
        max_val, _ = torch.max(z_vals, dim=1)
        min_val, max_val = min_val.unsqueeze(1), max_val.unsqueeze(1)
        scale = (far - near) / (max_val - min_val)
        z_vals = scale * (z_vals - min_val) + near  # limited to [near, far]
    else:
        near = (depth_priors * (1 - torch.clamp(depth_confidences, min=near_bound, max=far_bound))).unsqueeze(1)
        far = (depth_priors * (1 + torch.clamp(depth_confidences, min=near_bound, max=far_bound))).unsqueeze(1)
        t_vals = torch.linspace(0., 1., steps=N_samples)
        if not lindisp:
            z_vals = near * (1. - t_vals) + far * t_vals
        else:
            z_vals = 1. / (1. / near * (1. - t_vals) + 1. / far * t_vals)

    z_vals = z_vals.expand([N_rays, N_samples])

    if perturb > 0.:
        # get intervals between samples
        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], -1)
        lower = torch.cat([z_vals[..., :1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape)

        # Pytest, overwrite u with numpy's fixed random numbers
        if pytest:
            np.random.seed(0)
            t_rand = np.random.rand(*list(z_vals.shape))
            t_rand = torch.Tensor(t_rand)

        z_vals = lower + (upper - lower) * t_rand

    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # [N_rays, N_samples, 3]

    #     raw = run_network(pts)
    raw = network_query_fn(pts, viewdirs, network_fn)
    rgb_map, disp_map, acc_map, weights, depth_map, rgb = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd,
                                                                      pytest=pytest)

    ret = {'rgb_map': rgb_map, 'disp_map': disp_map, 'acc_map': acc_map, 'depth_map': depth_map, "weights": weights}
    if retraw:
        ret['raw'] = raw

    for k in ret:
        if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
            print(f"! [Numerical Error] {k} contains nan or inf.")

    return ret


def train(args):
    print("---------------------------------------------------------------------------------------------")
    print('Nerf begins !')
    # Load data
    images, poses, bds, times, render_poses, render_times, i_train, i_test, sc = load_llff_data(
        args.datadir, args.factor, recenter=True, bd_factor=.75, spherify=args.spherify,
        R_mat=args.R_matrix, t_vec=args.t_vec, dist=args.cam_dist)

    # load 'depth_priors'
    N_images, H, W, _ = images.shape
    load_test = (len(i_test) > 0)
    image_list = natsorted(load_img_list(args.datadir, load_test=load_test))
    image_list_train = load_img_list(args.datadir, load_test=False)
    fxl = float(args.cam_l.split(',')[0])
    fxr = float(args.cam_r.split(',')[0])
    depth_priors = load_depths(image_list,
                               os.path.join(args.basedir, args.expname, 'depth_priors', 'results'),
                               H, W, is_disp=False, load_right=True)  # (train_n, h, w)

    colmap_depths, colmap_masks = load_colmap(image_list, args.datadir, H, W,
                                              doAugment=True, logdir=os.path.join(args.basedir, args.expname),
                                              focal_x=float(args.cam_l.split(',')[0]), baseline=args.baseline,
                                              depth_gt_mean=args.depth_gt_mean)

    depth_priors = align_scales(depth_priors, colmap_depths, colmap_masks,
                                poses, sc, i_train, i_test)

    # depth_confidences
    hwf = poses[0, :3, -1]
    poses = poses[:, :3, :4]
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]
    print('Loaded llff', images.shape, render_poses.shape, hwf, args.datadir)

    poses_tensor = torch.from_numpy(poses).to(device)
    K = torch.FloatTensor([[focal, 0, -W / 2.0, 0],
                           [0, -focal, -H / 2.0, 0],
                           [0, 0, -1, 0],
                           [0, 0, 0, 1]]).to(device)
    if poses_tensor.shape[1] == 3:
        bottom = torch.FloatTensor([0, 0, 0, 1]).unsqueeze(0)
        bottom = bottom.repeat(poses_tensor.shape[0], 1, 1).to(poses_tensor.device)
        T = torch.cat([poses_tensor, bottom], 1)
    else:
        T = poses_tensor.clone()

    depth_confidences0 = cal_depth_confidences(torch.from_numpy(depth_priors[: N_images // 2, ...]).to(device),
                                               T[: N_images // 2, ...], K, i_train[: len(i_train) // 2], args.topk,
                                               save_dir=os.path.join(args.basedir, args.expname, 'nerf', 'results'))
    depth_confidences1 = cal_depth_confidences(torch.from_numpy(depth_priors[N_images // 2:, ...]).to(device),
                                               T[N_images // 2:, ...], K, i_train[: len(i_train) // 2], args.topk)
    depth_confidences = np.concatenate((depth_confidences0, depth_confidences1), axis=0)

    # tissue_mask
    tissue_mask1 = load_masks(os.path.join(args.datadir, 'mask_l'), image_list, H, W)
    tissue_mask2 = load_masks(os.path.join(args.datadir, 'mask_r'), image_list, H, W)
    tissue_masks_bin = np.concatenate((tissue_mask1, tissue_mask2))
    tissue_masks = apply_grad_conf(tissue_masks_bin, images, depth_confidences)  # apply gradient and d_conf
    refine_masks_bin = refine_area(tissue_masks_bin, depth_priors, images, save_path='./tmp')  # reflect on tissue

    # for i in range(N_images):
    #     tissue_mask = tissue_masks[i, :, :]
    #     # d_sc = (depth - np.min(depth)) / (np.max(depth) - np.min(depth))
    #     cv2.imshow("1", tissue_mask)
    #     cv2.imshow("img", images[i])
    #     cv2.waitKey(0)
    # cv2.destroyAllWindows()

    if not args.train_binocular:
        images = images[: N_images // 2, ...]
        poses = poses[: N_images // 2, ...]
        poses_tensor = poses_tensor[: N_images // 2, ...]
        bds = bds[: N_images // 2, ...]
        times = times[: N_images // 2, ...]
        depth_priors = depth_priors[: N_images // 2, ...]
        refine_masks_bin = refine_masks_bin[: N_images // 2, ...]
        tissue_masks_bin = tissue_masks_bin[: N_images // 2, ...]
        tissue_masks = tissue_masks[: N_images // 2, ...]
        depth_confidences = depth_confidences[: N_images // 2, ...]
        i_train = i_train[: len(i_train) // 2]
        i_test = i_test[: len(i_test) // 2]

    print('DEFINING BOUNDS')
    if args.no_ndc:
        near = np.ndarray.min(bds) * .9
        far = np.ndarray.max(bds) * 1.
    else:
        near = 0.
        far = 1.
    print('NEAR FAR', near, far)

    # Create nerf model
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args)
    global_step = start

    bds_dict = {
        'near': near,
        'far': far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    save_path = os.path.join(args.basedir, args.expname, 'nerf')

    # Short circuit if only rendering out from trained model
    if args.render_only:
        print('RENDER ONLY')
        with torch.no_grad():
            if args.render_test:
                testsavedir = os.path.join(save_path, 'results',
                                           'renderonly_{}_{:06d}'.format('test', start))
                render_poses = poses_tensor[i_test]
                depth_priors = depth_priors[i_test]
                depth_confidences = depth_confidences[i_test]
                image_list = image_list[i_test]
            else:
                testsavedir = os.path.join(save_path, 'results',
                                           'renderonly_{}_{:06d}'.format('train', start))
                render_poses = poses_tensor[i_train]
                depth_priors = depth_priors[i_train]
                depth_confidences = depth_confidences[i_train]
                image_list = image_list[i_train]

            os.makedirs(testsavedir, exist_ok=True)
            rgbs, disps, depths = render_path(render_poses, hwf, args.chunk, render_kwargs_test, sc=sc,
                                              depth_priors=torch.from_numpy(depth_priors).to(device),
                                              depth_confidences=torch.from_numpy(depth_confidences).to(device),
                                              savedir=testsavedir, render_factor=args.render_factor,
                                              image_list=image_list)
            print('Done rendering', testsavedir)

            return

    # Prepare raybatch tensor if batching random rays
    N_rand = args.N_rand

    # For random ray batching
    print('get rays')
    rays = np.stack([get_rays_np(H, W, focal, p) for p in poses[:, :3, :4]], 0)  # [N, ro+rd, H, W, 3]
    print('done, concats')
    rays_rgb = np.concatenate([rays, images[:, None]], 1)  # [N, ro+rd+rgb, H, W, 3]
    depths_pri = np.stack([depth_priors, depth_confidences.astype(np.float32),
                           np.ones(depth_priors.shape).astype(np.float32)], -1)  # [N, H, W, 3]
    rays_rgb = np.concatenate([rays_rgb, depths_pri[:, None]], 1)
    rays_rgb = np.transpose(rays_rgb, [0, 2, 3, 1, 4])  # [N, H, W, ro+rd+rgb(d)+prior, 3]
    rays_rgb_ori = rays_rgb.copy()

    if args.mask_guide_sample_rate > 1e-16:
        rays_rgb = importance_sampling(rays_rgb_ori, H, W, tissue_masks, device, args.mask_guide_sample_rate)

    rays_rgb = np.stack([rays_rgb[i] for i in i_train], 0)  # train images only
    rays_rgb = np.reshape(rays_rgb, [-1, 4, 3])
    rays_rgb = rays_rgb.astype(np.float32)
    print('shuffle rays')
    np.random.shuffle(rays_rgb)
    print('done')

    # Move training data to GPU
    rays_rgb = torch.Tensor(rays_rgb).to(device)

    print(args.chunk)
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)

    # Summary writers
    writer = SummaryWriter(os.path.join(args.basedir, args.expname, 'nerf', 'summary'))

    i_batch = 0
    N_iters = args.N_iters
    start = start + 1
    for i in trange(start, N_iters):
        time0 = time.time()

        # Random over all images
        batch = rays_rgb[i_batch:i_batch + N_rand]  # [B, 2+1, 3*?]
        batch = torch.transpose(batch, 0, 1)
        batch_rays, target_s = batch[:2], batch[2]
        target_prior = batch[3]
        i_batch += N_rand
        if i_batch >= rays_rgb.shape[0]:
            print("Shuffle data after an epoch!")
            if args.mask_guide_sample_rate > 1e-16:
                rays_rgb = importance_sampling(rays_rgb_ori, H, W, tissue_masks, device, args.mask_guide_sample_rate)
                rays_rgb = np.stack([rays_rgb[i] for i in i_train], 0)
                rays_rgb = np.reshape(rays_rgb, [-1, 4, 3])
                rays_rgb = rays_rgb.astype(np.float32)
                np.random.shuffle(rays_rgb)
                rays_rgb = torch.Tensor(rays_rgb).to(device)
            else:
                rand_idx = torch.randperm(rays_rgb.shape[0])
                rays_rgb = rays_rgb[rand_idx]
            i_batch = 0

        #####  Core optimization loop  #####
        rgb, disp, acc, depth, extras = render(H, W, focal, chunk=args.chunk, rays=batch_rays,
                                               depth_priors=target_prior[:, 0],
                                               depth_confidences=target_prior[:, 1],
                                               retraw=True, **render_kwargs_train)
        # pdb.set_trace()
        optimizer.zero_grad()
        img_loss = img2mse(rgb, target_s)
        depth_loss = F.smooth_l1_loss(depth, target_prior[:, 0], beta=0.2)
        trans = extras['raw'][..., -1]
        loss = img_loss + args.depth_loss_weight * depth_loss
        psnr = mse2psnr(img_loss)

        loss.backward()
        optimizer.step()

        # NOTE: IMPORTANT!
        ###   update learning rate   ###
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate
        ################################

        ##### Refine depth maps and ray importance maps #####
        refinement_round = (i - args.depth_refine_start) // args.depth_refine_period
        if depth_priors is not None and i > args.depth_refine_start and \
                i % args.depth_refine_period == 0 and refinement_round <= args.depth_refine_rounds:
            print('Render RGB and depth maps for refinement...')
            print()
            refinement_save_path = os.path.join(save_path, 'refinement{:04d}'.format(refinement_round))
            if not os.path.exists(refinement_save_path):
                os.makedirs(refinement_save_path)

            with torch.no_grad():
                # Refine depth maps
                # inference depth
                masks_t = np.expand_dims(refine_masks_bin, axis=-1)
                masks_t = np.repeat(masks_t, 4, axis=-1)
                rays_rgb_t = rays_rgb_ori[masks_t == 1].reshape(-1, 4, 3)

                batch_t = torch.Tensor(rays_rgb_t).to(device)  # [-1, 4, 3]
                batch_t = torch.transpose(batch_t, 0, 1)  # [4, -1, 3]
                batch_rays_t, target_s_t = batch_t[:2], batch_t[2]
                target_prior_t = batch_t[3]

                rgb_t, _, _, depth_t, _ = render(H, W, focal, chunk=args.chunk, rays=batch_rays_t,
                                                 depth_priors=target_prior_t[:, 0],
                                                 depth_confidences=target_prior_t[:, 1],
                                                 retraw=True, **render_kwargs_train)
                img_diff = torch.mean((rgb_t - target_s_t) ** 2, dim=1)
                quantile = torch.quantile(img_diff, 0.5)
                depth_to_refine1 = (img_diff < quantile).reshape(*depth_t.shape)
                depth_diff = torch.pow(depth_t - target_prior_t[:, 0], 2)
                quantile = torch.quantile(depth_diff, 0.2)
                depth_to_refine2 = (depth_diff > quantile).reshape(*depth_t.shape)
                depth_to_refine = depth_to_refine1 * depth_to_refine2
                d_t = target_prior_t[:, 0]
                d_t[depth_to_refine] = depth_t[depth_to_refine]
                rays_rgb_t = torch.transpose(batch_t, 0, 1).cpu().numpy()  # [-1, 4, 3]
                rays_rgb_t = rays_rgb_t.reshape(-1, 3)
                rays_rgb_ori[masks_t == 1] = rays_rgb_t

                # median filter
                for j in range(depth_priors.shape[0]):
                    depth_t = rays_rgb_ori[j, :, :, 3, 0].copy()
                    depth_prior = depth_priors[j].copy()
                    mask_gt = (refine_masks_bin[j]).astype(np.uint8)
                    kernel = np.ones((3, 3), np.uint8)
                    mask_gt = cv2.dilate(mask_gt, kernel, iterations=1)
                    depth_confidences[mask_gt] = 1.
                    coords = np.transpose(np.where(mask_gt == 1))  # [n, 2]
                    for coord in coords:
                        d_block = depth_prior[max(coord[0] - 5, 0): min(coord[0] + 5, W - 1),
                                  max(coord[1] - 5, 0): min(coord[1] + 5, W - 1)]
                        outer_layer = np.concatenate((d_block[0], d_block[-1], d_block[1:-1, 0], d_block[1:-1, -1]))
                        depth_prior[coord[0], coord[1]] = np.median(outer_layer)

                    depth_diff = np.abs(depth_t - depth_prior)
                    depth_diff_non0 = depth_diff[depth_diff > 0]
                    quantile = np.percentile(depth_diff_non0, 1 - args.depth_refine_quantile)
                    depth_to_refine = (depth_diff > quantile).reshape(H, W)
                    depth_priors[j][depth_to_refine] = depth_prior[depth_to_refine]
                    rays_rgb_ori[j, :, :, 3, 0][depth_to_refine] = depth_prior[depth_to_refine]

                for j in range(depth_priors.shape[0]):
                    # depth_priors[j][refine_masks_bin[j].astype(np.uint8) == 1] = 0
                    filename = os.path.join(refinement_save_path, '{}_depth.npy'.format(j + 1))
                    np.save(filename, depth_priors[j])
                    disp_visual = visualize_depth(depth_priors[j])
                    filename = os.path.join(refinement_save_path, '{}_depth.png'.format(j + 1))
                    cv2.imwrite(filename, disp_visual)
                refine_masks_bin = refine_area(tissue_masks_bin, depth_priors, images)

                print('\nRefinement finished, intermediate results saved at', refinement_save_path)

        # Rest is logging
        if i % args.i_weights == 0:
            path = os.path.join(save_path, 'checkpoints', '{:06d}.tar'.format(i))
            torch.save({
                'global_step': global_step,
                'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, path)
            print('Saved checkpoints at', path)

        if i % args.i_print == 0:
            tqdm.write(f"[TRAIN] Iter: {i}  img_loss: {img_loss.item()} depth_loss: {depth_loss.item()} "
                       f"Loss: {loss.item()}  PSNR: {psnr.item()}")
            writer.add_scalar("Loss", loss.item(), i)
            writer.add_scalar("PSNR", psnr.item(), i)

        global_step += 1

    with torch.no_grad():
        testsavedir = os.path.join(save_path, 'results')
        render_poses = poses_tensor
        if args.depth_refine_rounds > 0:
            refinement_save_path = os.path.join(save_path, 'refinement{:04d}'.format(args.depth_refine_rounds))
            for j in range(render_poses.shape[0]):
                depth_priors[j] = np.load(os.path.join(refinement_save_path, '{}_depth.npy'.format(j + 1)))
        os.makedirs(testsavedir, exist_ok=True)
        rgbs, disps, depths = render_path(render_poses, hwf, args.chunk, render_kwargs_test, sc=sc,
                                          depth_priors=torch.from_numpy(depth_priors).to(device),
                                          depth_confidences=torch.from_numpy(depth_confidences).to(device),
                                          savedir=testsavedir, render_factor=args.render_factor,
                                          image_list=image_list)
        print('Done rendering', testsavedir)


if __name__ == '__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    parser = config_parser()
    args = parser.parse_args()
    train(args)
