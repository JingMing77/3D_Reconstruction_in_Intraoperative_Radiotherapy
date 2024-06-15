import configargparse


def config_parser():
    
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True,
                        help='config file path')
    parser.add_argument("--expname", type=str,
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/',
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='./data/llff/fern',
                        help='input data directory')
    parser.add_argument("--demo", action='store_true',
                        help='if demo, no evaluation')
    parser.add_argument("--blender", action='store_true',
                        help='use blender dataset')
    parser.add_argument("--cam_l", default='942.05, 181.44, 156.68', type=str,
                        help='fx, cx, cy')
    parser.add_argument("--cam_r", default='950.56, 215.83, 154.01', type=str,
                        help='fx, cx, cy')
    parser.add_argument("--R_matrix", default='0.76, 0.00, -0.65,'
                                              '0.00, 1.00, 0.00,'
                                              '0.65, 0.00, 0.76', type=str)
    parser.add_argument("--t_vec", default='-406.05, 1.26, 152.08', type=str)
    parser.add_argument("--baseline", default=435, type=float)
    parser.add_argument("--cam_dist", default=300, type=float)
    parser.add_argument("--pixel_size", default=3.45e-3, type=float)
    parser.add_argument("--delta_d", default=833, type=float)
    parser.add_argument("--depth_gt_mean", default=600, type=float)

    # spsg match options
    parser.add_argument(
        '--resize', type=int, nargs='+', default=[640, 480],
        help='Resize the input image before running inference. If two numbers, '
             'resize to the exact dimensions, if one number, resize the max '
             'dimension, if -1, do not resize')
    parser.add_argument(
        '--viz', action='store_true',
        help='Visualize the matches and dump the plots')

    # depth priors options
    parser.add_argument("--depth_N_rand", type=int, default=2,
                        help='batch size for depth')
    parser.add_argument("--depth_N_iters", type=int, default=101,
                        help='number of iterations for depth')
    parser.add_argument("--depth_H", type=int, default=288, 
                        help='the height of depth image (must be 16x)')
    parser.add_argument("--depth_W", type=int, default=384,
                        help='the width of depth image (must be 16x)')
    parser.add_argument("--depth_lrate", type=float, default=4e-4,
                        help='learning rate for depth')
    parser.add_argument("--depth_i_weights", type=int, default=50,
                        help='frequency of weight ckpt saving for depth')
    parser.add_argument("--depth_i_print",   type=int, default=20,
                        help='frequency of console printout and metric loggin')
    
    # nerf options
    parser.add_argument("--nerf_type", type=str, default="original",
                        help='nerf network type')
    parser.add_argument("--netdepth", type=int, default=8,
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256,
                        help='channels per layer')
    parser.add_argument("--N_rand", type=int, default=32*32*4,
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--N_iters", type=int, default=100001,
                        help='number of iterations')
    parser.add_argument("--lrate", type=float, default=5e-4,
                        help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=250,
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--chunk", type=int, default=1024*32,
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024*64,
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_reload", action='store_true',
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None,
                        help='specific weights npy file to reload for coarse network')
    parser.add_argument("--N_samples", type=int, default=64,
                        help='number of coarse samples per ray')

    parser.add_argument("--mask_loss", action='store_true',
                        help='enable erasing loss for masked pixels')
    parser.add_argument("--train_binocular", action='store_true',
                        help='use binocular images for training')
    parser.add_argument("--mask_guide_sample_rate", type=float,
                        default=0.4, help='percentage of sampled rays')
    parser.add_argument("--no_depth_sampling", action='store_true',
                        help='disable depth-guided ray sampling?')

    parser.add_argument("--depth_loss_weight", type=float, default=0.01,
                        help='weight of depth loss')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--do_half_precision", action='store_true',
                        help='do half precision training and inference')

    parser.add_argument("--no_depth_refine", action='store_true',
                        help='disable depth refinement')
    parser.add_argument("--depth_refine_start", type=int, default=0,
                        help='number of iters to start refine depth maps')
    parser.add_argument("--depth_refine_period", type=int, default=4,
                        help='number of freq to refine depth maps')
    parser.add_argument("--depth_refine_rounds", type=int, default=4,
                        help='number of rounds of depth map refinement')
    parser.add_argument("--depth_refine_quantile", type=float, default=0.5,
                        help='proportion of pixels to be updated during depth refinement')

    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true',
                        help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=1,
                        help='set 1 for hashed embedding, 0 for default positional encoding, 2 for spherical')
    parser.add_argument("--i_embed_views", type=int, default=2,
                        help='set 1 for hashed embedding, 0 for default positional encoding, 2 for spherical')
    parser.add_argument("--multires", type=int, default=10,
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4,
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0.,
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')
    parser.add_argument("--render_only", action='store_true',
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true',
                        help='render the test set instead of render_poses path')
    parser.add_argument("--render_factor", type=int, default=0,
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')
    parser.add_argument("--white_bkgd", action='store_true',
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--factor", type=int, default=8,
                        help='downsample factor for LLFF images')
    parser.add_argument("--N_views", type=int, default=120,
                        help='the number of render views')
    parser.add_argument("--no_ndc", action='store_true',
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--lindisp", action='store_true',
                        help='sampling linearly in disparity rather than depth')
    parser.add_argument("--spherify", action='store_true',
                        help='set for spherical 360 scenes')
    parser.add_argument("--near", type=float, default=0.01,
                        help='abs near bound')
    parser.add_argument("--far", type=float, default=0.1,
                        help='abs far bound')
    parser.add_argument("--topk", type=int, default=4,
                        help='topk for consis error')
    parser.add_argument("--llff_renderpath", type=str, default='spiral',
                        help='options: spiral, fixidentity, zoom')

    parser.add_argument("--i_print", type=int, default=1000,
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img", type=int, default=10000,
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=20000,
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=100000,
                        help='frequency of testset saving')
    parser.add_argument("--i_video", type=int, default=100000,
                        help='frequency of render_poses video saving')
    parser.add_argument("--video_fps", type=int, default=30,
                        help='FPS of render_poses video')


    # filter options
    parser.add_argument("--worker_num", type=int, default=8,
                        help='the number of worker for multiprocessing')
    
    return parser