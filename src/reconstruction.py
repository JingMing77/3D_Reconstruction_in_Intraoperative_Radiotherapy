import numpy as np
from cv2 import cv2

import open3d as o3d
from matplotlib import pyplot as plt


def vis_sparse_pc(rgb_path=None, depth_path=None, mask_path=None, fluorescent_path=None,
                  H=307, W=409, factor=0.5, is_disp=False):
    if rgb_path is None or depth_path is None or mask_path is None:
        raise RuntimeError(f'need to input rgb_path, depth_path and mask_path')

    print("Read Redwood dataset")

    mask = cv2.imread(mask_path, 0)
    # mask = cv2.erode(mask, kernel=np.ones((25, 25)))
    mask = (mask > 10)
    color_image = cv2.imread(rgb_path, 1)
    color_image = cv2.resize(color_image, (W, H))
    color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
    if fluorescent_path is not None:
        fluor = cv2.imread(fluorescent_path, 1)
        fluor = cv2.resize(fluor, (W, H))
        fluor = cv2.cvtColor(fluor, cv2.COLOR_BGR2RGB)
        fluor[:, :, 0] = np.zeros((H, W))
        fluor[:, :, 2] = np.zeros((H, W))

        color_image[mask] = fluor[mask]
        color_image = color_image / 2
        color_image = np.uint8(color_image)

    depth_map = np.load(depth_path)
    depth_map = cv2.resize(depth_map, (W, H))

    depth0 = depth_map.copy()
    if is_disp:
        depth_map = 1 / (depth_map + 1e-6)

    # depth_map: [0, 1]
    invalid_mask = np.logical_or(np.isnan(depth_map), np.logical_not(np.isfinite(depth_map)))
    # invalid_mask = np.logical_or(invalid_mask, mask)
    depth_min = np.percentile(depth_map[np.logical_not(invalid_mask)], 1)
    depth_max = np.percentile(depth_map[np.logical_not(invalid_mask)], 99)
    depth_map[depth_map < depth_min] = depth_min
    depth_map[depth_map > depth_max] = depth_max
    depth_map = 0.1 * (depth_map - depth_min) / (depth_max - depth_min)

    depth_map = depth_map.astype(np.float32)  # [0, 0.1]

    depth_map[invalid_mask] = 1
    depth_map[depth0 == 0] = 1
    depth_map += factor  # # [factor, 0.1 + factor]

    # Create Open3D depth image
    depth_image = o3d.geometry.Image(depth_map)

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(color_image), depth_image,
        depth_scale=5, convert_rgb_to_intensity=False)

    print(rgbd_image)
    plt.subplot(1, 2, 1)
    plt.title('Redwood grayscale image')
    plt.imshow(rgbd_image.color)
    plt.subplot(1, 2, 2)
    plt.title('Redwood depth image')
    plt.imshow(rgbd_image.depth)
    plt.show()

    Cam_Intrinsic = o3d.camera.PinholeCameraIntrinsic()
    Cam_Intrinsic.set_intrinsics(
        width=depth_map.shape[1], height=depth_map.shape[0],
        fx=942.05, fy=942.05, cx=181.44, cy=156.68)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image, Cam_Intrinsic)

    # Flip it, otherwise the pointcloud will be upside down
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    o3d.io.write_point_cloud("test.ply", pcd)

    o3d.visualization.draw_geometries([pcd])


def remove_isolated_ones(input_array):
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(input_array.astype(np.uint8))

    # Filter out isolated ones
    filtered_array = np.zeros_like(input_array)
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area > 1:  # Keep only components with area greater than 1
            component_mask = (labels == label)
            filtered_array[component_mask] = 1

    return filtered_array


def fill_arr(arr):
    arr = remove_isolated_ones(arr)
    h, w = arr.shape
    processed_arr = np.zeros_like(arr)
    for i in range(2):
        for r in range(h):
            row = arr[r, :]
            first_one = np.argmax(row)
            last_one = w - np.argmax(row[::-1])
            if last_one == w:
                last_one = 0

            processed_arr[r, first_one:last_one] = 1

        for col in range(w):
            column = arr[:, col]
            first_one = np.argmax(column)
            last_one = h - np.argmax(column[::-1])
            if last_one == h:
                last_one = 0

            processed_arr[first_one:last_one, col] = 1

    contours, _ = cv2.findContours(processed_arr.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output_array = np.copy(processed_arr)

    # Fill the areas surrounded by contours with 1
    for contour in contours:
        cv2.fillPoly(output_array, [contour], 1)

    return output_array


def convert_depth_to_3d_model(depth_file_path, mask_image_path, output_mha_path,
                              N_layer=100, depth_r=800):
    import SimpleITK as sitk
    # Load depth from .npy file and normalize
    depth_data = np.load(depth_file_path)

    # Load mask image
    mask_image = cv2.imread(mask_image_path, cv2.IMREAD_GRAYSCALE)
    mask_image = np.uint8(mask_image > 10)
    H, W = mask_image.shape
    depth_data = cv2.resize(depth_data, (W, H))
    depth_data = depth_data * mask_image
    depth_data_min = np.percentile(depth_data[depth_data > 0], 1)
    depth_data_max = np.percentile(depth_data[depth_data > 0], 98)
    depth_data[depth_data < depth_data_min] = depth_data_min
    depth_data[depth_data > depth_data_max] = depth_data_max
    depth_data_median = np.median(depth_data)
    depth_data = depth_data * (depth_r / depth_data_median)
    # depth_data = (depth_data - depth_data_min) / (depth_data_max - depth_data_min)

    # Create 3D model
    N_layer = int(N_layer)
    model_data = np.zeros((N_layer, H, W))
    depth_interval = (np.max(depth_data) - np.min(depth_data)) / N_layer

    for i in range(N_layer):
        depth_min = i * depth_interval + np.min(depth_data)
        depth_max = (i + 1) * depth_interval + np.min(depth_data)
        mask_region = (depth_data >= depth_min) & (depth_data < depth_max) & (mask_image > 0)

        # Fill internal region with 1
        mask_region = mask_region.astype(np.uint8)
        mask_region = fill_arr(mask_region)
        model_data[i][mask_region == 1] = 1

        # cv2.imshow("!", model_data[i] * 255)
        # cv2.waitKey(0)

    # Create SimpleITK image from the model data
    model_image = sitk.GetImageFromArray(model_data)

    # Set the spacing (resolution) of the image
    spacing = (1.2695, 1.2695, 1.2695)
    model_image.SetSpacing(spacing)

    # Save the model image as .mha file
    sitk.WriteImage(model_image, output_mha_path)


if __name__ == '__main__':
    # parser = config_parser()
    # args = parser.parse_args()
    #

    n = 1
    data_name = '20230523'
    convert_depth_to_3d_model('../logs/' + data_name + '/filter/{}.npy'.format(n),
                              '../' + data_name + '/mask_l/{}.png'.format(n),
                              data_name + ".mha")
    exit()

    vis_sparse_pc('../20230523/l/images/{}.jpg'.format(n),
                  '../logs/20230523/filter/{}.npy'.format(n),
                  '../20230523/mask_l/{}.png'.format(n),
                  factor=0.01)

    # vis_sparse_pc('../dia/l/images/{}.jpg'.format(n),
    #               '../logs/dia/filter/{}.npy'.format(n),
    #               '../dia/mask_l/{}.png'.format(n),
    #               '../dia/image16.png',
    #               factor=0.5)

