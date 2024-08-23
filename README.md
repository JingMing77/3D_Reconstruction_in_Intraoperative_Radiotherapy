# This is an implementation of the paper "3D Reconstruction based Descending Stereo Images in Intraoperative Radiotherapy"

## Get Start

1. You will need to follow the instructions in [weiyithu/NerfingMVS](https://github.com/weiyithu/NerfingMVS) to **install COLMAP** on your machine.

    (Notice that you will encounter an error if you download the original COLMAP )

2. **Download the dataset** from [here](https://drive.google.com/file/d/165Dkw_Ot9HOxTiU4zf9A7aBFE2Lk5eqk/view?usp=sharing) , which includes (a)some descending stereo images we acquired with our own equipment, (b)the public data we used in the experiment, and (c) the data of each scene has been rectified and segmented.

    You can test your demo dataset according to this folder structure.

```shell
|───exp_name
|    |────── l
|    |   |    |   images
|    |   |    |    | 1.jpg
|    |   |    |    | ...
|    |   |    |   rectify
|    |   |    |    | 1_$cut_width.jpg
|    |   |    |    | ...
|    |   |    |───── disparity / depth (for metric)
|    |   |    |    | 1.png
|    |   |    |    | ...
|    |—————— r
|    |   |    |   images
|    |   |    |    | 1.jpg
|    |   |    |    | ...
|    |   |    |   rectify
|    |   |    |    | 1_$cut_width.jpg
|    |   |    |    | ...
|    |—————— mask_l
|    |   |    | 1.png
|    |   |    | ...
|    |—————— mask_r
|    |   |    | 1.png
|    |   |    | ...
|    |—————— train.txt
|    |—————— test.txt
|───configs
|    $exp_name.txt
|     ...
```

3. **Install the packages** used by the following command

```shell
pip install torch torchvision torchaudio
pip install -r requirements.txt
```

## Train and Test

1. Run COLMAP

```shell
sh colmap.sh $scene_name
```

2. Run demo dataset

```shell
python run.py --config configs/$scene_name.txt --mask_guide_sample_rate 0.4 --depth_refine_start 100000 --depth_refine_period 1000 --depth_refine_rounds 3 --N_iters 200001 --train_binocular --demo
```

3. Run public dataset

```shell
python run.py --config configs/$scene_name.txt
```

## Acknowledgement

Our code is based on [weiyithu/NerfingMVS](https://github.com/weiyithu/NerfingMVS). We also refer to [med-air/EndoNeRF](https://github.com/med-air/EndoNeRF).

