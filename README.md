##### colmap 3.7

用一个只有 左/右相机 拍摄的图片的文件夹，拿左右图片分别重建稀疏点云 用 colmap.sh

```bash
sh colmap.sh ./data/f5/l
# 貌似必须按 NerfingMVS 配置 colmap
```

<img src="README.assets/L.jpg" alt="L" style="zoom:67%;" title="原图"/>  <img src="README.assets/image-20230228171917716.png" alt="image-20230228171917716" style="zoom:20%;" />   <img src="README.assets/image-20230228172002171.png" alt="image-20230228172002171" style="zoom:30%;" />

> 相机参数（缩放图像为原来的1/5）
>
> cam0：fx, cx, cy 
> 942.05, 181.44, 156.68
>
> cam1：fx, cx, cy
> 950.56,  215.83, 154.01
>
> R
>
> '0.76, 0.00, -0.65,'
> '0.00, 1.00, 0.00,'
> '0.65, 0.00, 0.76'
>
> t
> -406.05, 1.26, 152.08
>
> **x2' = R' * x1' + t'**

指定相机参数，只用左相机拍的图，用colmap 计算相机位姿和深度图。再根据左右相机对应点视差相等，计算右相机视差。用外参R和t 估计右相机位姿。

colmap 结果   <img src="README.assets/1-16778368130842.png" alt="1" style="zoom:33%;" />      <img src="README.assets/depthMap.png" alt="depthMap" style="zoom:33%;" />   <img src="README.assets/1-16779322785971.png" alt="1" style="zoom:33%;" />  

 <img src="README.assets/image-20230303174910248.png" alt="image-20230303174910248" style="zoom:33%;" />   <img src="README.assets/image-20230303174935055.png" alt="image-20230303174935055" style="zoom:30%;" />

视差图       <img src="README.assets/disp-16779340651963.png" alt="disp" style="zoom:40%;" /><img src="README.assets/disp-16779341218505.png" alt="disp" style="zoom:40%;" />    附加梯度和高光的mask   <img src="README.assets/mask_grad_reflect-16779363107787.png" alt="mask_grad_reflect" style="zoom: 80%;"  title="mask"/>



##### SuperPoint + SuperGlue 稀疏点云匹配

[SuperPoint / SuperGlue](https://blog.csdn.net/weixin_44580210/article/details/122284145?ops_request_misc=%7B%22request%5Fid%22%3A%22167768365216782425195789%22%2C%22scm%22%3A%2220140713.130102334.pc%5Fall.%22%7D&request_id=167768365216782425195789&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~first_rank_ecpm_v1~rank_v31_ecpm-2-122284145-null-null.142^v73^pc_search_v2,201^v4^add_ask,239^v2^insert_chatgpt&utm_term=superpoint super&spm=1018.2226.3001.4187)

> 使用预训练模型，可以得到大约 100+ 个匹配点对，resize = [640, 480]
>
> 添加 **极线约束**: 删除 dy > 20 的匹配点
>
> 删除 confidence < 0.4 的点对

<div>
<center class="half">
<img src="README.assets/1_matches.png" alt="1_matches" style="zoom: 33%;" title="f5"/>
<img src="README.assets/1_matches-16776838217542.png" alt="1_matches" style="zoom: 33%;" title="原图匹配结果"/>
<img src="README.assets/10_matches.png" alt="1_matches" style="zoom: 33%;" title="校正后（不去除黑边）匹配结果"/>
<img src="README.assets/18_655_matches.png" alt="1_matches" style="zoom: 33%;" title="校正后（去除黑边）匹配结果"/>
<br> 匹配点对
    <br><img src="README.assets/1-16785117129037.jpg" alt="expand_disp" style="zoom:50%;" title="f5_l1"/>
    	<img src="README.assets/expand_disp-16785115927074.png" alt="expand_disp" style="zoom:50%;" title="f5_l1"/>
    	<img src="README.assets/1_655.jpg" alt="expand_disp" style="zoom:50%;" title="heart_l1"/>
    	<img src="README.assets/expand_disp.png" alt="expand_disp" style="zoom:50%;" title="heart_l1"/>
    <br> 根据匹配点扩充后的视差图
    <br> 方法：在匹配点的领域内，通过匹配点的信息限制视差范围（在disp_center ± 3 内），使用 NCC 方法进行立体匹配，得到扩充后的视差图，有值的点翻了几倍，但是用这种方法得到的视差图并不能作为STTR 网络的监督，训练出来效果还是很差）
    <br><img src="README.assets/disp_sup.png" alt="expand_disp" style="zoom:50%;" />
    <img src="README.assets/disp_sup-167776592547810.png" alt="disp_sup" style="zoom:50%;" />
    <br> 根据匹配点得到稠密的视差图
    <br> 方法：用匹配点做标记，opencv的分水岭算法做分割，在分割的区域内通过匹配点的信息限制视差范围，同样使用 NCC 方法进行立体匹配 
</center>
</div>


​    **根据匹配点扩充后的稀疏视差图生成的稀疏点云**           <img src="README.assets/bandicam 2023-03-11  -small-horizontal.gif" alt="bandicam 2023-03-11  -small-horizontal" style="zoom:50%;" title="根据匹配点扩充后的稀疏视差图生成的稀疏点云"/>



> **用 SPSG 特征匹配得到的稀疏点云做监督**
>
> <div>
> <center class="half">
>   <img src="README.assets/10_depth-16776851509451.png" alt="10_depth" style="zoom:50%;" />
>   <img src="README.assets/0_depth-16776851714583.png" alt="0_depth" style="zoom:50%;" />
> 	<br>  得到的深度先验
> </center>
> <center class="half">
>   <img src="README.assets/10_depth_confidence.png" alt="10_depth" />
>   <br>  深度先验的置信度
> </center>
> <center class="half">
>   <img src="README.assets/10_depth-16777221834531.png" alt="10_depth" />
>   <br>  nerf 优化的深度图
> </center>
>
> **增加SPSG得到的点云数量**：给匹配点对周围领域的点赋值



##### depth prior

###### STTR

```python
# sttr 
input：左右视图，左图的视差图和遮挡区域，
# 不再使用真实视差图监督 -> 改用稀疏点云监督
（colmap只能用于静态场景的稀疏点云生成，动态场景要逐帧计算）

output：得到稠密的视差图：dense_disp (depthMap可以由 dispMap 计算得到)

重写 dataset 和 dataloader 加载输入数据（left, right, disp, occ_mask）
预处理时 augment 标记并去除视差值超出范围的点、明显的噪声、遮挡点（不能找到右图中的对应点） 
```

<div>
<center class="half">
    <img src="README.assets/disp-16774046563692.png" alt="disp" style="zoom:60%;" title="处理前"/>
    <img src="README.assets/disp_color-16778142052602.png" alt="disp" style="zoom:60%;" title="处理前_color"/>
    <img src="README.assets/disp-16774049146324.png" alt="disp" style="zoom:60%;" title="处理后"/>
    <img src="README.assets/disp_color.png" alt="disp" style="zoom:60%;" title="处理后_color"/>
<br>
处理前后的视差图（colmap重建结果） point_num: 109822 -> 93248
</center>
</div>
> ```python
> # models/stereo_depth_prior/utils/preprocess.py 中进行预处理
> sc = depth_gt_mean / colmap_median  # 因为 colmap 算出来很多噪声，所以用中值代替均值  600/20+ = 20+
> disp_gt = (fx * baseline / (colmap_depth * sc + 1e-6))  
> # 	真实的 disp_gt 大概在 550-700 左右，
> # z = f*B/d ∈ (590, 751)
> # 真实的 disp_gt 与未校正的 disp 有一个差值 delta_d，输入网络的（体现在图像上的） disp ∈ (-400, 400) -> disp_gt ∈ (400, 1200)
> 
> # 在自己的数据集上，由于噪声太多，所以 augment 之前全是黑的
> 1. set large/small values to be 0
> 2. compute left and right occ_region (做了之后会把很多值删掉，就不做这一步了)
> 
> ```
> <img src="README.assets/disp-16779937015752.png" alt="disp" style="zoom:50%;" title="增强前"/>      <img src="README.assets/disp-16779955066774.png" alt="disp" style="zoom:50%;" title="用于监督的视差图"/>    <img src="README.assets/21_depth.png" alt="21_depth" style="zoom:50%;" title="直接用disp_gt训练STTR的结果"/>   <img src="README.assets/1_depth-16780268832737.png" alt="1_depth" style="zoom:50%;" title="直接用disp_gt训练STTR的结果"/>
>
> ```python
> # 根据特征匹配的加权均值disp_mean, 把 disp_gt 转换到图像上的 disp。
> 理论上 disp_mean ∈ (-w, w) 且应该是随着视角升高逐渐减小的，(实际是从 225 ~ -138 递减)
> delta_d = disp_gt_median - disp_mean  
> disp_in_image = disp_gt - delta_d  # 把 disp_gt 转换到图像上
> # disp_in_image ∈ (-w, w) 可以把上限在扩大一点：(-1.5w, 1.5w) ，不然中间的点会被滤掉
> # 最后要 ascontiguousarray 得到完整的视差图
> 
> # occ_mask: disp_gt = 0 的点（即disp_mean = -delta_d 的点）
> ```
>
>  **用 disp_in_image 训练的结果仍然不行，可能由于这个 归一化到（-w，w） 的 disp_in_image 只是一个相对的视差，只能反应物体的远近，不能用于计算左右图像上的匹配点**
>
> **为左右视图和视差图补上黑边，使得视差值＞0，结果依然不好**
>
>   <img src="README.assets/disp_augment-167803192662111.png" alt="disp_augment" style="zoom:50%;" title="(-w, w) 中间的点会被滤掉"/>    <img src="README.assets/disp_augment-16780317830179.png" alt="disp_augment" style="zoom:50%;" title="disp_in_image"/>    <img src="README.assets/disp_augment-167803205801913.png" alt="disp_augment" style="zoom:50%;" title="用于监督的稀疏视差∈(-1.5w, 1.5w)"/>  
>
>  <img src="README.assets/occ_mask.png" alt="occ_mask" style="zoom:50%;" title="colmap occ_mask 白色为有效点"/>    <img src="README.assets/occ_mask-16780732370382.png" alt="occ_mask" style="zoom:50%;" title="occ_mask for training"/>    <img src="README.assets/occ_mask-16780734917534.png" alt="occ_mask" style="zoom:50%;" title="occ_mask for training"/>   
>
> <img src="README.assets/1_depth-16796243159031.png" alt="1_depth" style="zoom:50%;" title="disp_in_image 训练的结果"/>     <img src="README.assets/10_depth-16796243386114.png" alt="10_depth" style="zoom:50%;" title="disp_in_image 训练的结果"/>     <img src="README.assets/23_depth-16796243499086.png" alt="23_depth" style="zoom:50%;" title="disp_in_image 训练的结果"/>
>



<div>
<center class="half">
    f5
    <img src="README.assets/0_depth.png" alt="disp" style="zoom:60%;" title="colmap 稀疏点云监督"/>
    <img src="README.assets/sttr_disp_pred_kitti_finetuned.png" alt="disp" style="zoom:60%;" title="kitti_finetuned" />
    <img src="README.assets/sttr_disp_pred_light_sceneflow.png" alt="disp" style="zoom:60%;" title="light_sceneflow"/>
	<br>
colmap 稀疏点云监督， STTR不同预训练模型 得到的视差图
    <br> heart
    <img src="README.assets/sttr_disp_pred_light_sceneflow-16799080350891.png" alt="disp" style="zoom:60%;" title="kitti_finetuned"/>
    <img src="README.assets/sttr_disp_pred_light_sceneflow-16799082114463.png" alt="disp" style="zoom:60%;" title="kitti_finetuned"/>
    </center>
</div>





1) colmap 不能直接用于 动态场景的稀疏点云重建，不过这样做得到的视差图还不错，可能是由于心脏动的幅度不大，如果用 colmap 对左右图像做重建会失败，找不到匹配点
2) colmap 在自己拍的数据集上重建失败，geometric.bin 会比原图数量少
3) 用更强的特征匹配方法（如SuperGlue）对每一帧立体像对做特征点匹配，得到当前时刻的稀疏点云
4) 或者用针对动态场景的三维重建方法，【[动态场景下实时三维重建论文汇总](https://blog.csdn.net/qinqinxiansheng/article/details/119737051)】

###### PAM

> 基于视差注意力的无监督立体匹配方法
>
> input：左右RGB图
>
> output：视差图

<img src="README.assets/1-16787222352873.jpg" alt="1" style="zoom:40%;" title="原图"/>   <img src="README.assets/1_depth-16787221819431.png" alt="1_depth" style="zoom:40%;" title="直接在数据集上训练，无稀疏点云监督 epoch=100 缩放为 h*w=288*384"/>   <img src="README.assets/10_depth-16787999330591.png" alt="10_depth" style="zoom:40%;" title="直接用PAMS SceneFlow 预训练模型 10.jpg" />  <img src="README.assets/18_depth.png" alt="18_depth" style="zoom:40%;" title="直接用PAMS SceneFlow 预训练模型 18.jpg"/>

<img src="README.assets/image-20230314193937636.png" alt="image-20230314193937636" style="zoom:80%;" title="直接用PAMS SceneFlow 预训练模型"/>

###### monocular

> 用单目估计方法得到深度图，再根据匹配点的结果确定尺度

**直接使用colmap预测结果作为监督训练单目深度估计网络**

<img src="README.assets/image-20230323162654438.png" alt="image-20230323162654438" style="zoom:67%;" />

<img src="README.assets/1_depth-16795600637341.png" alt="1_depth" style="zoom:50%;" title="直接使用colmap作为监督"/>    <img src="README.assets/14_depth.png" alt="14_depth" style="zoom:50%;" title="直接使用colmap作为监督"/>    <img src="README.assets/23_depth-16795601003674.png" alt="23_depth" style="zoom:50%;" title="直接使用colmap作为监督"/>



**剔除噪声和遮挡区域后的深度图作为监督训练单目深度估计网络**
<img src="README.assets/image-20230323204949024.png" alt="image-20230323204949024" style="zoom:67%;" />

<img src="README.assets/1_depth-16795758934996.png" alt="1_depth" style="zoom:50%;" />     <img src="README.assets/14_depth-16795759044408.png" alt="14_depth" style="zoom:50%;" />     <img src="README.assets/23_depth-167957594598310.png" alt="23_depth" style="zoom:50%;" />




##### render

> ​	用原 NerfingMVS 方法，只有 ==深度和置信度== 
>
> ​	用STTR 估计得到的左视差图推算右视差图：d = disp_left[y, x]，则disp_right[y, x-d] = d
>
> ​	从 视差图 推算 深度图的 z = f * B / d，f 用标定的 focal_x_left 
>
> ​	得到深度图后要 align_scales，将深度和colmap得到的深度对齐



> 用 多视角一致性检测 方法算出的 error map，背景和边缘 error 较大。
>
> （用当前视角下的 depth_prior，计算投影到其他视角下的深度图，和depth_prior 比较得到 error_map）

<img src="README.assets/depth_confidence.jpg" alt="depth_confidence" style="zoom:50%;" /><img src="README.assets/8_depth_confidence.png" alt="8_depth_confidence" style="zoom:50%;" />  <img src="README.assets/1_depth_confidence.png" alt="1_depth_confidence" style="zoom:50%;" /> <img src="README.assets/10_depth_confidence-16797298686102.png" alt="10_depth_confidence" style="zoom:50%;" />

> tissue mask + 梯度(sobel) + 深度误差过大= mask

<img src="README.assets/1.jpg" alt="1" style="zoom: 25%;" title="原图"/>  <img src="README.assets/1.png" alt="image-20230302170940475" style="zoom: 25%;" /> <img src="README.assets/grad.png" alt="grad" style="zoom:50%;" /> <img src="README.assets/reflect.png" alt="reflect" style="zoom:50%;" /> <img src="README.assets/mask_grad_reflect.png" alt="mask_grad_reflect" style="zoom:50%;" />

<img src="README.assets/image-20230324162938889.png" alt="image-20230324162938889" style="zoom:50%;" />   <img src="README.assets/image-20230324163045460.png" alt="image-20230324163045460" style="zoom:50%;" />   <img src="README.assets/image-20230324163056277.png" alt="image-20230324163056277" style="zoom:50%;" />    <img src="README.assets/image-20230324163109788.png" alt="image-20230324163109788" style="zoom:50%;" />

> 对组织内部，优化深度值不连续的区域（高光）





> 在 run_nerf.py 的 def render_rays() 中改变采样方法
> 基于高斯函数，均值 depth-prior，方差sigma 和 错误率、梯度正相关
> 标记了背景和高光区域的掩膜，让采样光线绕过去

> 训练时使用 左右相机拍到的（2N张）立体图像，而不是只使用单个相机的，右相机位姿通过外参计算，需要根据已知相机实际移动的距离，将尺度对齐到colmap估计的位姿上。
>
> 推理结果存在 ./logs/exp/nerf/results 里面

![1_depth](README.assets/1_depth.png)![10_depth](README.assets/10_depth.png) 



##### post-filter

基于NeRF渲染出的rgb图像和输入图像之间的差异，计算置信度

好像没有什么必要

##### ns

colmap -> nerf -> monocular

**colmap：** 前15个视角的位姿估计比较准，后面5个不准

![image-20230421220535657](README.assets/image-20230421220535657.png)

**nerf：** 加 深度、置信度、掩膜 引导采样 

得到的深度比较准，但是rgb不好。

![1](README.assets/1-16820862489593.png)   ![1_depth](README.assets/1_depth-16820862392861.png)

**monocular：** 加 NS Loss，改善遮挡和无纹理区域 

计算中间虚拟相机的rgb图和深度图（后面5张无效点较多，说明相机位姿估计不正确）

![image-20230423105931364](README.assets/image-20230423105931364.png)

> 环形路线
>
> **colmap - monocular**（原来的方法：深度网络
>
>  abs_rel |   sq_rel |     rmse | rmse_log |       a1 |       a2 |       a3 | 
> &   0.223  &   0.011  &   0.041  &   0.243  &   0.604  &   0.931  &   1.000	（全图）
>
> &   0.039  &   0.000  &   0.003  &   0.048  &   1.000  &   1.000  &   1.000	（组织区域）
>
> psnr: 10.717974662780762, ssim: 0.21593473249946205, lpips: 0.6527086496353149
>
> **NeRF - monocular（组织区域）**
>
>    abs_rel |   sq_rel |     rmse | rmse_log |       a1 |       a2 |       a3 | 
> &   0.046  &   0.000  &   0.004  &   0.056  &   0.982  &   1.000  &   1.000	（NeRF
>
> &   0.039  &   0.000  &   0.003  &   0.048  &   1.000  &   1.000  &   1.000	（深度网络
>
> psnr: 10.608259201049805, ssim: 0.2105353322807661, lpips: 0.614071249961853



不同高度图片 (psnr会更低一些) <img src="README.assets/image-20230423133253436.png" alt="image-20230423133253436" style="zoom:50%;" />   

> **NeRF - monocular（组织区域）** 
>
>    abs_rel |   sq_rel |     rmse | rmse_log |       a1 |       a2 |       a3 | 
> &   0.018  &   0.000  &   0.000  &   0.022  &   1.000  &   1.000  &   1.000	（NeRF
>
> &   0.018  &   0.000  &   0.000  &   0.023  &   1.000  &   1.000  &   1.000	（深度网络
>
> psnr: 8.441925048828125, ssim: 0.09985994900922941, lpips: 0.7266696691513062	

原图 	深度网络 	nerf 	滤波后

<img src="README.assets/1-16822297036053.jpg" alt="1" style="zoom:50%;" />  <img src="README.assets/1_depth-16822297249345.png" alt="1_depth" style="zoom:50%;" />  ![1_depth](README.assets/1_depth-16822296930381.png)   ![1](README.assets/1-16822297437257.png)

<img src="README.assets/10.jpg" alt="10" style="zoom:50%;" />   <img src="README.assets/10_depth-16822298899449.png" alt="10_depth" style="zoom:50%;" />  ![10_depth](README.assets/10_depth-168222990658811.png)   ![11](README.assets/11.png)

<img src="README.assets/1-168222998541515.jpg" alt="1" style="zoom: 45%;" />   <img src="README.assets/1_depth-168223000675217.png" alt="1_depth" style="zoom:50%;" />   <img src="README.assets/1_depth-168223003758519.png" alt="1_depth" style="zoom:90%;" />   <img src="README.assets/1-168223006603021.png" alt="1" style="zoom:90%;" />

**在自己拍摄的数据集上补充实验，验证加入NeRF对高光区域的改善**



**补充带高光的数据实验，和colmap比较** 

<img src="README.assets/image-20230423161238901.png" alt="image-20230423161238901" style="zoom:67%;" />

colmap：  <img src="README.assets/image-20230423162230051.png" alt="image-20230423162230051" style="zoom:50%;" />

**加了一段更新深度值的代码，迭代到5万轮之后更新4次无效的深度值** 

 原图， 深度先验，需要优化的区域（找深度梯度大的地方，把不是高光的地方剔除）

<img src="README.assets/image-20230423162124913.png" alt="image-20230423162124913" style="zoom:50%;" />   <img src="README.assets/17_depth.png" alt="17_depth" style="zoom:50%;" />  ![mask_refine_17](README.assets/mask_refine_17-16824339027241.jpg)   

<img src="README.assets/2.jpg" alt="2" style="zoom:50%;" />   <img src="README.assets/2_depth.png" alt="2_depth" style="zoom:50%;" />   ![mask_refine_2](README.assets/mask_refine_2.jpg)



17

<img src="README.assets/image-20230423162124913.png" alt="image-20230423162124913" style="zoom:50%;" />   <img src="README.assets/image-20230425131639946.png" alt="image-20230425131639946" style="zoom:50%;" />   <img src="README.assets/17.png" alt="17" style="zoom:50%;" />   <img src="README.assets/17_depth.png" alt="17_depth" style="zoom:50%;" />   

20

<img src="README.assets/20-16822379076785.jpg" alt="20" style="zoom:50%;" />   <img src="README.assets/20-16823998571961.jpg" alt="20" style="zoom:50%;" />   <img src="README.assets/40-16822378889233.png" alt="40" style="zoom:50%;" />   <img src="README.assets/20_depth.png" alt="20_depth" style="zoom:50%;" />   

### 实验

> **./utils/json2png.py**		将labelme标注的mask.json 文件转成图片，保存在mask_l / mask_r 文件夹中，同时保存添加掩膜的图像 ./f5/l/images_vis 中。
>
> ```python
> file_name = '../heart'  # 把这个改了就行
> image_dir = 'rectify'   # 'rectify'：校正后的图像（"_"后面是黑边的宽度）.or. 'images': 原图
> ```

##### 文件夹结构

```
111
|───exp_name
|    |────── l
|    |   |    |   images
|    |   |    |    | 1.jpg
|    |   |    |    | ...
|    |   |    |───── disparity / depth (for metric)
|    |   |    |    | 1.png
|    |   |    |    | ...
|    |—————— r
|    |   |    |   images
|    |   |    |    | 1.jpg
|    |   |    |    | ...
|    |   |    |───── disparity / depth (for metric)
|    |   |    |    | 1.png
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

> ```python
> gen_poses  # 生成poses_boundsl.npy
> # (N_img * 17)  ( pose(3*5), bd(close_depth, inf_depth) )
> 
> ```



##### 指令

```bash
cd monocular
conda activate NerfingMVS
nvidia-smi
export CUDA_VISIBLE_DEVICES=3

python run.py --config configs/c1.txt

# nohup
nohup python run.py --config configs/20230523.txt --mask_guide_sample_rate 0 --depth_loss_weight 0.01 --depth_refine_start 10000 --depth_refine_period 1000 --depth_refine_rounds 0 --N_iters 20001 --i_weights 3000 --train_binocular >20230523.log 2>&1

nohup python run.py --config configs/kidney.txt --mask_guide_sample_rate 0.4 --depth_loss_weight 0.01 --depth_refine_start 10000 --depth_refine_period 1000 --depth_refine_rounds 3 --N_iters 100001 --train_binocular >kidney.log 2>&1

nohup python run.py --config configs/liver.txt --mask_guide_sample_rate 0.4 --depth_loss_weight 0.01 --depth_refine_start 10000 --depth_refine_period 1000 --depth_refine_rounds 3 --N_iters 20001 --train_binocular >liver.log 2>&1

nohup python run.py --config configs/dia.txt --mask_guide_sample_rate 0.8 --depth_loss_weight 0.001 --depth_refine_start 10000 --depth_refine_period 1000 --depth_refine_rounds 3 --N_iters 100001 --train_binocular >dia.log 2>&1

nohup python run.py --config configs/f5.txt --mask_guide_sample_rate 0.4 --depth_loss_weight 0.01 --depth_refine_start 10000 --depth_refine_period 1000 --depth_refine_rounds 3 --N_iters 20001 --train_binocular >f5.log 2>&1

nohup python run.py --config configs/scared.txt --mask_guide_sample_rate 0.4 --depth_loss_weight 0.01 --depth_refine_start 10000 --depth_refine_period 1000 --depth_refine_rounds 3 --N_iters 20001 --train_binocular >scared.log 2>&1

nohup python run.py --config configs/scared2.txt --mask_guide_sample_rate 0.4 --depth_loss_weight 0.01 --depth_refine_start 10000 --depth_refine_period 1000 --depth_refine_rounds 3 --N_iters 20001 --train_binocular >scared2.log 2>&1

N_iters					# NeRF轮数，默认100001
train_binocular  		# 使用双目图像进行训练，不然默认只用左边的图训练

depth_loss_weight  		# depth_loss权重，为0则不用depth_loss
mask_guide_sample_rate  # 用mask引导采样的光线占比，为0则不用mask引导
depth_refine_start，depth_refine_period，depth_refine_rounds 		# depth_refine 开始，周期，轮数（e.g. 从5万轮开始，每1万轮优化一次，一共优化4次）
```



**dia** 

w/o refine





w/o mask

abs_rel |   sq_rel |     rmse | rmse_log |       a1 |       a2 |       a3 | 
&   0.026  &   0.000  &   0.001  &   0.031  &   1.000  &   1.000  &   1.000  \\

psnr: 26.23227310180664, ssim: 0.8748652576385915, lpips: 0.21984907984733582

psnr: 14.200566291809082, ssim: 0.46087278583379643, lpips: 0.8073318004608154



w/o filter

abs_rel |   sq_rel |     rmse | rmse_log |       a1 |       a2 |       a3 |
&   0.027  &   0.000  &   0.001  &   0.032  &   1.000  &   1.000  &   1.000  \\

训练集 psnr: 26.898359298706055, ssim: 0.8693463705321121, lpips: 0.19969585537910461

psnr: 15.194534301757812, ssim: 0.48118295980400244, lpips: 0.8467506170272827



filter

   abs_rel |   sq_rel |     rmse | rmse_log |       a1 |       a2 |       a3 | 
&   0.019  &   0.000  &   0.001  &   0.022  &   1.000  &   1.000  &   1.000  \\



f5



prior depth evaluation:

   abs_rel |   sq_rel |     rmse | rmse_log |       a1 |       a2 |       a3 |
&   0.090  &   0.728  &   6.382  &   0.119  &   0.911  &   1.000  &   1.000  \\

-> Done!
nerf depth evaluation:

   abs_rel |   sq_rel |     rmse | rmse_log |       a1 |       a2 |       a3 |
&   0.087  &   0.701  &   6.263  &   0.116  &   0.913  &   1.000  &   1.000  \\

-> Done!
filter depth evaluation:

   abs_rel |   sq_rel |     rmse | rmse_log |       a1 |       a2 |       a3 |
&   0.085  &   0.673  &   6.136  &   0.114  &   0.911  &   1.000  &   1.000  \\

-> Done!
nerf novel view synthesis evaluation:
train set rgbs:
psnr: 29.838516235351562, ssim: 0.9151979680894414, lpips: 0.10798460245132446
test set rgbs:
psnr: 26.234378814697266, ssim: 0.8603718766090627, lpips: 0.1645752489566803
