 # PPACO——智能零售结算平台

## 项目介绍

### 背景简介

   目前，在零售行业的实际运营过程中，会产生巨大的人力成本，例如导购、保洁、结算等，而其中，尤其需要花费大量的人力成本和时间成本在商品识别并对其进行价格结算的过程中，并且在此过程中，顾客也因此而需要排队等待。这样一来零售行业人力成本较大、工作效率极低，二来也使得顾客的购物体验下降。 

   随着计算机视觉技术的发展，以及无人化、自动化超市运营理念的提出，利用图像识别技术及目标检测技术实现产品的自动识别及自动化结算的需求呼之欲出，即自动结账系统（Automatic checkout, ACO）。基于计算机视觉的自动结账系统能有效降低零售行业的运营成本，提高顾客结账效率，从而进一步提升用户在购物过程中的体验感与幸福感。  
<div style="align: center">
<img src="https://ai-studio-static-online.cdn.bcebos.com/fa2920c589d247a4a3657bdc141fb3838b3c208d19ce454297420ee1e91bef17">
</div>

### 适用场景  

 >**袋鼯麻麻——智能零售结算平台**致力于为**大型线下零售体验店**提供基于视觉的零售结算方案。

### 痛点问题

   1. **结算效率要求极高**：在大型线下零售体验店购物场景中，若顾客购买的商品较多，采用传统的条形码结算，效率较低，顾客购物体验较差；
   2. **不同商品相似度极高**：比如同一种饮料的不同口味，就很可能拥有非常类似的包装。而且即便对于同一件商品，**在不同情况下所获得的商品图像都往往存在相当大的差异**；
   3. **品类更新极快**：像新零售这种行业，新品几乎都是**按小时级别**在更新，每增加新的产品都需要重新训练模型，仅靠单一模型想要跟上步伐，着实望尘莫及；
   4. **商品类别数以万计**：根本没法事先把所有类别都放入训练集。

### 解决方案

   “**袋鼯麻麻——智能零售结算平台**”具体实现在零售过程中对用户购买商品的自动结算。即：利用PaddleClas团队开源的[图像识别PP-ShiTu](https://arxiv.org/pdf/2111.00775.pdf)技术，精准地定位顾客购买的商品，并进行智能化、自动化的价格结算。当顾客将自己选购的商品放置在制定区域内时，“**袋鼯麻麻——智能零售结算平台**”能够精准地定位识别每一个商品，并且能够返回完整的购物清单及顾客应付的实际商品总价格，整个结算过程控制在**4秒钟**以内，极大地提高了结算效率。而当系统有新商品增加时，本系统只需**30秒不到**的时间更新检索库即可，无需重新训练模型，能够做到“即增即用”，大大提高了人工智能在线下零售行业的应用落地可能性。   

### 模型工具简介

   飞桨图像识别套件PaddleClas是飞桨为工业界和学术界所准备的一个图像识别任务的工具集，助力使用者训练出更好的视觉模型和应用落地。  

   而[PP-ShiTu](https://arxiv.org/pdf/2111.00775.pdf)是一个实用的轻量级通用图像识别系统，主要由主体检测、特征学习和向量检索三个模块组成。该系统从骨干网络选择和调整、损失函数的选择、数据增强、学习率变换策略、正则化参数选择、预训练模型使用以及模型裁剪量化8个方面，采用多种策略，对各个模块的模型进行优化，最终得到在CPU上仅0.2s即可完成10w+库的图像识别的系统。

## 安装说明

### 环境要求
- Python >= 3.6
- PaddlePaddle >= 2.1
- Linux 环境最佳

- 安装PaddleClas

    - 克隆 PaddleClas
    ```shell
    git clone https://github.com/PaddlePaddle/PaddleClas.git -b release/2.3
    ```
    - 安装 Python 依赖库  
    PaddleClas 的 Python 依赖库在 `requirements.txt` 中给出，可通过如下命令安装：
    ```shell
    pip install --upgrade -r requirements.txt -i https://mirror.baidu.com/pypi/simple


```python
!git clone https://github.com/PaddlePaddle/PaddleClas.git -b release/2.3
```


```python
%cd PaddleClas/
!pip install --upgrade -r requirements.txt -i https://mirror.baidu.com/pypi/simple
```

## 数据准备

### 数据集介绍<详细讲述>

1. 目前开源的商品识别方向的数据集

- [Products-10K Large Scale Product Recognition Dataset](https://www.kaggle.com/c/products-10k/data?select=train.csv) :数据集中的所有图片均来自京东商城。数据集中共包含 1 万个经常购买的 SKU。所有 SKU 组织成一个层次结构。总共有近 19 万张图片。在实际应用场景中，图像量的分布是不均衡的。所有图像都由生产专家团队手工检查/标记。

- [RP2K: A Large-Scale Retail Product Dataset for Fine-Grained Image Classification](https://arxiv.org/abs/2006.12634) :收集了超过 500,000 张货架上零售产品的图像，属于 2000 种不同的产品。所有图片均在实体零售店人工拍摄，自然采光，符合实际应用场景。

2. 本项目数据集基于上述数据集，结合图片爬虫等数据集搜索方式，开源了一份更符合本项目实际应用背景的demo数据集，目前已在[AIStudio平台](https://aistudio.baidu.com/aistudio/datasetdetail/113685)开源。此数据集总计覆盖商品357类，涵盖包括厨房用品、日用品、饮料等生活日常购买商品，商品类别细粒度较高，涉及诸如同一品牌的不同规格商品、同一品类的不同品牌商品等，能够模拟实际购物场景下的购买需求。

### 商品部分list

> 	东古酱油一品鲜  
	东古黄豆酱750G  
	东鹏特饮罐装  
	中华（硬）  
	中华（软）  
	乳酸菌600亿_2  
	乳酸菌600亿_3  
	乳酸菌600亿原味  
	乳酸菌600亿芒果  
	乳酸菌600亿芦荟  
   ...
   
### 数据集格式
* 训练集合（train dataset）：用来训练模型，使模型能够学习该集合的图像特征。
* 底库数据集合（gallery dataset）：用来提供图像检索任务中的底库数据，该集合可与训练集或测试集相同，也可以不同，当与训练集相同时，测试集的类别体系应与训练集的类别体系相同。
* 测试数据集合（query dataset）：用来测试模型的好坏，通常要对测试集的每一张测试图片进行特征提取，之后和底库数据的特征进行距离匹配，得到识别结果，后根据识别结果计算整个测试集的指标。

训练集、底库数据集和测试数据集均使用 `txt` 文件指定，训练数据集 `train_list.txt`文件内容格式如下所示：

```shell
# 采用"空格"作为分隔符号
...
train/10/1283.jpg 10 624
train/10/1284.jpg 10 625
train/10/1285.jpg 10 626
train/10/1286.jpg 10 627
...
```
验证数据集(本数据集中既是 gallery dataset，也是 query dataset)test_list.txt 文件内容格式如下所示：
```shell
...
test/103/743.jpg 103 743
test/103/744.jpg 103 744
test/103/745.jpg 103 745
test/103/746.jpg 103 746
...
```

**注：**
1. 每行数据使用“空格”分割，三列数据的含义分别是训练数据的路径、训练数据的label信息、训练数据的unique id;
2. 本数据集中由于 gallery dataset 和 query dataset 相同，为了去掉检索得到的第一个数据（检索图片本身无须评估），每个数据需要对应一个 unique id（每张图片的 id 不同即可，可以用行号来表示 unique id），用于后续评测 mAP、recall@1 等指标。yaml 配置文件的数据集选用 VeriWild。

## 模型选择

### 主体检测  

主体检测技术是目前应用非常广泛的一种检测技术，它指的是检测出图片中一个或者多个主体的坐标位置，然后将图像中的对应区域裁剪下来，进行识别，从而完成整个识别过程。主体检测是识别任务的前序步骤，可以有效提升识别精度。  

考虑到商品识别实际应用场景中，需要快速准确地获得识别结果，故本项目选取**轻量级主体检测模型**[PicoDet](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/models/pretrain/picodet_PPLCNet_x2_5_mainbody_lite_v1.0_pretrained.pdparams)作为本项目主体检测部分的模型。此模型inference模型大小(MB)仅**30.1MB**，mAP可达**40.1%**，在**cpu**下单张图片预测耗时仅**29.8ms**【具体模型参评标准请见[PicoDet系列模型介绍](https://github.com/PaddlePaddle/PaddleDetection/blob/develop/configs/picodet/README.md)】，完美符合本项目实际落地需求。

### 特征提取 

特征提取是图像识别中的关键一环，它的作用是将输入的图片转化为固定维度的特征向量，用于后续的向量检索。好的特征需要具备相似度保持性，即在特征空间中，相似度高的图片对其特征相似度要比较高（距离比较近），相似度低的图片对，其特征相似度要比较小（距离比较远）。Deep Metric Learning用以研究如何通过深度学习的方法获得具有强表征能力的特征。  

考虑到本项目的真实落地的场景中,推理速度及预测准确率是考量模型好坏的重要指标，所以本项目采用 [PP_LCNet_x2_5](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.3/docs/zh_CN/models/PP-LCNet.md) 作为骨干网络，并结合度量学习**arcmargin**算法，对高相似物体的区分效果远超单一模型，能更好地适应 Intel CPU，不仅准确率超越大模型ResNet50，预测速度还能快3倍。  

### 向量检索  

向量检索技术在图像识别、图像检索中应用比较广泛。其主要目标是，对于给定的查询向量，在已经建立好的向量库中，与库中所有的待查询向量，进行特征向量的相似度或距离计算，得到相似度排序。在图像识别系统中，本项目使用 [Faiss](https://github.com/facebookresearch/faiss) 对此部分进行支持。在此过程中，本项目选取 **HNSW32** 为检索算法，使得检索精度、检索速度能够取得较好的平衡，更为贴切本项目实际应用场景的使用需求。

## 模型训练
这里主要介绍特征提取部分的模型训练，其余部分详情请参考[PaddleClas](https://github.com/PaddlePaddle/PaddleClas)。
### 训练流程

1. 数据准备
   
首先，需要基于任务定制自己的数据集。数据集格式参见格式说明。在启动模型训练之前，需要在配置文件中修改数据配置相关的内容, 主要包括数据集的地址以及类别数量。对应到配置文件中的位置如下所示：

```
  Head:
    name: ArcMargin
    embedding_size: 512
    class_num: 358    #此处表示类别数
```
```
  Train:
    dataset:
      name: ImageNetDataset
      image_root: /home/aistudio/dataset/ #此处表示train数据所在的目录
      cls_label_path: /home/aistudio/dataset/train_list.txt  #此处表示train数据集label文件的地址
```
```
   Query:
     dataset:
       name: VeriWild
       image_root: /home/aistudio/dataset/	#此处表示query数据集所在的目录
       cls_label_path: /home/aistudio/dataset/test_list.txt #此处表示query数据集label文件的地址
```
```
   Gallery:
     dataset:
       name: VeriWild
       image_root: /home/aistudio/dataset/	#此处表示gallery数据集所在的目录
       cls_label_path: /home/aistudio/dataset/test_list.txt   #此处表示gallery数据集label文件的地址
```

2. 模型训练  



- 单机单卡训练


```python
!python tools/train.py \
    -c ./ppcls/configs/GeneralRecognition/GeneralRecognition_PPLCNet_x2_5.yaml \
    -o Arch.Backbone.pretrained=True \
    -o Global.device=gpu
```

### 模型优化思路

- batch_size 是训练神经网络中的一个重要的超参数，该值决定了一次将多少数据送入神经网络参与训练。而通过实验发现，当 batch_size 的值与学习率的值呈线性关系时，收敛精度几乎不受影响。而PP=shitu在训练数据时，初始学习率为 0.1，batch_size 是 256，所以根据AIStudio平台实际硬件情况，可以将学习率设置为0.0025, batch_size 设置为 48。

```
Optimizer:
  name: Momentum
  momentum: 0.9
  lr:
    name: Cosine
    learning_rate: 0.0025
    warmup_epoch: 5
  regularizer:
    name: 'L2'
    coeff: 0.00001
```
```
  Train:
    dataset:
      name: ImageNetDataset
      image_root: /home/aistudio/dataset/
      cls_label_path: /home/aistudio/dataset/train_list.txt
      transform_ops:
        - DecodeImage:
            to_rgb: True
            channel_first: False
        - RandCropImage:
            size: 224
        - RandFlipImage:
            flip_code: 1
        - NormalizeImage:
            scale: 1.0/255.0
            mean: [0.485, 0.456, 0.406]
            std: [0.229, 0.224, 0.225]
            order: ''

    sampler:
      name: DistributedBatchSampler
      batch_size: 48
      drop_last: False
      shuffle: True
    loader:
      num_workers: 4
      use_shared_memory: True

```
```
Eval:
    Query:
      dataset: 
        name: ImageNetDataset
        image_root: /home/aistudio/dataset/
        cls_label_path: /home/aistudio/dataset/test_list.txt
        transform_ops:
          - DecodeImage:
              to_rgb: True
              channel_first: False
          - ResizeImage:
              size: 224
          - NormalizeImage:
              scale: 0.00392157
              mean: [0.485, 0.456, 0.406]
              std: [0.229, 0.224, 0.225]
              order: ''
      sampler:
        name: DistributedBatchSampler
        batch_size: 16
        drop_last: False
        shuffle: False
      loader:
        num_workers: 4
        use_shared_memory: True


```
```
    Gallery:
      dataset: 
        name: ImageNetDataset
        image_root: /home/aistudio/dataset/
        cls_label_path: /home/aistudio/dataset/test_list.txt
        transform_ops:
          - DecodeImage:
              to_rgb: True
              channel_first: False
          - ResizeImage:
              size: 224
          - NormalizeImage:
              scale: 0.00392157
              mean: [0.485, 0.456, 0.406]
              std: [0.229, 0.224, 0.225]
              order: ''
      sampler:
        name: DistributedBatchSampler
        batch_size: 16
        drop_last: False
        shuffle: False
      loader:
        num_workers: 4
        use_shared_memory: True

```

### 调参方案及结果

训练完成之后，会在`output`目录下生成`best_model`模型文件。 

3. 模型评估

- 单卡评估

```python
!python tools/eval.py -c ./ppcls/configs/GeneralRecognition/GeneralRecognition_PPLCNet_x2_5.yaml -o Global.pretrained_model="output/RecModel/best_model"
```
评估部分log如下：
```
[2022/01/08 12:59:04] root INFO: Build query done, all feat shape: [25738, 512], begin to eval..
/opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/framework.py:744: DeprecationWarning: `np.bool` is a deprecated alias for the builtin `bool`. To silence this warning, use `bool` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.bool_` here.
Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
  elif dtype == np.bool:
[2022/01/08 12:59:05] root INFO: [Eval][Epoch 0][Avg]recall1: 0.98368, recall5: 0.99137
```
可见recall1为0.98368，能够符合实际产业场景应用需求。

4. 模型推理

推理过程包括两个步骤： 1)导出推理模型, 2)获取特征向量

- 导出推理模型
PaddlePaddle框架保存的权重文件分为两种：支持前向推理和反向梯度的**训练模型** 和 只支持前向推理的**推理模型**。二者的区别是推理模型针对推理速度和显存做了优化，裁剪了一些只在训练过程中才需要的tensor，降低显存占用，并进行了一些类似层融合，kernel选择的速度优化。因此可执行如下命令导出推理模型。

```python
!python tools/export_model -c ppcls/configs/ResNet50_vd_SOP.yaml -o Global.pretrained_model="output/RecModel/best_model"
```
生成的推理模型位于 inference 目录，里面包含三个文件，分别为 inference.pdmodel、inference.pdiparams、inference.pdiparams.info。 其中: inference.pdmodel 用来存储推理模型的结构, inference.pdiparams 和 inference.pdiparams.info 用来存储推理模型相关的参数信息。

- 获取特征向量

```python
%cd deploy
!python python/predict_rec.py -c configs/inference_rec.yaml  -o Global.rec_inference_model_dir="../inference"
```
得到的特征输出格式如下log所示：
```

```


### 测试代码

这里串联主体检测、特征提取、向量检索，从而构成一整套图像识别系统：

1. 建立索引库
```python
# 建立新的索引库
!python3.7 python/build_gallery.py \
    -c configs/build_general.yaml \
    -o IndexProcess.data_file="/home/aistudio/dataset/data_file.txt" \
    -o IndexProcess.index_dir="/home/aistudio/dataset/index_inference"
```
2. 识别单张图片
运行下面的命令，对图像 /home/aistudio/dataset/sijibao.jpg 进行识别与检索
```python
#基于索引库的图像识别
!python3.7 python/predict_system.py \
    -c configs/inference_general.yaml \
    -o Global.infer_imgs="/home/aistudio/dataset/sijibao.jpg" \
    -o IndexProcess.index_dir="/home/aistudio/dataset/index_inference"
```
最终输出结果如下：
```
Inference: 31.720638275146484 ms per batch image
[{'bbox': [0, 0, 500, 375], 'rec_docs': '四季宝花生酱', 'rec_scores': 0.79656786}]

```
其中 bbox 表示检测出的主体所在位置，rec_docs 表示索引库中与检测框最为相似的类别，rec_scores 表示对应的置信度。  
检测的可视化结果也保存在 output 文件夹下，对于本张图像，识别结果可视化如下所示：
![]()

### 测试效果图
以下为参与模型训练的商品的测试效果图：
![](../image/recognition_3.png)
以下为未参与模型训练的商品的测试效果图：
![](../image/recognition_2.jpg)


## 模型部署
在项目中为用户提供了基于服务器的部署Demo方案。用户可根据实际情况自行参考。  

![](https://github.com/thomas-yanxin/Smart_container/raw/master/image/main.png)
![](https://github.com/thomas-yanxin/Smart_container/raw/master/image/recognition_1.png)
![](https://github.com/thomas-yanxin/Smart_container/raw/master/image/wx_all.png)
具体部署方式可以参考：[袋鼯麻麻——智能购物平台](https://github.com/thomas-yanxin/Smart_container)  


<!-- 1. 行业场景痛点和解决方案说明
2. 文档用词规范化
3. 数据集格式规划
4. 100% 准确率  - 再看一下是什么原因
5. 新入库- 新加入的商品品类如何更新索引库 效果如何
6. 优化思路和策略 - 优化方案如果没有特别好的思路可以若干
7. 部署方案的展示 - 速度如何 硬件参数对比 -->
