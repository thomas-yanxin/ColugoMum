<!-- # 袋鼯麻麻——智能零售购物平台

## 项目介绍

### 背景简介

   目前，在零售行业的实际运营过程中，会产生巨大的人力成本，例如导购、保洁、结算等，而其中，尤其需要花费大量的人力成本和时间成本在商品识别并对其进行价格结算的过程中，并且在此过程中，顾客也因此而需要排队等待。这样一来零售行业人力成本较大、工作效率极低，二来也使得顾客的购物体验下降。 

   随着计算机视觉技术的发展，以及无人化、自动化超市运营理念的提出，利用图像识别技术及目标检测技术实现产品的自动识别及自动化结算的需求呼之欲出，即自动结账系统（Automatic checkout, ACO）。基于计算机视觉的自动结账系统能有效降低零售行业的运营成本，提高顾客结账效率，从而进一步提升用户在购物过程中的体验感与幸福感。  

### 适用场景  

> 大型线下零售体验店

### 通点问题

   1. **结算效率要求极高**：在大型线下零售体验店购物场景中，若顾客购买的商品较多，采用传统的条形码结算，效率较低，顾客购物体验较差；
   2. **品类更新极快**：像新零售这种行业，新品几乎都是**按小时级别**在更新，每增加新的产品都要辛辛苦苦重新训练模型，仅靠单一模型想要跟上步伐，着实望尘莫及；
   3. **不同商品相似度极高**：比如同一种饮料的不同口味，就很可能拥有非常类似的包装。而且即便对于同一件商品，**在不同情况下所获得的商品图像都往往存在相当大的差异**；
   4. **商品类别数以万计**：根本没法事先把所有类别都放入训练集。


### 解决方案

   “**袋鼯麻麻——智能购物平台**”具体实现在零售过程中对用户购买商品的自动结算。即：利用PaddleClas团队开源的[图像识别PP-ShiTu](https://arxiv.org/pdf/2111.00775.pdf)技术，精准地定位顾客购买的商品，并进行智能化、自动化的价格结算。当顾客将自己选购的商品放置在制定区域内时，“**袋鼯麻麻——智能购物平台**”能够精准地定位识别每一个商品，并且能够返回完整的购物清单及顾客应付的实际商品总价格。而当系统有新商品增加时，本系统只需更新检索库即可，无需重新训练模型。   

### 模型工具简介

   飞桨图像识别套件PaddleClas是飞桨为工业界和学术界所准备的一个图像识别任务的工具集，助力使用者训练出更好的视觉模型和应用落地。  

   而[PP-ShiTu](https://arxiv.org/pdf/2111.00775.pdf)是一个实用的轻量级通用图像识别系统，主要由主体检测、特征学习和向量检索三个模块组成。该系统从骨干网络选择和调整、损失函数的选择、数据增强、学习率变换策略、正则化参数选择、预训练模型使用以及模型裁剪量化8个方面，采用多种策略，对各个模块的模型进行优化，最终得到在CPU上仅0.2s即可完成10w+库的图像识别的系统。

---

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
---

## 数据准备

### 数据集介绍

目前开源的商品识别方向的数据集

- [Products-10K Large Scale Product Recognition Dataset](https://www.kaggle.com/c/products-10k/data?select=train.csv) :数据集中的所有图片均来自京东商城。数据集中共包含 1 万个经常购买的 SKU。所有 SKU 组织成一个层次结构。总共有近 19 万张图片。在实际应用场景中，图像量的分布是不均衡的。所有图像都由生产专家团队手工检查/标记。

- [RP2K: A Large-Scale Retail Product Dataset for Fine-Grained Image Classification](https://arxiv.org/abs/2006.12634) :收集了超过 500,000 张货架上零售产品的图像，属于 2000 种不同的产品。所有图片均在实体零售店人工拍摄，自然采光，符合实际应用场景。

本项目数据集以上述数据集为背景，结合图片爬虫等，开源了一份新的更符合本项目背景的数据集，目前已在[AIStudio平台](https://aistudio.baidu.com/aistudio/datasetdetail/113685)开源.

#### 商品部分list

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
   
#### 数据集格式
* 训练集合（train dataset）：用来训练模型，使模型能够学习该集合的图像特征。
* 底库数据集合（gallery dataset）：用来提供图像检索任务中的底库数据，该集合可与训练集或测试集相同，也可以不同，当与训练集相同时，测试集的类别体系应与训练集的类别体系相同。
* 测试数据集合（query dataset）：用来测试模型的好坏，通常要对测试集的每一张测试图片进行特征提取，之后和底库数据的特征进行距离匹配，得到识别结果，后根据识别结果计算整个测试集的指标。

训练集、底库数据集和测试数据集均使用 `txt` 文件指定，训练数据集 `train_list.txt` 文件内容格式如下所示：

```shell
# 采用"空格"作为分隔符号
...
train/99/Ovenbird_0136_92859.jpg 99 2
...
train/99/Ovenbird_0128_93366.jpg 99 6
...
```


每行数据使用“空格”分割，三列数据的含义分别是训练数据的路径、训练数据的label信息、训练数据的unique id。

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

### 训练流程

1. 数据准备
   
首先，需要基于任务定制自己的数据集。数据集格式参见格式说明。在启动模型训练之前，需要在配置文件中修改数据配置相关的内容, 主要包括数据集的地址以及类别数量。对应到配置文件中的位置如下所示：

```
  Head:
    name: ArcMargin
    embedding_size: 512
    class_num: 185341    #此处表示类别数
```
```
  Train:
    dataset:
      name: ImageNetDataset
      image_root: ./dataset/     #此处表示train数据所在的目录
      cls_label_path: ./dataset/train_reg_all_data.txt  #此处表示train数据集label文件的地址
```
```
   Query:
     dataset:
       name: VeriWild
       image_root: ./dataset/Aliproduct/.    #此处表示query数据集所在的目录
       cls_label_path: ./dataset/Aliproduct/val_list.txt.    #此处表示query数据集label文件的地址
```
```
   Gallery:
     dataset:
       name: VeriWild
       image_root: ./dataset/Aliproduct/    #此处表示gallery数据集所在的目录
       cls_label_path: ./dataset/Aliproduct/val_list.txt.   #此处表示gallery数据集label文件的地址
```
   
2. 模型训练  

- 单机单卡训练
```
python tools/train.py -c ppcls/configs/ResNet50_vd_SOP.yaml
```
- 单机多卡训练
```
python -m paddle.distributed.launch 
    --gpus="0,1,2,3" tools/train.py 
    -c ppcls/configs/ResNet50_vd_SOP.yaml
```
训练完成之后，会在`output`目录下生成`best_model`模型文件。 

3. 模型评估

- 单卡评估
```
python tools/eval.py -c ppcls/configs/ResNet50_vd_SOP.yaml -o Global.pretrained_model="output/RecModel/best_model"
```
- 多卡评估
```
python -m paddle.distributed.launch 
    --gpus="0,1,2,3" tools/eval.py 
    -c  ppcls/configs/ResNet50_vd_SOP.yaml
    -o  Global.pretrained_model="output/RecModel/best_model"
```

4. 模型推理

推理过程包括两个步骤： 1)导出推理模型, 2)获取特征向量
- 导出推理模型
```
python tools/export_model -c ppcls/configs/ResNet50_vd_SOP.yaml -o Global.pretrained_model="output/RecModel/best_model"
```
生成的推理模型位于`inference`目录，名字为`inference.pd*`

- 获取特征向量
```
cd deploy
python python/predict_rec.py -c configs/inference_rec.yaml  -o Global.rec_inference_model_dir="../inference"
```

### bsseline指标结果

### 测试代码

### 测试效果图

## 模型优化

### 模型优化思路

### 调参方案及结果

### 最优模型方案

## 模型部署

<!-- ### 1. 简介
[Paddle Serving](https://github.com/PaddlePaddle/Serving) 旨在帮助深度学习开发者轻松部署在线预测服务，支持一键部署工业级的服务能力、客户端和服务端之间高并发和高效通信、并支持多种编程语言开发客户端。

该部分以 HTTP 预测服务部署为例，介绍怎样在 PaddleClas 中使用 PaddleServing 部署模型服务。


### 2. Serving安装

Serving 官网推荐使用 docker 安装并部署 Serving 环境。首先需要拉取 docker 环境并创建基于 Serving 的 docker。

```shell
nvidia-docker pull hub.baidubce.com/paddlepaddle/serving:0.2.0-gpu
nvidia-docker run -p 9292:9292 --name test -dit hub.baidubce.com/paddlepaddle/serving:0.2.0-gpu
nvidia-docker exec -it test bash
```

进入 docker 后，需要安装 Serving 相关的 python 包。

```shell
pip install paddlepaddle-gpu
pip install paddle-serving-client
pip install paddle-serving-server-gpu
pip install paddle-serving-app
```

* 如果安装速度太慢，可以通过 `-i https://pypi.tuna.tsinghua.edu.cn/simple` 更换源，加速安装过程。

* 如果希望部署 CPU 服务，可以安装 serving-server 的 cpu 版本，安装命令如下。

```shell
pip install paddle-serving-server

```

## 4.图像识别服务部署
使用PaddleServing做服务化部署时，需要将保存的inference模型转换为Serving模型。 下面以PP-ShiTu中的超轻量图像识别模型为例，介绍图像识别服务的部署。
## 4.1 模型转换
- 下载通用检测inference模型和通用识别inference模型
```
cd deploy
# 下载并解压通用识别模型
wget -P models/ https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/models/inference/general_PPLCNet_x2_5_lite_v1.0_infer.tar
cd models
tar -xf general_PPLCNet_x2_5_lite_v1.0_infer.tar
# 下载并解压通用检测模型
wget https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/models/inference/picodet_PPLCNet_x2_5_mainbody_lite_v1.0_infer.tar
tar -xf picodet_PPLCNet_x2_5_mainbody_lite_v1.0_infer.tar
```
- 转换识别inference模型为Serving模型：
```
# 转换识别模型
python3 -m paddle_serving_client.convert --dirname ./general_PPLCNet_x2_5_lite_v1.0_infer/ \
                                         --model_filename inference.pdmodel  \
                                         --params_filename inference.pdiparams \
                                         --serving_server ./general_PPLCNet_x2_5_lite_v1.0_serving/ \
                                         --serving_client ./general_PPLCNet_x2_5_lite_v1.0_client/
```
识别推理模型转换完成后，会在当前文件夹多出`general_PPLCNet_x2_5_lite_v1.0_serving/` 和`general_PPLCNet_x2_5_lite_v1.0_serving/`的文件夹。修改`general_PPLCNet_x2_5_lite_v1.0_serving/`目录下的serving_server_conf.prototxt中的alias名字： 将`fetch_var`中的`alias_name`改为`features`。
修改后的serving_server_conf.prototxt内容如下：
```
feed_var {
  name: "x"
  alias_name: "x"
  is_lod_tensor: false
  feed_type: 1
  shape: 3
  shape: 224
  shape: 224
}
fetch_var {
  name: "save_infer_model/scale_0.tmp_1"
  alias_name: "features"
  is_lod_tensor: true
  fetch_type: 1
  shape: -1
}
```
- 转换通用检测inference模型为Serving模型：
```
# 转换通用检测模型
python3 -m paddle_serving_client.convert --dirname ./picodet_PPLCNet_x2_5_mainbody_lite_v1.0_infer/ \
                                         --model_filename inference.pdmodel  \
                                         --params_filename inference.pdiparams \
                                         --serving_server ./picodet_PPLCNet_x2_5_mainbody_lite_v1.0_serving/ \
                                         --serving_client ./picodet_PPLCNet_x2_5_mainbody_lite_v1.0_client/
```
检测inference模型转换完成后，会在当前文件夹多出`picodet_PPLCNet_x2_5_mainbody_lite_v1.0_serving/` 和`picodet_PPLCNet_x2_5_mainbody_lite_v1.0_client/`的文件夹。

**注意:** 此处不需要修改`picodet_PPLCNet_x2_5_mainbody_lite_v1.0_serving/`目录下的serving_server_conf.prototxt中的alias名字。

- 下载并解压已经构建后的检索库index
```
cd ../
wget https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/data/drink_dataset_v1.0.tar && tar -xf drink_dataset_v1.0.tar
```
## 4.2 服务部署和请求
**注意:**  识别服务涉及到多个模型，出于性能考虑采用PipeLine部署方式。Pipeline部署方式当前不支持windows平台。
- 进入到工作目录
```shell
cd ./deploy/paddleserving/recognition
```
paddleserving目录包含启动pipeline服务和发送预测请求的代码，包括：
```
__init__.py
config.yml                    # 启动服务的配置文件
pipeline_http_client.py       # http方式发送pipeline预测请求的脚本
pipeline_rpc_client.py        # rpc方式发送pipeline预测请求的脚本
recognition_web_service.py    # 启动pipeline服务端的脚本
```
- 启动服务：
```
# 启动服务，运行日志保存在log.txt
python3 recognition_web_service.py &>log.txt &
```
成功启动服务后，log.txt中会打印类似如下日志
![](https://github.com/PaddlePaddle/PaddleClas/raw/release/2.3/deploy/paddleserving/imgs/start_server_shitu.png)

- 发送请求：
```
python3 pipeline_http_client.py
```
成功运行后，模型预测的结果会打印在cmd窗口中，结果示例为：
![](https://github.com/PaddlePaddle/PaddleClas/raw/release/2.3/deploy/paddleserving/imgs/results_shitu.png) 


在项目中为用户提供了基于服务器的部署Demo方案。用户可根据实际情况自行参考。  

![](https://github.com/thomas-yanxin/Smart_container/raw/master/image/main.png)
![](https://github.com/thomas-yanxin/Smart_container/raw/master/image/recognition_1.png)
![](https://github.com/thomas-yanxin/Smart_container/raw/master/image/wx_all.png)
具体部署方式可以参考：[袋鼯麻麻——智能购物平台](https://github.com/thomas-yanxin/Smart_container)  
 -->
 
 
 
 # 袋鼯麻麻——智能零售购物平台

## 项目介绍

### 背景简介

   目前，在零售行业的实际运营过程中，会产生巨大的人力成本，例如导购、保洁、结算等，而其中，尤其需要花费大量的人力成本和时间成本在商品识别并对其进行价格结算的过程中，并且在此过程中，顾客也因此而需要排队等待。这样一来零售行业人力成本较大、工作效率极低，二来也使得顾客的购物体验下降。 

   随着计算机视觉技术的发展，以及无人化、自动化超市运营理念的提出，利用图像识别技术及目标检测技术实现产品的自动识别及自动化结算的需求呼之欲出，即自动结账系统（Automatic checkout, ACO）。基于计算机视觉的自动结账系统能有效降低零售行业的运营成本，提高顾客结账效率，从而进一步提升用户在购物过程中的体验感与幸福感。  
<div style="align: center">
<img src="https://ai-studio-static-online.cdn.bcebos.com/fa2920c589d247a4a3657bdc141fb3838b3c208d19ce454297420ee1e91bef17">
</div>

### 适用场景  

 >**袋鼯麻麻——智能零售购物平台**致力于为**大型线下零售体验店**提供基于视觉的零售结算方案。

### 通点问题

   1. **结算效率要求极高**：在大型线下零售体验店购物场景中，若顾客购买的商品较多，采用传统的条形码结算，效率较低，顾客购物体验较差；
   2. **品类更新极快**：像新零售这种行业，新品几乎都是**按小时级别**在更新，每增加新的产品都要辛辛苦苦重新训练模型，仅靠单一模型想要跟上步伐，着实望尘莫及；
   3. **不同商品相似度极高**：比如同一种饮料的不同口味，就很可能拥有非常类似的包装。而且即便对于同一件商品，**在不同情况下所获得的商品图像都往往存在相当大的差异**；
   4. **商品类别数以万计**：根本没法事先把所有类别都放入训练集。

### 解决方案

   “**袋鼯麻麻——智能购物平台**”具体实现在零售过程中对用户购买商品的自动结算。即：利用PaddleClas团队开源的[图像识别PP-ShiTu](https://arxiv.org/pdf/2111.00775.pdf)技术，精准地定位顾客购买的商品，并进行智能化、自动化的价格结算。当顾客将自己选购的商品放置在制定区域内时，“**袋鼯麻麻——智能购物平台**”能够精准地定位识别每一个商品，并且能够返回完整的购物清单及顾客应付的实际商品总价格。而当系统有新商品增加时，本系统只需更新检索库即可，无需重新训练模型。   

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

### 数据集介绍

目前开源的商品识别方向的数据集

- [Products-10K Large Scale Product Recognition Dataset](https://www.kaggle.com/c/products-10k/data?select=train.csv) :数据集中的所有图片均来自京东商城。数据集中共包含 1 万个经常购买的 SKU。所有 SKU 组织成一个层次结构。总共有近 19 万张图片。在实际应用场景中，图像量的分布是不均衡的。所有图像都由生产专家团队手工检查/标记。

- [RP2K: A Large-Scale Retail Product Dataset for Fine-Grained Image Classification](https://arxiv.org/abs/2006.12634) :收集了超过 500,000 张货架上零售产品的图像，属于 2000 种不同的产品。所有图片均在实体零售店人工拍摄，自然采光，符合实际应用场景。

本项目数据集以上述数据集为背景，结合图片爬虫等，开源了一份新的更符合本项目背景的数据集，目前已在[AIStudio平台](https://aistudio.baidu.com/aistudio/datasetdetail/113685)开源。(欢迎各位开发者对数据集进行补充扩展！)

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

训练集、底库数据集和测试数据集均使用 `txt` 文件指定，训练数据集 `train_list.txt` 文件内容格式如下所示：

```shell
# 采用"空格"作为分隔符号
...
train/99/Ovenbird_0136_92859.jpg 99 2
...
train/99/Ovenbird_0128_93366.jpg 99 6
...
```

每行数据使用“空格”分割，三列数据的含义分别是训练数据的路径、训练数据的label信息、训练数据的unique id。

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

4. 模型推理

推理过程包括两个步骤： 1)导出推理模型, 2)获取特征向量


- 导出推理模型



```python
!python tools/export_model -c ppcls/configs/ResNet50_vd_SOP.yaml -o Global.pretrained_model="output/RecModel/best_model"
```


生成的推理模型位于`inference`目录，名字为`inference.pd*`

- 获取特征向量



```python
%cd deploy
!python python/predict_rec.py -c configs/inference_rec.yaml  -o Global.rec_inference_model_dir="../inference"
```

### 测试代码


```python
# 建立新的索引库
!python3.7 python/build_gallery.py \
    -c configs/build_general.yaml \
    -o IndexProcess.data_file="/home/aistudio/dataset/data_file.txt" \
    -o IndexProcess.index_dir="/home/aistudio/dataset/index_inference"
```


```python
#基于索引库的图像识别
!python3.7 python/predict_system.py \
    -c configs/inference_general.yaml \
    -o Global.infer_imgs="/home/aistudio/dataset/jianjiao.jpg" \
    -o IndexProcess.index_dir="/home/aistudio/dataset/index_inference"
```

### 测试效果图

![](https://ai-studio-static-online.cdn.bcebos.com/26b32cf71c174b429626f8e3423267749bae3790d0344cf8bcb9a33ddd7a644c)


## 模型部署
在项目中为用户提供了基于服务器的部署Demo方案。用户可根据实际情况自行参考。  

![](https://github.com/thomas-yanxin/Smart_container/raw/master/image/main.png)
![](https://github.com/thomas-yanxin/Smart_container/raw/master/image/recognition_1.png)
![](https://github.com/thomas-yanxin/Smart_container/raw/master/image/wx_all.png)
具体部署方式可以参考：[袋鼯麻麻——智能购物平台](https://github.com/thomas-yanxin/Smart_container)  
