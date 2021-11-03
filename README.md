
# 袋鼯麻麻——智能购物平台

## 项目背景
<font size=3 >目前在零售行业的实际运营过程中，会产生巨大的人力成本，例如导购、保洁、结算等，而其中，尤其需要花费大量的人力成本和时间成本在识别商品并对其进行价格结算的过程中，并且在此过程中，顾客也因此而需要排队等待。这样一来零售行业人力成本较大、工作效率极低，二来也使得顾客的购物体验下降。  
  
随着计算机视觉技术的发展，以及无人化、自动化超市运营理念的提出，利用图像识别技术及目标检测技术实现产品的自动识别及自动化结算的需求呼之欲出，及自动结账系统（Automatic checkout, ACO）。基于计算机视觉的自动结账系统能有效降低零售行业的运营成本，提高顾客结账效率，从而进一步提升用户在购物过程中的体验感与幸福感。  </font>

## 实现功能
<font size=3 >本项目具体实现在零售过程中对用户购买商品的自动结算。即：利用计算机视觉领域中的图像识别及目标检测技术，精准地定位顾客购买的商品，并进行智能化、自动化的价格结算。当顾客将自己选购的商品放置在制定区域的时候，“**袋鼯麻麻——智能购物平台**”能够精准地定位识别每一个商品，并且能够返回完整地购物清单及顾客应付的实际商品总价格，极大地降低零售行业实际运营过程中巨大的人力成本，提升零售行业无人化、自动化、智能化水平。  </font>  
<div style="align: center">
<img src="https://ai-studio-static-online.cdn.bcebos.com/56a6521f80754fcdb12ab433e35ce343b7a5e475b56446e8beb4d9c93213e7b3"/>
</div>

## 整体架构

![](https://ai-studio-static-online.cdn.bcebos.com/59f875129c854cdfb7cbb3435f5004c37ffed920756b41e5bde49e98c09cd0ab) 

## 技术路线
<font size=3 >**袋鼯麻麻——智能购物平台** 主要基于[PaddleClas](https://github.com/PaddlePaddle/PaddleClas)作为主要的功能开发套件，利用其开源的[图像识别技术](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.3/docs/zh_CN/tutorials/quick_start_recognition.md)，并通过PaddleInference将其部署于Jetson Nano，开发一套符合实际应用需求的智能零售购物平台。  </font>

### [图像识别](https://arxiv.org/pdf/2111.00775.pdf)介绍
![](https://ai-studio-static-online.cdn.bcebos.com/277b4a0641234484a529054791505f9f3fba0faa4b2047838ebbadc3663af0ac) 

<font size=3 >整个图像识别系统分为三步：  
（1）通过一个目标检测模型，检测图像物体候选区域；  
（2）对每个候选区域进行特征提取；  
（3）与检索库中图像进行特征匹配，提取识别结果。

对于新的未知类别，无需重新训练模型，只需要在检索库补入该类别图像，重新建立检索库，就可以识别该类别。  </font>

### 数据集介绍
【The first one】:[Products-10K Large Scale Product Recognition Dataset](https://arxiv.org/abs/2006.12634)  

【The second one】:[RP2K: A Large-Scale Retail Product Dataset for Fine-Grained Image Classification](https://www.pinlandata.com/rp2k_dataset)  

**袋鼯麻麻——智能购物平台**基于上述两个数据集，并对此两种数据集进行适应性处理。  


目前处理后的数据集已在[AIStudio](https://aistudio.baidu.com/aistudio/datasetdetail/108651)开源。 </font>


## 部署方式
本项目已打通**Jetson Nano、Windows、linux**系统

<font size=3 >
  
  - 使用[QPT](https://github.com/QPT-Family/QPT)打包的百度网盘链接：https://pan.baidu.com/s/1pVr4zSZB6qV10VIPvgWCsA  提取码：mpq2  
  
    解压后运行**启动程序.exe**即可
  
  - 服务器部署  
  
    安装python依赖库：pip install -r requestment.txt；  
  
    执行python manage.py makemigrations;  

    执行python manage.py migrate;  
  
    执行python manage.py runserver # 默认运行在8000端口  

  - 微信小程序
    打开开发者工具，导入系统文件夹下wx_mini_app文件夹并运行，即可运行小程序端；  </font>

## [bilibili](https://www.bilibili.com/video/BV19q4y1G7bx#reply5654379507)效果演示

