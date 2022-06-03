</div>  
<div align="center">
<img src="./image/logo2.svg" width = "720" height = "320"/>
</div>  
 


![GitHub Repo stars](https://img.shields.io/github/stars/thomas-yanxin/Smart_container)
![GitHub forks](https://img.shields.io/github/forks/thomas-yanxin/Smart_container)
![GitHub](https://img.shields.io/github/license/thomas-yanxin/Smart_container)
![python](https://img.shields.io/badge/python-3.79+-aff.svg)
![os](https://img.shields.io/badge/os-win%2C%20linux(Nano)-pink.svg)
![contributors](https://img.shields.io/github/contributors/thomas-yanxin/Smart_container)  
 English | [简体中文](./README_CN.md) | [gitee 支持国产](https://gitee.com/E_Light/Smart_container) | [github](https://github.com/thomas-yanxin/Smart_container)

> Remember to give it a ![T_ZKW6KJ_X{% %P_AY$`( X](https://user-images.githubusercontent.com/58030051/158103534-0eea7fe9-b3bc-485f-a752-8151b3982901.png)! 
 
[![Star History Chart](https://api.star-history.com/svg?repos=thomas-yanxin/Smart_container&type=Date)](https://star-history.com/#thomas-yanxin/Smart_container&Date)
## 😉Recent Update😉  
-  **Release training code**：Publish model training code and experimental results for adaptive tuning;
-  **Fixe Code Bug**： Restart the service after the database is updated in Pipeline deployment mode;
-  **Improve Accuracy Greatly**：The test accuracy of the self-collected data set is **98.442%**；
-  **Upgrade Document**：Provides detailed documentation of [PP-ShiTu model training and deployment](https://github.com/thomas-yanxin/Smart_container/blob/master/docs/readme_industry_v5.md)；
-  **Optimize Deployment Mode**：The predicted speed increase is **65**%, based on the overall CPU flow control at **0.9s**； 
-  **Upgrade Product Function**：Add inventory information management function, provide one-click data visualization analysis platform;

## 🧁Project Context🧁
<font size=3 >Currently in the process of actual operations of the retail industry, will produce a great human cost, such as guides, cleaning, settlement, and among them, especially need to spend a lot of labor cost and time cost in the identification of goods and settlement in the process of the price, and in the process, and so the customer need to wait in line.  As a result, the retail industry has high labor costs and low work efficiency. Secondly, it also reduces the shopping experience of customers.    
  
With the development of computer vision technology, as well as the unmanned and automated supermarket operation concept, the use of image recognition technology and target detection technology to achieve Automatic product identification and Automatic settlement demand, namely Automatic checkout system (ACO).  The automatic checkout system based on computer vision can effectively reduce the operating cost of retail industry, improve the checkout efficiency of customers, so as to further enhance the user experience and happiness in the process of shopping.    </font>

## Applicable Scene  

 >**ColugoMum——Intelligent Retail Rettlement Platform**Committed to provide **the largest offline retail experience store** with retail settlement solution based on vision.  

## Pain Problem

   1. **Settlement efficiency is highly required**：In the shopping scenario of large offline retail experience stores, if customers buy more goods, the traditional bar code settlement is adopted, which is inefficient and leads to poor shopping experience of customers；
   2. **Category update very fast**：For in the new retail industry, new products are almost always updated on an hourly basis. Every time new products are added, the model has to be retrained so hard that it is impossible for a single model to keep pace；
   3. **Different products are very similar**：Different flavors of the same drink, for example, are likely to have very similar packaging. And even for the same product, **there are often considerable differences in the product images obtained under different circumstances**;
   4. **Tens of thousands of commodity categories**：There is no way to put all categories into the training set beforehand。

## 🍑Realize function🍑
<font size=3 >**ColugoMumr**ealize automatic settlement of goods purchased by users in the retail process. We take advantage of the PaddleClas team's open source [PP-ShiTu](https://arxiv.org/pdf/2111.00775.pdf) technology, precise positioning of customers to buy goods, and intelligent, automatic price settlement. When customers place their chosen products in the designated area, **ColugoMum** can accurately locate and identify each product, and can return a complete shopping list and the actual total price of goods that customers should pay. When the system has a new product increase, the system only need to update the retrieval database, without retraining the model.   
    
This project is a lightweight general PP - ShiTu image recognition system provides the solid ground application cases, the new one of the retail industry and retail visual intelligent solution provides a very good basis and train of thought, especially for solving many categories, small sample, high similarity, and frequently updated the special image recognition scene pain difficulties provides reference of demonstrations, Greatly reduce the retail industry in the actual operation of the huge human cost, improve the retail industry unmanned, automation, intelligent level.  </font>  

<div align="center">
<img src="https://ai-studio-static-online.cdn.bcebos.com/56a6521f80754fcdb12ab433e35ce343b7a5e475b56446e8beb4d9c93213e7b3" width = "480" height = "320"/>
</div>


## 🍎Overall Architecture🍎
<div align="center">
<img src="https://ai-studio-static-online.cdn.bcebos.com/59f875129c854cdfb7cbb3435f5004c37ffed920756b41e5bde49e98c09cd0ab" width = "1080" height = "640"/>
</div>


## 🐻Technical Route🐻
<font size=3 >**ColugoMum** Based on PaddleClas as the main feature development suite, leveraging its open source [PP-ShiTu](https://arxiv.org/pdf/2111.00775.pdf) for core feature development. Through PaddleInference, it was deployed in Jetson Nano, and was packaged based on [QPT](https://github.com/QPT-Family/QPT) to develop an industrial-level intelligent retail settlement platform in line with actual application requirements.  </font>

### [PP-ShiTu](https://arxiv.org/pdf/2111.00775.pdf) Introduce

PP-ShiTu is a practical lightweight general image recognition system, which is mainly composed of three modules: subject detection, feature learning and vector retrieval.  The system from the selection and adjustment of backbone networks, the choice of loss function, vector data, transform strategy, choice of regularization parameter, use the training model and quantitative model cutting eight aspects, use a variety of strategies, optimize the model of the various modules, finally got on the CPU is only 0.2 s to complete 10 w + library image recognition system.  

<div align="center">
<img src="./image/structure.jpg" width = "1080" height = "540"/>
</div>

<font size=3 >The whole image recognition system is divided into three steps（[See PP-ShiTu training module for details](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.3/docs/zh_CN/models_training/recognition.md)）：  
(1) The candidate regions of image objects are detected by a target detection model;  
(2) Feature extraction for each candidate region;  
(3) Feature matching with images in the retrieval database, and extraction of recognition results.  

For the new unknown category, there is no need to retrain the model, but only need to add the image of the category in the retrieval database and rebuild the retrieval database to recognize the category.   </font>

### Introduction to Data Set
【The first one】:[Products-10K Large Scale Product Recognition Dataset](https://arxiv.org/abs/2006.12634)  

【The second one】:[RP2K: A Large-Scale Retail Product Dataset for Fine-Grained Image Classification](https://www.pinlandata.com/rp2k_dataset)  

**ColugoMum** based on the above two data sets and combined with the actual characteristics of the retail scene, adaptive processing is carried out.  

### List of Commodities 

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


The processed dataset is now open source in [AIStudio](https://aistudio.baidu.com/aistudio/datasetdetail/113685). </font>
## Ablation experiments ##
 |  model  | num epoch |  batch size/gpu cards |  learning rate  |  use cutout  |  use ssld  |  top1 recall  | config |
 | :----: | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
 | PP_LCNet_x2_5 | 400 | 256/4 | 0.01 | N | N | [98.189%](./exprements/log/98189.log) | [config](./exprements/PaddleClas/ppcls/configs/GeneralRecognition/GeneralRecognition_PPLCNet_x2_5_01.yaml) |
 | PP_LCNet_x2_5 | 400 | 256/4 | 0.01 | Y | N | [98.21%](./exprements/log/98216.log) | [config](./exprements/PaddleClas/ppcls/configs/GeneralRecognition/GeneralRecognition_PPLCNet_x2_5_01_cutout.yaml) |
 | PP_LCNet_x2_5 | 400 | 256/4| 0.005 | N | N | [98.201%](./exprements/log/98201.log) | [config](./exprements/PaddleClas/ppcls/configs/GeneralRecognition/GeneralRecognition_PPLCNet_x2_5_005.yaml) |
 | PP_LCNet_x2_5 | 400 | 256/4| 0.005 | Y | N | [98.29%](./exprements/log/98291.log) | [config](./exprements/PaddleClas/ppcls/configs/GeneralRecognition/GeneralRecognition_PPLCNet_x2_5_005_cutout.yaml) |
 | PP_LCNet_x2_5 | 400 | 256/4 | 0.001 | Y | N | 98.26% |config|
 | PP_LCNet_x2_5 | 400 | 64/4 | 0.005 | Y | Y | 98.30% | config|
 | PP_LCNet_x2_5 | 400 | 64/4 | 0.0025 | Y | Y | [98.37%](./exprements/log/98379.log) | config |
 | PP_LCNet_x2_5 | 400 | 64/4 | 0.002 | N | Y | [98.38%](./exprements/log/98383.log) | [config](./exprements/PaddleClas/ppcls/configs/GeneralRecognition/GeneralRecognition_PPLCNet_x2_5_dml_002.yaml) |
 | PP_LCNet_x2_5 | 400 | 64/4 | 0.002 | Y | Y | [98.39%]((./exprements/log/98395.log)) | [config](./exprements/PaddleClas/ppcls/configs/GeneralRecognition/GeneralRecognition_PPLCNet_x2_5_dml_002_cutout.yaml) |
<!--  | PP_LCNet_x2_5 | 400 | 128/4 | 0.004 | N | Y | [98.44%](./exprements/log/98442.log) | [config](./exprements/PaddleClas/ppcls/configs/GeneralRecognition/GeneralRecognition_PPLCNet_x2_5_004.yaml) |
 | PP_LCNet_x2_5 | 400 | 128/4 | 0.004 | Y | Y | [98.38%](./exprements/log/98376.log) | [config](./exprements/PaddleClas/ppcls/configs/GeneralRecognition/GeneralRecognition_PPLCNet_x2_5_004_cutout.yaml) | -->

**Attention**:
1. This experiment is based on GPU:Tesla V100* 4; CPU:Inter Xeon* 32; RAM:DDR4 128GB for training and testing;
2. The experiments are based on the above data set [Retail Product Characteristics Study Data Set](https://aistudio.baidu.com/aistudio/datasetdetail/113685) for training and testing;
3. Evaluation of RP2K and other large open source data sets of retail products will be carried out soon.
   

## 🌍Deployment Mode🌍
**ColugoMum** has been connected to**Jetson Nano, Windows, linux** system.  

<font size=3 >
  
  - Windows  
  [**ColugoMum** provides a relatively simple demo version]  
	
    We use [QPT](https://github.com/QPT-Family/QPT) for packaging.   
    Download the project code, enter the QPT_client folder, and Click the "启动程序.exe".
	  
  - Linux  
    Download the project code, enter the client folder, and run the following code to run it  ：
    ```
      python client.py
     ```
  
  - For details of the image recognition part deployment, you can see [PP-ShiTu Development](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.3/docs/zh_CN/inference_deployment/python_deploy.md#%E4%B8%BB%E4%BD%93%E6%A3%80%E6%B5%8B%E3%80%81%E7%89%B9%E5%BE%81%E6%8F%90%E5%8F%96%E5%92%8C%E5%90%91%E9%87%8F%E6%A3%80%E7%B4%A2%E4%B8%B2%E8%81%94)
  
  - Wechat applet 
   Open the wechat developer tool, import the AIContainer folder under the system folder and run it；

## 💃[bilibili](https://www.bilibili.com/video/BV19q4y1G7bx#reply5654379507) Results Demonstrate💃  
 
    
- Main Interface
  <div align="center"><img src="./image/all.jpg" width = "720" height = "540"/></div>
  
- Client Side Interface  
 
<div align="center"><img src="./image/pic_paddle.gif" width = "720" height = "540"/></div>


- Applets Interface
  <div align="center">
<img src="./image/wx_all.png" width = "840" height = "480"/>
</div>

- Big Data Visualization Analysis Interface
	<div align="center">
<img src="./image/datacenter.jpg" width = "840" height = "480"/>
</div>  


## ⛽️To Do List⛽️

  
 |  number  | complete degree |  priority |  category  |  Functional description  | 
 | :----: | :---- | :---- | :---- | :---- |
 |  1  |  completed  |★★★★★  | Applets | ~~Add inventory information display, add data analysis module~~|
 |  2  |  Doing  |★★★★★  | Applets |  Initial function online  |
 |  3  |  completed  |★★★★★  | Client Side |   ~~Jetson Nano Depth adaptation~~  |
 |  4  |  planning  |★★★★  | Applets |  Separation of functions for managers and customers  |
 |  5  |  completed  |★★★★  | web |   ~~ the establishment of web information management system~~  |
 |  6  |  planning  |★★★  | Applets |  Realize the automatic entry of commodity name  |
 |  7  |  planning  |★★  | APP | Enabling deployment on the IOS and Android  |


## 🚀 Development Team🚀

  |  Duty   |  Name  |
|  :----:  | :----:  |
| PM | [X. Yan](https://github.com/thomas-yanxin) |
| Algorithm | [X. Yan](https://github.com/thomas-yanxin) |
| Side of the front end | [X. Yan](https://github.com/thomas-yanxin) |
| Applets front end  | [C. Shen](https://github.com/Scxw010516) |
 | Back End  | [D. DU](https://github.com/DXD-agumo) |
  
## ☕Sponsor☕
A cup of coffee will refresh your mind, and product updates will be faster and better!  
<div><img width="490" alt="图片" src="./image/pay.png"></div>
	  
## 🌟Thanks🌟
  - [PaddleClas](https://github.com/PaddlePaddle/PaddleClas) provides the image recognition core function development;
  - [QPT](https://github.com/QPT-Family/QPT) provides Windows side package;
  - [jkfx](https://github.com/jkfx) fixed some bugs.

## ❤️Welcome to build together❤️
  We welcome you to contribute code or provide suggestions for "**ColugoMum**". Whether you have a bug, fix a bug, or add a new feature, feel free to submit Issue or Pull Requests.

##  <img src="https://user-images.githubusercontent.com/48054808/157835276-9aab9d1c-1c46-446b-bdd4-5ab75c5cfa48.png" width="20"/> Citation
```
@software{ColugoMum2021,
  author = {Xin Yan, Chen Shen and XuDong Du},
  title = {{ColugoMum: Intelligent Retail Settlement Platform}},
  howpublished = {\url{https://github.com/thomas-yanxin/Smart_container}},
  year = {2021}
}
```
