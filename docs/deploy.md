# 项目部署流程

## 服务端部署
1. 获取项目代码并安装依赖包
git clone https://git.openi.org.cn/ColugoMum/Smart_container.git  # clone
cd Smart_container
pip install -r requirements.txt  # install

2. 导入数据库并修改数据库信息
container.sql

修改数据库信息

Smart_container/src/branch/master/Smart_container/djangoProject/settings.py

```shell
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.mysql',	# 要连接的 数据库类型
        'HOST': 'localhost',	# 要连接的远程数据库的 ip地址
        'PORT': '3306',	# 数据库连接端口，mysql默认3306
        'USER': 'SM_C',		# 数据库已有用户名`，需要更改
        'PASSWORD': 'XXXXX',	# 数据库已有用户密码，需要更改
        'NAME': 'container',	# 要连接的 数据库名
    }
}
```
3. 启动Django框架

在终端执行此命令:
Smart_container/Smart_container
python manage.py runserver 0.0.0.0:8001

## 模型服务化部署

1.进入到工作目录
PaddleClas/deploy/paddleserving/recognition

2.修改配置文件config.yml路径
```
op:
    rec:
            ···
            #uci模型路径
            model_config: /你的项目路径/PaddleClas/deploy/inference_PPLCNet_serving
            
            #计算硬件类型: 空缺时由devices决定(CPU/GPU)，0=cpu, 1=gpu, 2=tensorRT, 3=arm cpu, 4=kunlun xpu
            device_type: 1

            #计算硬件ID，当devices为""或不写时为CPU预测；当devices为"0", "0,1,2"时为GPU预测，表示使用的GPU卡
            devices: "0" # "0,1"

            #client类型，包括brpc, grpc和local_predictor.local_predictor不启动Serving服务，进程内预测
            client_type: local_predictor

            #Fetch结果列表，以client_config中fetch_var的alias_name为准
            fetch_list: ["features"]
     det:
        concurrency: 1
        local_service_conf:
            client_type: local_predictor
            device_type: 1
            devices: '0'
            fetch_list:
            - save_infer_model/scale_0.tmp_1
            model_config: /你的项目路径/PaddleClas/deploy/picodet_PPLCNet_x2_5_mainbody_lite_v1.0_serving
```

3.修改启动服务脚本recognition_web_service.py 这里主要修改服务所需的检索库地址
```
        ···
        index_dir = "/你的项目路径/dataset/index_update"
        
        assert os.path.exists(os.path.join(
            index_dir, "vector.index")), "vector.index not found ..."
        assert os.path.exists(os.path.join(
            index_dir, "id_map.pkl")), "id_map.pkl not found ... "
        ···
 ```
 
4.启动/停止服务

启动服务---
在终端执行此命令:
```python
python recognition_web_service.py &>log.txt &
```
启动服务后，运行日志保存在 log.txt可查看是否正常运行。

停止服务---
在终端执行此命令:
```python
python -m paddle_serving_server.serve stop
```

## 客户端运行

进入client文件夹内，执行以下代码即可运行：
```shell
python client.py
```

注：Linux下运行需要修改client.py文件,注释掉win32相关代码

修改self.rate = (w,h)
```
class Ui_MainWindow(QtWidgets.QWidget):

    def __init__(self):
        # hDC = win32gui.GetDC(0)
        # # 横向分辨率
        # w = win32print.GetDeviceCaps(hDC, win32con.DESKTOPHORZRES)
        # # 纵向分辨率
        # h = win32print.GetDeviceCaps(hDC, win32con.DESKTOPVERTRES)
        
        self.rate = (800,500)
        if self.rate != 1 and self.rate != 1.25 and self.rate != 1.5 and self.rate != 1.75 :
            self.rate = 1
        super(Ui_MainWindow, self).__init__()

```
修改请求地址
res=requests.post('http://127.0.0.1:8001/reference_client/', data=req)


## 小程序端运行

打开微信开发者工具，导入系统文件夹下AIContainer文件夹并运行，即可运行小程序端。

需要修改Smart_container/AIContainer/miniprogram/app.js 
改为自己IP地址方可正常请求后台。
========================
小程序商品图片显示问题修改
========================
pages/main/revisepage/revise.js
pages/main/main/revise.wxml

注:上传商品前数据表t_container只留一条数据，如果全部清空数据上传商品时会报错。
