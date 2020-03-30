# face_mark_v2

通过 face_recognition + Keras + Flask + Ngrok 部署 AI Beauty 颜值检测网站

## Requirements

python 3.5+

Keras 2.3.1

face_recognition 1.3.0

glob2 0.7

Flask 1.1.1

## Explanation

face_mark 文件夹下是后台颜值检测算法的整套代码。  

SCUT-FBP5500_v2: 华南理工大学男女颜值检测数据集 [Github](https://github.com/HCIILAB/SCUT-FBP5500-Database-Release)

prepare.py: 数据预处理代码 

train.py: 训练神经网络代码

predict.py: 测试代码

keras-flask-deploy-webapp 文件夹下是具体如何实现前端网页 demo 的代码

app.py: 实现交互的核心部分

需要将生成的 h5 文件放到 model 文件夹下

## Demo

![Image text](https://github.com/KyrinCode/face_mark/raw/master/test.PNG)

## 参考致谢

[CSDN 基于神经网络搭建自己的颜值检测网页](https://blog.csdn.net/lwpyh/article/details/87902245) 

[Github SCUT-FBP5500-Database-Release](https://github.com/HCIILAB/SCUT-FBP5500-Database-Release) 

[Github keras-flask-deploy-webapp](https://github.com/mtobeiyf/keras-flask-deploy-webapp)

[使用ngrok让你的本地Flask程序外网可访问](http://greyli.com/use-ngrok-to-expose-your-local-application/)