# face_mark
教程地址：https://blog.csdn.net/lwpyh/article/details/87902245 

需要：

python3.5+, keras2.2.4, face_recognition1.0.0, glob2 0.6 

face_mark 文件夹下是后台颜值检测算法的整套代码。  

SCUT-FBP5500_v2: 华南理工大学男女颜值检测数据集[Github](https://github.com/HCIILAB/SCUT-FBP5500-Database-Release)

prepare.py: 数据预处理代码 

train.py: 训练神经网络代码

predict.py: 测试代码

keras-flask-deploy-webapp 文件夹下是具体如何实现前端网页 demo 的代码

app.py: 实现交互的核心部分

注意将生成的 h5 文件放到 model 文件夹下
![Image text](https://github.com/KyrinCode/face_mark/raw/master/test.PNG)

