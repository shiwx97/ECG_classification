# ECG_classification 心电信号识别
This repository contains code for ECG-classificaton based on deep learning model.
心拍类型：N A V L R

# DataSet
Mit-bih-arrhythmia-database-1.0.0

# Software
Python
PyTorch
Scikit-learn
Tqdm
Wfdb
Pandas
Numpy
Scipy

# 文件说明
## read_data.py  从dataset中提取数据，生成数据集和标签(X_data,Y_data)
## split_data.py  拆分数据集，得到X_train,Y_train,X_label,Y_label  #根据R峰的位置，向前取99个点，向后取201个点，作为一组数据。以R峰的心拍类型确定此段数据的类型。
## preprocessing.py  对信号进行小波变换
## network.py  神经网络
## main.py  模型训练和预测

# Run 运行
## Preprocessing 数据预处理
$ python read_data.py
## Train 训练
$ python main.py --data-dir /ECG_data --use-gpu



