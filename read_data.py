import wfdb
import numpy as np
import pandas as pd
import os
import scipy
from scipy import signal
import pickle
from preprocessing import *

#使用MIT_BIH Arrhythmia Database数据集
#使用MLII导联的数据
path = "C:/Users/shiwx97/Desktop/香港科技大学学习文件/2023自学deep-learning/ECG_learn/ECG_data/mit-bih-arrhythmia-database-1.0.0" #you should change it according to your data_path
files = os.listdir(path)
names = []
names_list = []
final_names = []  #使用MLII导联方式的数据

for file in files:
    if file[0:3] in names:
        continue
    else:
        names.append(file[0:3])

for name in names:
    if name[0] not in ["1","2"]:
        continue
    else:
        names_list.append(name)

#判断是否使用MLII导联
for _ in names_list:
    file = wfdb.rdheader(path + "/" + _)
    #print(file.sig_name)
    if "MLII" in file.sig_name:
        final_names.append(_)

#对信号进行预处理
X_data = []
Y_data = []
ecgClassSet = ["N","A","V","L","R"]  #标签

for _ in final_names:
    file = wfdb.rdrecord(path+"/"+_,channel_names=["MLII"])
    data = file.p_signal.flatten()

    #小波变换去噪
    data = denoise(data)

    #读取.atr文件中的标签信息
    annotation = wfdb.rdann(path+"/"+_,"atr")
    Rlocation = annotation.sample #每个R峰的位置
    Rclass = annotation.symbol #每个心拍的标注

    #根据R峰的位置，向前取99个点，向后取201个点，作为一个数据
    #去除开头和结尾的R峰
    start = 10
    end = 5
    i = start #从第10个R峰开始
    j = len(annotation.symbol) - end  #一共可以使用的
    while i<j:
        try:
            label = ecgClassSet.index(Rclass[i])  #确定每个心拍的标签
            #通过R峰进行定位
            x_data = data[Rlocation[i]-99:Rlocation[i]+201]  #取R峰的前99个点，后201个点
            X_data.append(x_data)
            Y_data.append(label)
            i = i+1
        except ValueError:
            i = i+1


X_data = np.array(X_data)
Y_data = np.array(Y_data)

Y_analysis = pd.DataFrame(Y_data)  #查看一下Y分类的数据类型
print(Y_analysis.value_counts())
#可以看到0：74330，1：2529，2：7092，3：8035，4：7202


with open("ECG_X_data.pkl","wb") as file:
    pickle.dump(X_data,file)

with open("ECG_Y_data.pkl","wb") as file:
    pickle.dump(Y_data,file)

