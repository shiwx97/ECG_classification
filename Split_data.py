import sklearn
from sklearn.model_selection import train_test_split
import pickle
import os
import torch

def split_data(data_path,label_path): #对数据集进行分层采样
    with open(data_path,"rb") as file:
        X = pickle.load(file)
    with open(label_path,"rb") as file:
        Y = pickle.load(file)
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,stratify=Y,shuffle=True)  #按Y分层采样
    #将数据转换为Tensor 类型
    X_train = torch.Tensor(X_train)
    X_train = X_train.unsqueeze(-1)  #batch,time_stp,dim

    X_test = torch.Tensor(X_test)
    X_test = X_test.unsqueeze(-1)

    Len_Y_train = len(Y_train)
    Y_train = torch.Tensor(Y_train)
    Len_Y_test = len(Y_test)
    Y_test = torch.Tensor(Y_test)

    return X_train,Y_train,X_test,Y_test,Len_Y_train,Len_Y_test




