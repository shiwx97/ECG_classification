import argparse
import os
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from tqdm import tqdm
from Split_data import split_data
from Network import Classification
import sys

def parse_args():
    parser = argparse.ArgumentParser() #创建一个解析对象
    parser.add_argument("--data-dir",type=str, default = "C:/Users/shiwx97/Desktop/香港科技大学学习文件/2023自学deep-learning/ECG_review/",help = "Directory for data dir")  #数据的存放地址
    parser.add_argument("--lr","--learning-rate", type=float, default=0.0001, help ="learning_rate" )
    parser.add_argument("--batch-size",type=int,default=32,help="batch_size")
    parser.add_argument("--epochs",type=int,default=10,help="Training epochs")
    parser.add_argument("--save-path",type=str,default="",help="Path to saved model")
    #action="store_true"就相当于把"use-gpu"设置成了一个开关，我们不需要给这个开关传递具体的值。当启动的时候，use-gpu就会变为true
    parser.add_argument("--use-gpu",default="False",action="store_true",help="use_GPU")
    parser.add_argument("--phase",type=str,default="train",help="Phase: train or test")
    return parser.parse_args()  #调用parse_args()方法进行解析，解析成功后即可使用


def train(train_loader,net,args,loss_function,epoch,scheduler_lr,optimizer,device,Len_Y_train):
    #train
    print("------第{}轮开始-------".format(epoch+1))
    net.train()
    running_loss = 0.0
    predict_true = 0.0
    train_bar = tqdm(train_loader,file=sys.stdout)
    for step,data in enumerate(train_bar):
        x,y = data
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()  #梯度置0
        logits = net(x)  #正向传播
        loss = loss_function(logits,y.long())  #计算损失函数,标签y是longtensor类型
        loss.backward()  #反向传播
        optimizer.step()  #梯度更新

        #计算训练集的准确率
        #torch.max(input,dim)返回按dim维度的最大值以及最大值的索引
        predict_y_this_step = torch.max(logits,dim=1)[1]  #这个step预测的标签
        predict_true += torch.eq(predict_y_this_step,y).sum().item()  #将predict和true label进行比较，并累计，才是一个epoch上所有的数据
        running_loss += loss.item()
        train_bar.desc = "train epoch [{}/{}] loss:{:.3f}".format(epoch+1,args.epochs,loss)
    accuracy_train = predict_true / Len_Y_train
    return accuracy_train, running_loss

def test(test_loader,net,args,epoch,device,Len_Y_test):
    best_acc = 0.0
    acc = 0.0
    with torch.no_grad():
        val_bar = tqdm(test_loader,file=sys.stdout)
        for val_data in val_bar:
            x_val,y_val = val_data
            x_val = x_val.to(device)
            y_val = y_val.to(device)
            output = net(x_val)

            precict_this_step = torch.max(output,dim=1)[1]
            acc += torch.eq(precict_this_step,y_val).sum().item()
            val_bar.desc = "test epoch [{}/{}]".format(epoch+1,args.epochs)

        val_accurate = acc/Len_Y_test

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(),args.save_path)
            return val_accurate

def confusion_matrix(preds,labels,conf_matrix):
        for p,t in zip(preds,labels):
            conf_matrix[p,t] +=1
        return conf_matrix


if __name__ == '__main__':
    args = parse_args()
    #best_acc = 0.0  #目前的准确率

    if not args.save_path:
        args.save_path = f"ecg_classification_model.pth"

    if args.use_gpu and torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = "cpu"

    data_pkl = os.path.join(args.data_dir,"ECG_X_data.pkl")
    label_pkl = os.path.join(args.data_dir,"ECG_Y_data.pkl")

    conf_matrix = torch.zeros(5,5)


    #对数据集进行划分，划分训练集和测试集
    X_train,Y_train,X_test,Y_test,Len_Y_train,Len_Y_test = split_data(data_pkl,label_pkl)
    #构建数据集
    trainset = TensorDataset(X_train,Y_train)
    testset = TensorDataset(X_test,Y_test)
    #装载进dataloader
    train_loader = DataLoader(dataset=trainset,batch_size=args.batch_size,drop_last=True)
    test_loader = DataLoader(dataset=testset,batch_size=args.batch_size,drop_last=True)
    train_step = len(train_loader)

    net = Classification()                     #网络

    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params,lr=args.lr)  #优化器
    scheduler_lr = lr_scheduler.StepLR(optimizer,step_size=5,gamma=0.1)  #learning_rate
    loss_function = nn.CrossEntropyLoss()
    loss_function = loss_function.to(device)    #损失函数

    if args.phase == "train":
        for epoch in range(args.epochs):
            accuracy_train,running_loss = train(train_loader,net,args,loss_function,epoch,scheduler_lr,optimizer,device,Len_Y_train)
            scheduler_lr.step() #学习率更新
            val_accurate = test(test_loader,net,args,epoch,device,Len_Y_test)
            print("[epoch %d] train_loss: %.3f, accuracy_train: %.3f, val_accuracy: %.3f" %(epoch+1,running_loss/train_step,accuracy_train,val_accurate))

        #查看最好模型的混淆矩阵
        net.load_state_dict(torch.load(args.save_path))
        for data in test_loader:
            x,y = data
            outputs = net(x.to(device))
            predict = torch.max(outputs,dim=1)[1]
            conf_matrix = confusion_matrix(predict,y.long(),conf_matrix)

        print("fuison_matrix",conf_matrix)



    else:
        net.load_state_dict(torch.load(args.save_path))
        val_accurate = test(test_loader,net,args,args.epochs,device,Len_Y_test)
        for data in test_loader:
            x,y = data
            outputs = net(x.to(device))
            predict = torch.max(outputs,dim=1)[1]
            conf_matrix = confusion_matrix(predict,y.long(),conf_matrix)

        print("fuison_matrix",conf_matrix)








