# library
# standard library
import os, sys

# third-party library
import numpy as np
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from dataset.dataset_RAFML import Dataset_RAFML,Generate_distribution,distribution_to_class
from timeit import default_timer as timer
from utils.report import report_precision_se_sp_yi, report_mae_mse
from utils.utils import Logger, AverageMeter, time_to_str, weights_init
from utils.genLD import genLD
from model.resnet50 import resnet50
from model.mlp import Mlp
import torch.backends.cudnn as cudnn
from transforms.affine_transforms import *
import time
import logging
from pathlib import Path
import warnings
import scipy.io as scio

warnings.filterwarnings("ignore")
import os
from markov_method import markov

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def criterion(lesions_num):
    if lesions_num <= 5:
        return 0
    elif lesions_num <= 20:
        return 1
    elif lesions_num <= 50:
        return 2
    else:
        return 3


def trainval_test(model,train_loader,test_loader, generator, log, epochs, learning_rate, sigma, lam, result_dir, num_label,num_classes, rate):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    loss_func = nn.L1Loss()#nn.CrossEntropyLoss().to(device)
    kl_loss_1 = nn.KLDivLoss().to(device)
    kl_loss_2 = nn.KLDivLoss().to(device)
    kl_loss_3 = nn.KLDivLoss().to(device)

    # training and testing
    start = timer()
    test_acc_his = 0.7
    test_mae_his = 8
    test_mse_his = 18

    O = np.zeros((num_label, num_classes))
    start_end_array = np.linspace(start=0, stop=num_label, num=num_classes + 1, dtype=int)
    for i in range(num_classes):
        O[start_end_array[i]:start_end_array[i + 1], i] = 1
    O = torch.from_numpy(O).to(device).float()  # 0-1矩阵
    O.requires_grad = False
    start_end_array = torch.from_numpy(start_end_array).to(device).float()
    O_total = []
    start_end_array_total = []
    for epoch in range(epochs):  # (EPOCH):#
        # scheduler.step(epoch)
        print(f"------------epoch{epoch} start------------------")
        losses_cls = AverageMeter()
        losses_cou = AverageMeter()
        losses_cou2cls = AverageMeter()
        losses = AverageMeter()
        if (epoch) % 10 == 0:
            model.eval()
            with torch.no_grad():
                test_loss = 0
                test_corrects_add = 0
                test_corrects1 = 0
                test_corrects2 = 0
                y_true = np.array([])
                y_pred = np.array([])
                y_pred_m = np.array([])
                l_true = np.array([])
                l_pred = np.array([])
                for name, baseDCNN_feature, DBM_CNN_feature, LBP_feature, classes, classes01 in test_loader:  # x是图像，y是等级，l是个数

                    baseDCNN_feature = torch.tensor(baseDCNN_feature).to(device).float()
                    distribution = generator.genld(classes)  # distribution是分布的曲线
                    distribution = torch.tensor(distribution).to(device).float()
                    label, cls = model(baseDCNN_feature)
                    label = label + 1e-5
                    loss1 = kl_loss_1(label.log(), distribution) * num_label
                    loss2 = loss_func(cls, classes) * num_classes  # 直接预测的误差
                    predict_ldl = torch.matmul(label, O)
                    loss_count_cou = kl_loss_3((torch.log(predict_ldl)), classes) * num_classes
                    test_loss += (loss1+loss_count_cou) * lam + loss2 * (1.0 - lam)
                    preds_add = distribution_to_class((cls+predict_ldl)/2)
                    preds_ldl = distribution_to_class(predict_ldl)
                    preds_cls = distribution_to_class(cls)

                    batch_corrects_add = torch.sum((preds_add == classes01)).data.cpu().numpy()
                    test_corrects_add += batch_corrects_add
                    batch_corrects1 = torch.sum((preds_ldl == classes01)).data.cpu().numpy()
                    test_corrects1 += batch_corrects1
                    batch_corrects2 = torch.sum((preds_cls == classes01)).data.cpu().numpy()
                    test_corrects2 += batch_corrects2

                test_loss = test_loss.float() / len(test_loader)
                test_acc1 = test_corrects_add / len(test_loader.dataset)/num_classes
                test_acc2 = test_corrects1 / len(test_loader.dataset)/num_classes
                test_acc_add = test_corrects2 / len(test_loader.dataset)/num_classes
                startend_message = start_end_array.cpu().numpy().tolist()
                mess = ''
                for i in range(len(startend_message)):
                    mess += str(startend_message[i]) + ", "
                message = '%s %6.1f | %0.3f | %0.3f | %0.3f | %0.3f | %s| \n' % ( \
                    "test ", epoch,
                    test_loss.data,
                    test_acc1,
                    test_acc2,
                    test_acc_add,
                    mess,)
                log.write(message)
                # _, _, pre_se_sp_yi_report = report_precision_se_sp_yi(y_pred, y_true)
                # _, _, pre_se_sp_yi_report_m = report_precision_se_sp_yi(y_pred_m, y_true)
                #_, MAE, MSE, mae_mse_report = report_mae_mse(l_true, l_pred, y_true)

                # if True:
                #    log.write(str(pre_se_sp_yi_report) + '\n')
                #    log.write(str(pre_se_sp_yi_report_m) + '\n')
                #log.write(str(mae_mse_report) + '\n')
        # '''train
        model.train()
        for name, baseDCNN_feature, DBM_CNN_feature, LBP_feature, classes, _ in train_loader:  # features是图像，classess是不同类别的置信度大小

            baseDCNN_feature = baseDCNN_feature.to(device).float()
            classes = classes.to(device).float()
            distribution = generator.genld(classes)  #distribution是分布的曲线
            distribution = torch.tensor(distribution).to(device).float()
            label, cls = model(baseDCNN_feature)
            label = label + 1e-5
            loss1 = kl_loss_1(label.log(),distribution)*num_label
            loss2 = loss_func(cls,classes)*num_classes  #直接预测的误差

            with torch.no_grad():
                O, start_end_array = markov(label, O, classes, start_end_array, kl_loss_3)
            loss_count_cou = kl_loss_3((torch.log(torch.matmul(label, O))), classes) * num_classes #模型预测的个数加权与真实类别标签的差异

            loss = (loss1+loss_count_cou) * lam + loss2 * (1.0 - lam)
            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients

            losses_cls.update(loss2.item(), baseDCNN_feature.size(0))
            losses_cou.update(loss1.item(), baseDCNN_feature.size(0))
            losses_cou2cls.update(loss_count_cou.item(),baseDCNN_feature.size(0))
            losses.update(loss.item(), baseDCNN_feature.size(0))

        message = '%s %6.0f | %0.3f | %0.3f | %0.3f | %0.3f | %s\n' % ( \
            "train", epoch,
            losses_cls.avg,  #直接预测的误差
            losses_cou.avg,  #分布之间的误差
            losses_cou2cls.avg,  #分布加权预测的误差
            losses.avg,   #总损失
            time_to_str((timer() - start), 'min'))
        log.write(message)

        # '''test


    number = O.numpy()
    np.savetxt(os.path.join(result_dir, 'O.txt'), number, fmt='%d', delimiter=',')


def main():
    # Hyper Parameters
    batch_size = 8
    epochs = 100
    lr = 0.001  # learning rate
    num_workers = 1  # 12
    num_classes = 6
    num_labels = 60
    rate = 10
    cur_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    result_dir = os.path.join('results/RAFML', cur_time)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    distribution_file = 'dataset/RAF-ML/distribution.txt'
    partition_label_file = 'dataset/RAF-ML/partition_label.txt'
    baseDCNN_mat = 'dataset/RAF-ML/baseDCNN.mat'
    DBM_CNN_mat = 'dataset/RAF-ML/DBM_CNN.mat'
    LBP_mat = 'dataset/RAF-ML/LBP.mat'

    model = Mlp(input_size=2000,hidden_size=2000,num_label=num_labels,num_classes=num_classes)
    s = [1003.711652, 511.380149, 1057.974047, 677.058356, 783.200313, 874.675486]
    distribution_generator = Generate_distribution(s,rate)

    train_dataset = Dataset_RAFML(distribution_file, partition_label_file, baseDCNN_mat, DBM_CNN_mat, LBP_mat,istrain=True)
    train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers)
    test_dataset = Dataset_RAFML(distribution_file, partition_label_file, baseDCNN_mat, DBM_CNN_mat, LBP_mat,istrain=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    names = ['RAMFL_baseDCNN','RAMFL_DBM_CNN','RAFML_LBP']

    trainDistribution = train_dataset.distribution
    trainFeature = train_dataset.baseDCNN_feature
    testDistribution = test_dataset.distribution
    testFeature = test_dataset.baseDCNN_feature
    trainNum = trainDistribution.shape[0]
    testNum = testDistribution.shape[0]
    name = names[0]
    if not os.path.exists(name + '.mat'):
        scio.savemat(name + '.mat', {'trainDistribution': trainDistribution, 'trainFeature': trainFeature,
                                     'testDistribution': testDistribution, 'testFeature': testFeature,
                                     'trainNum': trainNum, 'testNum': testNum})
    dir = os.path.join('dataset',name)
    if not os.path.exists(dir):
        os.makedirs(dir)
    feature = np.concatenate((trainFeature,testFeature))
    label = np.concatenate((trainDistribution,testDistribution))
    train_label = np.arange(trainNum)
    test_label = np.arange(testNum)+trainNum
    np.save(os.path.join(dir,'feature'), feature)
    np.save(os.path.join(dir, 'label'), label)
    np.save(os.path.join(dir, 'train_label'), train_label)
    np.save(os.path.join(dir, 'test_label'), test_label)


    trainDistribution = train_dataset.distribution
    trainFeature = train_dataset.DBM_CNN_feature
    testDistribution = test_dataset.distribution
    testFeature = test_dataset.DBM_CNN_feature
    trainNum = trainDistribution.shape[0]
    testNum = testDistribution.shape[0]
    name = names[1]
    if not os.path.exists(name + '.mat'):
        scio.savemat(name + '.mat', {'trainDistribution': trainDistribution, 'trainFeature': trainFeature,
                                     'testDistribution': testDistribution, 'testFeature': testFeature,
                                     'trainNum': trainNum, 'testNum': testNum})
    dir = os.path.join('dataset', name)
    if not os.path.exists(dir):
        os.makedirs(dir)
    feature = np.concatenate((trainFeature, testFeature))
    label = np.concatenate((trainDistribution, testDistribution))
    train_label = np.arange(trainNum)
    test_label = np.arange(testNum) + trainNum
    np.save(os.path.join(dir, 'feature'), feature)
    np.save(os.path.join(dir, 'label'), label)
    np.save(os.path.join(dir, 'train_label'), train_label)
    np.save(os.path.join(dir, 'test_label'), test_label)

    trainDistribution = train_dataset.distribution
    trainFeature = train_dataset.LBP_feature
    testDistribution = test_dataset.distribution
    testFeature = test_dataset.LBP_feature
    trainNum = trainDistribution.shape[0]
    testNum = testDistribution.shape[0]
    name = names[2]
    if not os.path.exists(name + '.mat'):
        scio.savemat(name + '.mat', {'trainDistribution': trainDistribution, 'trainFeature': trainFeature,
                                     'testDistribution': testDistribution, 'testFeature': testFeature,
                                     'trainNum': trainNum, 'testNum': testNum})
    dir = os.path.join('dataset', name)
    if not os.path.exists(dir):
        os.makedirs(dir)
    feature = np.concatenate((trainFeature, testFeature))
    label = np.concatenate((trainDistribution, testDistribution))
    train_label = np.arange(trainNum)
    test_label = np.arange(testNum) + trainNum
    np.save(os.path.join(dir, 'feature'), feature)
    np.save(os.path.join(dir, 'label'), label)
    np.save(os.path.join(dir, 'train_label'), train_label)
    np.save(os.path.join(dir, 'test_label'), test_label)

    log = Logger()
    log_file_name = os.path.join(result_dir, 'loss.log')
    log.open(log_file_name, mode="w")
    log.write('\n\nbegin_train: \n\n')
    trainval_test(model, train_loader,test_loader,generator=distribution_generator, log=log, epochs=epochs, learning_rate=lr,sigma=3, lam=0.5, result_dir=result_dir,num_label=num_labels, num_classes=num_classes,rate=10)



if __name__ == '__main__':
    main()

