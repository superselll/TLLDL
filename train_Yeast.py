# library
# standard library
import os, sys
import scipy.io as scio
# third-party library
import numpy as np
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from dataset.dataset_Yeast import Dataset_Yeast,Generate_distribution,distribution_to_class
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

from loss_function import eval,save


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

    loss_func = nn.CrossEntropyLoss().to(device)
    kl_loss_1 = nn.KLDivLoss().to(device)
    kl_loss_2 = nn.KLDivLoss().to(device)
    kl_loss_3 = nn.KLDivLoss().to(device)

    total_loss_matrix = []
    # training and testing
    start = timer()

    O = np.zeros((num_label, num_classes))
    start_end_array = np.linspace(start=0, stop=num_label, num=num_classes + 1, dtype=int)
    for i in range(num_classes):
        O[start_end_array[i]:start_end_array[i + 1], i] = 1
    O = torch.from_numpy(O).to(device).float()  # 0-1矩阵
    O.requires_grad = False
    start_end_array = torch.from_numpy(start_end_array).to(device).float()

    for epoch in range(epochs):  # (EPOCH):#
        # scheduler.step(epoch)
        print(f"------------epoch{epoch} start------------------")
        losses_cls = AverageMeter()
        losses_cou = AverageMeter()
        losses_cou2cls = AverageMeter()
        losses = AverageMeter()
        # '''test
        if (epoch) % 10 == 0:
            model.eval()
            with (torch.no_grad()):
                test_loss = 0
                test_corrects_add = 0
                test_corrects1 = 0
                test_corrects2 = 0

                loss_matrix = torch.zeros(12)

                for feature, classes in test_loader:  # x是图像，y是等级，l是个数
                    feature = feature.to(device).float()
                    classes = classes.to(device).float()
                    # baseDCNN_feature = torch.tensor(baseDCNN_feature).to(device).float()
                    distribution = generator.genld(classes)  # distribution是分布的曲线
                    # distribution = torch.tensor(distribution).to(device).float()
                    label, cls = model(feature)
                    label = label + 1e-5
                    loss1 = kl_loss_1(label.log(), distribution) * num_label
                    loss2 = loss_func(cls, classes) * num_classes  # 直接预测的误差
                    predict_ldl = torch.matmul(label, O)
                    # classes为gt的类别置信度

                    loss_matrix += eval(predict_ldl, cls, classes)

                    loss_count_cou = kl_loss_3((torch.log(predict_ldl)), classes) * num_classes
                    test_loss += (loss1 + loss_count_cou) * lam + loss2 * (1.0 - lam)


                test_loss = test_loss.float() / len(test_loader)

                startend_message = start_end_array.cpu().numpy().tolist()
                mess = ''
                for i in range(len(startend_message)):
                    mess += str(startend_message[i]) + ", "
                message = '%s %6.1f | %0.3f | %s| \n' % ( \
                    "test ", epoch,
                    test_loss.data,
                    mess,)
                log.write(message)

                total_loss_matrix.append(loss_matrix/len(test_loader))
        # '''train
        model.train()
        for feature, classes in train_loader:  # features是图像，classess是不同类别的置信度大小
            feature = feature.to(device).float()
            classes = classes.to(device).float()
            distribution = generator.genld(classes)
            # distribution = torch.tensor(distribution).to(device).float()
            label, cls = model(feature)
            label = label + 1e-5
            loss1 = kl_loss_1(label.log(), distribution) * num_label
            loss2 = kl_loss_2(cls.log(), classes) * num_classes  # 直接预测的误差
            if epoch < int(epochs/10):
                with torch.no_grad():
                    O, start_end_array = markov(label, O, classes, start_end_array, kl_loss_3)
                loss_count_cou = kl_loss_3((torch.log(torch.matmul(label, O))),
                                           classes) * num_classes  # 模型预测的个数加权与真实类别标签的差异

                loss = (loss1+loss_count_cou) * lam + (loss2) * (1.0 - lam)
                optimizer.zero_grad()  # clear gradients for this training step
                loss.backward()  # backpropagation, compute gradients
                optimizer.step()  # apply gradients

                losses_cls.update(loss2.item(), feature.size(0))  #直接预测的误差
                losses_cou.update(loss1.item(), feature.size(0))  #分布之间的误差

                losses_cou2cls.update(loss_count_cou.item(),feature.size(0))  #求和之后的误差
                losses.update(loss.item(), feature.size(0))
            else:
                loss = (loss1) * lam + (loss2) * (1.0 - lam)
                optimizer.zero_grad()  # clear gradients for this training step
                loss.backward()  # backpropagation, compute gradients
                optimizer.step()  # apply gradients
                losses_cls.update(loss2.item(), feature.size(0))  # 直接预测的误差
                losses_cou.update(loss1.item(), feature.size(0))  # 分布之间的误差
                #losses_cou2cls.update(0, feature.size(0))
                losses.update(loss.item(), feature.size(0))

        message = '%s %6.0f | %0.3f | %0.3f | %0.3f | %0.3f | %s\n' % ( \
            "train", epoch,
            losses_cls.avg,  # 直接预测的误差
            losses_cou.avg,  # 分布之间的误差
            losses_cou2cls.avg,  # 分布加权预测的误差
            losses.avg,  # 总损失
            time_to_str((timer() - start), 'min'))
        log.write(message)






    number = O.numpy()
    np.savetxt(os.path.join(result_dir, 'O.txt'), number, fmt='%d', delimiter=',')
    save(os.path.join(result_dir, 'loss.xls'), np.array(total_loss_matrix).astype(np.float_))




def main():
    # Hyper Parameters
    batch_size = 8
    epochs = 100
    lr = 0.0001  # learning rate
    rate = 10
    weight_method = 1
    #names = ['RAMFL_baseDCNN', 'Yeast_alpha', 'Yeast_cdc', 'SJAFFE', 'SBU_3DFE', 'Natural_Scene', 'fbp5500', 'Movie']
    names = ['Yeast_alpha']
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cur_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    #names = ['Movie','Natural_Scene','SBU_3DFE','SJAFFE','Twitter_ldl']#'fbp5500','Flicker_ldl','Human_Gene',
    for name in names:
        dataset_name = name
        result_dir = os.path.join('results', dataset_name, cur_time)
        #result_dir = os.path.join('results/RAFML', cur_time)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)


        feature_file = os.path.join('dataset', dataset_name,'feature.npy')
        label_file = os.path.join('dataset', dataset_name, 'label.npy')
        train_dataset = Dataset_Yeast(feature_file, label_file,'',istrain=True)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_dataset = Dataset_Yeast(feature_file, label_file,'', istrain=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        '''
        trainDistribution = train_dataset.label
        trainFeature = train_dataset.feature
        testDistribution = test_dataset.label
        testFeature = test_dataset.feature
        trainNum = trainDistribution.shape[0]
        testNum = testDistribution.shape[0]
                if not os.path.exists(name+'.mat'):
            scio.savemat(name+'.mat', {'trainDistribution': trainDistribution,'trainFeature':trainFeature,'testDistribution':testDistribution,'testFeature':testFeature,'trainNum':trainNum,'testNum':testNum})
        '''
        feature = np.load(feature_file)
        label = np.load(label_file)
        if not os.path.exists(name+'.mat'):
            scio.savemat(name+'.mat', {'Distribution': label,'Feature': feature})

        num_classes = train_dataset.outfeature

        num_labels = rate*num_classes
        if weight_method:
            s = train_dataset.weight
        else:
            s = [1] * num_classes
        distribution_generator = Generate_distribution(s,rate)

        model = Mlp(input_size=train_dataset.infeature,hidden_size=int((train_dataset.infeature+num_labels)/2),num_label=num_labels,num_classes=num_classes).to(device)


        log = Logger()
        log_file_name = os.path.join(result_dir, 'loss.log')
        log.open(log_file_name, mode="w")
        log.write('\n\nbegin_train: \n\n')
        log.write(f'input_feature_size:{train_dataset.infeature}\n')
        log.write(f'predict_label_size:{num_classes}\n')
        log.write(f'O_startend:{distribution_generator.startend}\n')
        trainval_test(model, train_loader,test_loader,generator=distribution_generator, log=log, epochs=epochs, learning_rate=lr,sigma=3, lam=0.8, result_dir=result_dir,num_label=num_labels, num_classes=num_classes,rate=10)



if __name__ == '__main__':
    main()

