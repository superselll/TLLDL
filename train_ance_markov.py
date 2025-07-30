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
from dataset import dataset_processing
from timeit import default_timer as timer
from utils.report import report_precision_se_sp_yi, report_mae_mse
from utils.utils import Logger, AverageMeter, time_to_str, weights_init
from utils.genLD import genLD
from model.resnet50 import resnet50
import torch.backends.cudnn as cudnn
from transforms.affine_transforms import *
import time
import logging
from pathlib import Path
from loss_function import eval
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


def trainval_test(cross_val_index, sigma, lam, log, result_dir, batch_size, lr, num_labels, num_clases, rate):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = batch_size
    BATCH_SIZE_TEST = 20
    LR = lr
    NUM_WORKERS = 1  # 12
    NUM_LABELS = num_labels
    NUM_CLASSES = num_clases
    lr_steps = [25, 50, 75, 1]
    rate = rate

    np.random.seed(42)

    DATA_PATH = 'dataset/acne4/Classification/JPEGImages'

    TRAIN_FILE = 'dataset/acne4/VOCdevkit2007/VOC2007/ImageSets/Main/NNEW_trainval_' + cross_val_index + '.txt'
    TEST_FILE = 'dataset/acne4/VOCdevkit2007/VOC2007/ImageSets/Main/NNEW_test_' + cross_val_index + '.txt'

    normalize = transforms.Normalize(mean=[0.45815152, 0.361242, 0.29348266],
                                     std=[0.2814769, 0.226306, 0.20132513])

    dset_train = dataset_processing.DatasetProcessing(
        DATA_PATH, TRAIN_FILE, transform=transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            RandomRotate(rotation_range=20),
            normalize,
        ]))

    dset_test = dataset_processing.DatasetProcessing(
        DATA_PATH, TEST_FILE, transform=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize,
        ]))

    train_loader = DataLoader(dset_train,
                              batch_size=BATCH_SIZE,
                              shuffle=True,
                              num_workers=NUM_WORKERS,
                              pin_memory=True)

    test_loader = DataLoader(dset_test,
                             batch_size=BATCH_SIZE,
                             shuffle=False,
                             num_workers=NUM_WORKERS,
                             pin_memory=True)

    cnn = resnet50()  # .cuda()
    cudnn.benchmark = True

    params = []
    new_param_names = ['fc', 'counting']
    for key, value in dict(cnn.named_parameters()).items():
        if value.requires_grad:
            if any(i in key for i in new_param_names):
                params += [{'params': [value], 'lr': LR * 1.0, 'weight_decay': 5e-4}]
            else:
                params += [{'params': [value], 'lr': LR * 1.0, 'weight_decay': 5e-4}]

    optimizer = torch.optim.SGD(params, momentum=0.9)  #

    loss_func = nn.CrossEntropyLoss().to(device)
    kl_loss_1 = nn.KLDivLoss().to(device)
    kl_loss_2 = nn.KLDivLoss().to(device)
    kl_loss_3 = nn.KLDivLoss().to(device)

    def adjust_learning_rate_new(optimizer, decay=0.5):
        """Sets the learning rate to the initial LR decayed by 0.5 every 20 epochs"""
        for param_group in optimizer.param_groups:
            param_group['lr'] = decay * param_group['lr']

    # training and testing
    start = timer()
    test_acc_his = 0.7
    test_mae_his = 8
    test_mse_his = 18

    O = np.zeros((NUM_LABELS, NUM_CLASSES))
    start_end_array = np.linspace(start=0, stop=NUM_LABELS, num=NUM_CLASSES + 1, dtype=int)
    for i in range(NUM_CLASSES):
        O[start_end_array[i]:start_end_array[i + 1], i] = 1
    O = torch.from_numpy(O).to(device).float()  # 0-1矩阵
    O.requires_grad = False
    start_end_array = torch.from_numpy(start_end_array).to(device).float()
    O_total = []
    start_end_array_total = []
    for epoch in range(lr_steps[-1]):  # (EPOCH):#
        if epoch in lr_steps:
            adjust_learning_rate_new(optimizer, 0.5)
        # scheduler.step(epoch)
        print(f"------------epoch{epoch} start------------------")
        losses_cls = AverageMeter()
        losses_cou = AverageMeter()
        losses_cou2cls = AverageMeter()
        losses = AverageMeter()
        # '''test

        if (epoch) % 10 == 0:
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
                cnn.eval()
                for step, (test_x, test_y, test_l, name) in enumerate(test_loader):  # x是图像，y是等级，l是个数

                    test_x = test_x.to(device)
                    test_y = test_y.to(device)
                    L = np.zeros((test_x.shape[0], NUM_CLASSES))
                    L[[x for x in range(test_x.shape[0])], test_y] = 1
                    L = torch.from_numpy(L).to(device).float()

                    y_true = np.hstack((y_true, test_y.data.cpu().numpy()))
                    l_true = np.hstack((l_true, test_l.data.cpu().numpy()))

                    cnn.eval()

                    cls, cou = cnn(test_x, None)
                    predict = torch.matmul(cou, O)
                    ma = eval(predict, cls, L)
                    loss = loss_func(predict, test_y.long())  # 预测等级的误差
                    test_loss += loss.data

                    _, preds_add = torch.max(cls + predict, 1)
                    _, preds1 = torch.max(cls, 1)
                    _, preds2 = torch.max(predict, 1)

                    _, preds_l = torch.max(cou, 1)
                    preds_l = (preds_l + 1).data.cpu().numpy()

                    # y_pred_m = np.hstack((y_pred_m, preds_add.data.cpu().numpy()))
                    # y_pred = np.hstack((y_pred, preds.data.cpu().numpy()))
                    l_pred = np.hstack((l_pred, preds_l))

                    batch_corrects_add = torch.sum((preds_add == test_y)).data.cpu().numpy()
                    test_corrects_add += batch_corrects_add
                    batch_corrects1 = torch.sum((preds1 == test_y)).data.cpu().numpy()
                    test_corrects1 += batch_corrects1
                    batch_corrects2 = torch.sum((preds2 == test_y)).data.cpu().numpy()
                    test_corrects2 += batch_corrects2


                    last_predict = torch.argmax(predict, dim=1)
                    last_label = torch.argmax(cou, dim=1)
                    with open(os.path.join(result_dir, f'epoch{epoch}.txt'), "a") as f:
                        for i in range(len(name)):
                            f.write(name[i] + "\t" + str(last_label[i].item()) + "  " + str(
                                last_predict[i].item()) + "\n")

                test_loss = test_loss.float() / len(test_loader)
                test_acc1 = test_corrects1 / len(test_loader.dataset)
                test_acc2 = test_corrects2 / len(test_loader.dataset)
                test_acc_add = test_corrects_add / len(test_loader.dataset)
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
                _, MAE, MSE, mae_mse_report = report_mae_mse(l_true, l_pred, y_true)

                # if True:
                #    log.write(str(pre_se_sp_yi_report) + '\n')
                #    log.write(str(pre_se_sp_yi_report_m) + '\n')
                log.write(str(mae_mse_report) + '\n')
                if epoch == lr_steps[-1] - 1:
                    number = O.numpy()
                    np.savetxt(os.path.join(result_dir, 'O.txt'), number, fmt='%d', delimiter=',')
        # '''train
        cnn.train()
        print(len(train_loader))
        for step, (b_x, b_y, b_l, _) in enumerate(train_loader):  # x是图像，y是等级，l是个数

            b_x = b_x.to(device)
            b_l = b_l.numpy()
            print(b_y.shape[0])
            # generating D
            b_l = b_l - 1
            D = genLD(b_l, sigma, 'klloss', NUM_LABELS)  # D是个数的分布，D= [B,65]
            D = torch.from_numpy(D).to(device).float()

            # generating L
            L = np.zeros((b_x.shape[0], NUM_CLASSES))

            L[[x for x in range(b_x.shape[0])], b_y] = 1
            L = torch.from_numpy(L).to(device).float()  # L是要预测的等级

            # train
            cnn.train()

            cls, cou = cnn(b_x, None)  # nn output
            loss_cls = kl_loss_1(torch.log(cls), L) * NUM_CLASSES  # 模型直接预测的类别与真实类别的差异
            loss_cou = kl_loss_2(torch.log(cou), D) * NUM_LABELS  # 模型预测的个数与真实个数的差异
            with torch.no_grad():
                O, start_end_array = markov(cou, O, L, start_end_array, kl_loss_3)
            loss_count_cou = kl_loss_3((torch.log(torch.matmul(cou, O))), L)
            # log.write(eval(torch.matmul(cou, O),cls,L))
            # loss_cls_cou = kl_loss_3(torch.log(cou2cls), ld_4) * NUM_CLASSES  #模型预测的个数加权与真实个数的差异
            loss = (loss_cls+loss_count_cou) * lam + loss_cou * (1.0 - lam)
            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients

            losses_cls.update(loss_cls.item(), b_x.size(0))
            losses_cou.update(loss_cou.item(), b_x.size(0))
            losses_cou2cls.update(loss_count_cou.item(), b_x.size(0))
            losses.update(loss.item(), b_x.size(0))

        message = '%s %6.0f | %0.3f | %0.3f | %0.3f | %0.3f | %s\n' % ( \
            "train", epoch,
            losses_cls.avg,
            losses_cou.avg,
            losses_cou2cls.avg,
            losses.avg,
            time_to_str((timer() - start), 'min'))
        log.write(message)
        '''
        with torch.no_grad():

            trainDistribution = []
            trainFeature = []
            for step, (b_x, b_y, b_l, _) in enumerate(train_loader):  # x是图像，y是等级，l是个数
                _, cou = cnn(b_x, None)
                trainFeature.append(cou)
                trainDistribution.append(b_l)
            trainDistribution = np.concatenate(trainDistribution)
            trainFeature = np.concatenate(trainFeature)
            trainNum = trainDistribution.shape[0]

            testDistribution = []
            testFeature = []
            for step, (b_x, b_y, b_l, _) in enumerate(train_loader):  # x是图像，y是等级，l是个数
                _, cou = cnn(b_x, None)
                testFeature.append(cou)
                testDistribution.append(b_l)
            testDistribution = np.concatenate(testDistribution)
            testFeature = np.concatenate(testFeature)
            testNum = testDistribution.shape[0]
            name = 'Dou'
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
        '''

def main():
    # Hyper Parameters
    BATCH_SIZE = 4
    LR = 0.001  # learning rate
    NUM_WORKERS = 1  # 12
    NUM_CLASSES = 4
    NUM_LABELS = 65
    DATA_PATH = 'dataset/acne4/Classification/JPEGImages'
    cur_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    result_dir = os.path.join('results/acne4', cur_time)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    lr_steps = [25, 50, 75, 100, 125]

    np.random.seed(42)

    cross_val_lists = ['0']
    for cross_val_index in cross_val_lists:
        result_dir = os.path.join(result_dir, cross_val_index)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        LOG_FILE_NAME = os.path.join(result_dir, 'loss.log')
        log = Logger()
        log.open(LOG_FILE_NAME, mode="w")
        log.write('\n\ncross_val_index: ' + cross_val_index + '\n\n')
        if True:
            trainval_test(cross_val_index, sigma=30 * 0.1, lam=6 * 0.1, log=log, result_dir=result_dir,
                          batch_size=BATCH_SIZE, lr=LR, num_clases=NUM_CLASSES, num_labels=65, rate=1)


if __name__ == '__main__':
    main()

