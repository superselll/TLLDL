import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import copy

def transfer(O,Ostartend,i,dirction):
    mid = int(Ostartend[i + 1])
    if dirction == 'left':   #从左边扒拉一个
        O[mid - 1, 0] = 0
        O[mid - 1, 1] = 1
        Ostartend[i + 1] = Ostartend[i + 1] - 1
    elif dirction == 'right':   #从右边扒拉一个
        O[mid, 0] = 1
        O[mid, 1] = 0
        Ostartend[i + 1] += 1
    return O,Ostartend

def markov(D,O,L,Ostartend,loss):
    levels = O.shape[1]
    for i in range(levels-1):
        Olist = [O[:,[i,i+1]]]
        startendlist = [Ostartend]
        start = Ostartend[i]
        mid = Ostartend[i+1]
        end = Ostartend[i+2]
        if mid>start+1:  #从左边扒拉一个,导致左边减少
            Onew,Ostartendnew = transfer(copy.deepcopy(O[:,[i,i+1]]),copy.deepcopy(Ostartend),i,'left')
            Olist.append(Onew)
            startendlist.append(Ostartendnew)
        if mid+1<end: #从右边扒拉一个
            Onew,Ostartendnew = transfer(copy.deepcopy(O[:,[i,i+1]]),copy.deepcopy(Ostartend),i,'right')
            Olist.append(Onew)
            startendlist.append(Ostartendnew)

        probs = torch.zeros(len(startendlist))
        for j in range(len(startendlist)):
            probs[j] = 1.0/loss(torch.log(torch.matmul(D,Olist[j])),L[:,[i,i+1]])
        probs = F.softmax(probs)
        index = torch.argmax(probs)
        O[:,[i,i+1]] = Olist[index]
        Ostartend = startendlist[index]
    return O, Ostartend


def main():
    device =  "cuda"
    O = np.zeros((65,4))

    batchsize = O.shape[0]
    startend = np.linspace(start=0, stop=65, num=5,dtype=int)
    for i in range(4):
        O[startend[i]:startend[i+1],i] = 1
    print(O.sum(axis=0))
    print(O.sum(axis=1))
    O = torch.from_numpy(O).to(device).float()  #0-1矩阵
    O.requires_grad = False
    startend = torch.from_numpy(startend).to(device).float()

    D = torch.randn(size=(2,65)).to(device)
    L = torch.randn(size=(2,4)).to(device)
    kl_loss_3 = nn.KLDivLoss().to(device)
    markov(D, O, L, startend,kl_loss_3)

if __name__ == '__main__':

    main()