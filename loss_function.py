import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import xlwt
import copy

class chebyshev(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        temp = abs(input - target)
        return torch.mean(torch.max(temp, 1).values)

class clark(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        temp = input - target
        temp = temp*temp
        temp2 = input+target
        temp2 = temp2*temp2
        temp = temp/temp2
        return torch.mean(torch.sqrt(torch.sum(temp,1)))

class canberra(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        temp = abs(input - target) / (input + target)
        return torch.mean(torch.sum(temp, 1))

class kldist(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, target, input):
        temp = input*torch.log(input/target)
        return torch.mean(torch.sum(temp, 1))

class cosine(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, input, target):
        inner = torch.sum(input*target,1)
        len = torch.sqrt(torch.sum(input*input,1))*torch.sqrt(torch.sum(target*target,1))
        return torch.mean(inner/len)

class Euclidean(nn.Module):
    def __init__(self):
        super().__init__()
        self.func = nn.PairwiseDistance(p=2)
    def forward(self,input,target):
        return torch.mean(self.func(input, target))


class Sorensen(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,input,target):
        temp = abs(input-target)/(input+target)
        return torch.mean(torch.sum(temp,1))
class SquaredX2(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,input,target):
        temp = (input-target)**2/(input+target)
        return torch.mean(torch.sum(temp,1))

class KL(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,input,target):
        return nn.KLDivLoss()(input.log() ,target)

class Intersection(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,input, target):
        min = input<target
        temp = copy.deepcopy(target)
        temp[min] = input[min]

        return torch.mean(torch.sum(temp,1))

class Fidelity(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, input, target):
        temp = torch.sqrt(input*target)
        return torch.mean(torch.sum(temp, 1))



def main():
    a = torch.tensor([[1,2,3,8],[1,2,4,0]]).float()
    b = torch.tensor([[1,2,6,4],[1,2,4,0]]).float()
    a = F.softmax(a,dim=1)
    b = F.softmax(b,dim=1)
    loss1 = chebyshev()
    loss2 = clark()
    loss3 = canberra()
    loss4 = KL()#kldist()
    loss5 = cosine()
    loss6 = Intersection()
    print(loss1(a, b))
    print(loss2(a, b))
    print(loss3(a, b))
    print(loss4(a, b))
    print(loss5(a, b))
    print(loss6(a, b))
    loss1 = Euclidean()
    loss2 = Sorensen()
    loss3 = SquaredX2()
    loss4 = KL()

    loss5 = Fidelity()
    loss6 = Intersection()
    print(loss1(a, b))
    print(loss2(a,b))
    print(loss3(a,b))
    print(loss4(a, b))
    print(loss5(a, b))
    print(loss6(a, b))


def eval(ldl_predict,cls,classes01):
    loss1 = chebyshev()
    loss2 = clark()
    loss3 = canberra()
    loss4 = KL()
    loss5 = cosine()
    loss6 = Intersection()
    res = []
    res.append(loss1(cls, classes01))
    res.append(loss1(ldl_predict, classes01))
    res.append(loss2(cls, classes01))
    res.append(loss2(ldl_predict, classes01))
    res.append(loss3(cls, classes01))
    res.append(loss3(ldl_predict, classes01))
    res.append(loss4(cls, classes01))
    res.append(loss4(ldl_predict, classes01))
    res.append(loss5(cls, classes01))
    res.append(loss5(ldl_predict, classes01))
    res.append(loss6(cls, classes01))
    res.append(loss6(ldl_predict, classes01))



    return torch.tensor(res)

def save(save_path,data):
    #data = np.concatenate(all, 0).reshape(-1, len(epochs))
    book = xlwt.Workbook(encoding='utf-8', style_compression=0)
    sheet = book.add_sheet('sheet1', cell_overwrite_ok=True)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            sheet.write(i, j, data[i][j])
    #save_path = 'data.xls'
    book.save(save_path)

if __name__ == '__main__':
    main()