import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, Conv3d, LayerNorm

ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu}
class Mlp(nn.Module):
    def __init__(self, input_size, hidden_size, num_label, num_classes):
        super(Mlp, self).__init__()
        self.fc1 = Linear(input_size, hidden_size)
        self.fc2 = Linear(hidden_size, num_label)
        self.fc3 = Linear(hidden_size, num_classes)
        self.act_fn = ACT2FN["gelu"]  # torch.nn.functional.gelu
        self.dropout = Dropout(p=0.1)  # Dropout(p=0.1)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        label = self.fc2(x)
        classes = self.fc3(x)
        label = F.softmax(label,dim=1)
        classes = F.softmax(classes,dim=1)
        return label, classes