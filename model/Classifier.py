'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.layer1 = nn.Linear(1024, 512)
        self.layer2 = nn.Linear(512,7)

    def forward(self, x):
        out = F.dropout(x, p=0.5, training=self.training)
        out = self.layer1(out)
        out = self.layer2(out)
        return out

