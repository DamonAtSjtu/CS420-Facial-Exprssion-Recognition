from __future__ import print_function

import copy
import numpy as np
from collections import Iterable
from scipy.stats import truncnorm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from adversarialbox.utils import to_var

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import transforms as transforms
import numpy as np
import os
import argparse
import utils
from fer import FER2013
from torch.autograd import Variable
from models import *
from PIL import Image

def perturb(X_nat, y, epsilon,Modell=None):
        """
        Given examples (X_nat, y), returns their adversarial
        counterparts with an attack length of epsilon.
        """
        '''
        # Providing epsilons in batch
        if epsilons is not None:
            self.epsilon = epsilons
        '''
        X = np.copy(X_nat)

        X_var = to_var(torch.from_numpy(X), requires_grad=True)
        y_var = to_var(torch.LongTensor(y))
        '''
        scores = self.model(X_var)
        loss = self.loss_fn(scores, y_var)
        '''
        if Modell == None:
            Modell = VGG('VGG19')
        scores = Modell(X_var)
        #loss = Modell.loss_fn(scores,y_var)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(scores,y_var)
        loss.backward()
        grad_sign = X_var.grad.data.cpu().sign().numpy()

        X += epsilon * grad_sign
        X = np.clip(X, 0, 1)

        return X

cut_size = 44
transform_train = transforms.Compose([
    #transforms.RandomCrop(44),
    #transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.TenCrop(cut_size),
    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
])

trainset = FER2013(split = 'Training', transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=0)
PublicTestset = FER2013(split = 'PublicTest', transform=transform_test)
PublicTestloader = torch.utils.data.DataLoader(PublicTestset, batch_size=128, shuffle=False, num_workers=0)
PrivateTestset = FER2013(split = 'PrivateTest', transform=transform_test)
PrivateTestloader = torch.utils.data.DataLoader(PrivateTestset, batch_size=128, shuffle=False, num_workers=0)
dataiter = iter(trainloader)
imgs, labels = next(dataiter)
unloader = transforms.ToPILImage()
print(np.shape(imgs))
print(np.shape(labels))
image = unloader(imgs[0])

plt.imshow(image)
plt.pause(0.3)

Modell = VGG('VGG19')
Modell.load_state_dict(torch.load('./FER2013_VGG19/PublicTest_model.t7',map_location='cpu')['net'])
Returnnn = perturb(imgs,labels,0.2,Modell=Modell)
Returnnn = torch.Tensor(Returnnn)

image = unloader(Returnnn[0])
plt.imshow(image)
plt.pause(1)
'''
Returnnn = perturb(imgs,labels,0.005)
Returnnn = torch.Tensor(Returnnn)

image = unloader(Returnnn[0])
plt.imshow(image)
plt.pause(0.3)
'''
print("hello")