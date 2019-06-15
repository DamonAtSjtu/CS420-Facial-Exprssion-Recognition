# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 21:12:53 2019

@author: 黄浩威
"""

from __future__ import print_function


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import numpy as np
import transforms as transforms
import os
import argparse
import utils
from fer import FER2013
from torch.autograd import Variable
from models import *
from adversarialbox.utils import to_var

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
        Modell.cuda()
        scores = Modell(X_var)
        #loss = Modell.loss_fn(scores,y_var)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(scores,y_var)
        loss.backward()
        grad_sign = X_var.grad.data.cpu().sign().numpy()

        X += epsilon * grad_sign
        X = np.clip(X, 0, 1)

        return X

parser = argparse.ArgumentParser(description='PyTorch Fer2013 CNN Training')
parser.add_argument('--model', type=str, default='VGG19', help='CNN architecture')
parser.add_argument('--dataset', type=str, default='FER2013', help='CNN architecture')
parser.add_argument('--bs', default=32, type=int, help='batch size')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--data_url', type=str, default='s3://test-pytorch/fer-2013-data/data/',help='the training data')
parser.add_argument('--init_method', type=str, default='tcp://job73079afd-job-trainjob-983f-0:6666',help='init')
parser.add_argument('--train_url', type=str, default='s3://test-pytorch/fer-2013-data/model/V0006/',help='model saved path')
opt = parser.parse_args([])

log_path = "model_ad_log.txt"
use_cuda = torch.cuda.is_available()
best_PublicTest_acc = 0  # best PublicTest accuracy
best_PublicTest_acc_epoch = 0
best_PrivateTest_acc = 0  # best PrivateTest accuracy
best_PrivateTest_acc_epoch = 0
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

learning_rate_decay_start = 80  # 50
learning_rate_decay_every = 5 # 5
learning_rate_decay_rate = 0.9 # 0.9

cut_size = 44
total_epoch = 251

path = os.path.join(opt.dataset + '_' + opt.model)
#path = '/home/work/user-job-dir/train_py'+path
'''
lis = os.listdir
print(lis('/home/work/user-job-dir'))
print(lis('/home/work/user-job-dir/train_py/data'))
print(lis('/home/work'))
'''
# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(44),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.TenCrop(cut_size),
    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
])

trainset = FER2013(split = 'Training', transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.bs, shuffle=True)
PublicTestset = FER2013(split = 'PublicTest', transform=transform_test)
PublicTestloader = torch.utils.data.DataLoader(PublicTestset, batch_size=opt.bs, shuffle=False)
PrivateTestset = FER2013(split = 'PrivateTest', transform=transform_test)
PrivateTestloader = torch.utils.data.DataLoader(PrivateTestset, batch_size=opt.bs, shuffle=False)

# Model
if opt.model == 'VGG19':
    net = VGG('VGG19')
elif opt.model  == 'Resnet18':
    net = ResNet18()

if opt.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir(path), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(os.path.join(path,'PrivateTest_model.t7'))

    net.load_state_dict(checkpoint['net'])
    best_PublicTest_acc = checkpoint['best_PublicTest_acc']
    best_PrivateTest_acc = checkpoint['best_PrivateTest_acc']
    best_PrivateTest_acc_epoch = checkpoint['best_PublicTest_acc_epoch']
    best_PrivateTest_acc_epoch = checkpoint['best_PrivateTest_acc_epoch']
    start_epoch = checkpoint['best_PrivateTest_acc_epoch'] + 1
else:
    print('==> Building model..')

if use_cuda:
    net.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=opt.lr, momentum=0.9, weight_decay=5e-4)

temp_train_loss = []
temp_train_acc = []
temp_putest_loss = []
temp_putest_acc = []
temp_prtest_loss = []
temp_prtest_acc = []
# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    global Train_acc
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    if epoch > learning_rate_decay_start and learning_rate_decay_start >= 0:
        frac = (epoch - learning_rate_decay_start) // learning_rate_decay_every
        decay_factor = learning_rate_decay_rate ** frac
        current_lr = opt.lr * decay_factor
        utils.set_lr(optimizer, current_lr)  # set the decayed rate
    else:
        current_lr = opt.lr
    print('learning_rate: %s' % str(current_lr))

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        bs, c, h, w = np.shape(inputs)
        spl = int(bs/2)
        input1 = inputs[0:spl]
        label1 = targets[0:spl]
        input2 = inputs[spl:bs]
        label2 = targets[spl:bs]
        input1 = perturb(input1,label1,0.2,net)
        input1 = torch.Tensor(input1)
        inputs = np.vstack((input1,input2))
        inputs = torch.Tensor(inputs)
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        utils.clip_gradient(optimizer, 0.1)
        optimizer.step()
        train_loss += loss.data.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        '''
        utils.progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        '''

    Train_acc = 100.*correct/total
    if((epoch+1)%10==0):
        Ta = float(Train_acc)
        temp_train_acc.append(Ta)
        temp_train_loss.append(train_loss)

def PublicTest(epoch):
    global PublicTest_acc
    global best_PublicTest_acc
    global best_PublicTest_acc_epoch
    net.eval()
    PublicTest_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(PublicTestloader):
        bs, ncrops, c, h, w = np.shape(inputs)
        spl = int(bs/2)
        input1 = inputs[0:spl]
        label1 = targets[0:spl]
        input2 = inputs[spl:bs]
        label2 = targets[spl:bs]
        input1 = np.reshape(input1,newshape = [10*spl,3,44,44])
        label = []
        for i in range(spl):
            for j in range(10):
                label.append(targets[i].item())
        label = torch.LongTensor(label)
        input1 = perturb(input1,label,0.2,net)
        input1 = torch.Tensor(input1)
        input1 = np.reshape(input1, newshape = [spl,10,3,44,44])
        inputs = np.vstack((input1,input2))
        inputs = torch.Tensor(inputs)
        inputs = inputs.view(-1, c, h, w)
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)
        outputs_avg = outputs.view(bs, ncrops, -1).mean(1)  # avg over crops
        loss = criterion(outputs_avg, targets)
        PublicTest_loss += loss.data.item()
        _, predicted = torch.max(outputs_avg.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        '''
        utils.progress_bar(batch_idx, len(PublicTestloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                           % (PublicTest_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        '''

    # Save checkpoint.
    PublicTest_acc = 100.*correct/total
    if((epoch+1)%10==0):
        Put = float(PublicTest_acc)
        temp_putest_loss.append(PublicTest_loss)
        temp_putest_acc.append(Put)
    if PublicTest_acc > best_PublicTest_acc:
        print('Saving..')
        print("best_PublicTest_acc: %0.3f" % PublicTest_acc)
        state = {
            'net': net.state_dict() if use_cuda else net,
            'acc': PublicTest_acc,
            'epoch': epoch,
        }
        if not os.path.isdir(path):
            os.mkdir(path)
        torch.save(state, os.path.join(path,'PublicTest_ad251_model.t7'))
        best_PublicTest_acc = PublicTest_acc
        best_PublicTest_acc_epoch = epoch

def PrivateTest(epoch):
    global PrivateTest_acc
    global best_PrivateTest_acc
    global best_PrivateTest_acc_epoch
    net.eval()
    PrivateTest_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(PrivateTestloader):
        bs, ncrops, c, h, w = np.shape(inputs)
        spl = int(bs/2)
        input1 = inputs[0:spl]
        label1 = targets[0:spl]
        input2 = inputs[spl:bs]
        label2 = targets[spl:bs]
        input1 = np.reshape(input1,newshape = [10*spl,3,44,44])
        label = []
        for i in range(spl):
            for j in range(10):
                label.append(targets[i].item())
        label = torch.LongTensor(label)
        input1 = perturb(input1,label,0.2,net)
        input1 = torch.Tensor(input1)
        input1 = np.reshape(input1, newshape = [spl,10,3,44,44])
        inputs = np.vstack((input1,input2))
        inputs = torch.Tensor(inputs)
        inputs = inputs.view(-1, c, h, w)
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)
        outputs_avg = outputs.view(bs, ncrops, -1).mean(1)  # avg over crops
        loss = criterion(outputs_avg, targets)
        PrivateTest_loss += loss.data.item()
        _, predicted = torch.max(outputs_avg.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
    # Save checkpoint.
    PrivateTest_acc = 100.*correct/total
    if((epoch+1)%10==0):
        Pt = float(PrivateTest_acc)
        temp_prtest_loss.append(PrivateTest_loss)
        temp_prtest_acc.append(Pt)
    #准确率更高则更新储存的模型
    if PrivateTest_acc > best_PrivateTest_acc:
        print('Saving..')
        print("best_PrivateTest_acc: %0.3f" % PrivateTest_acc)
        state = {
            'net': net.state_dict() if use_cuda else net,
	        'best_PublicTest_acc': best_PublicTest_acc,
            'best_PrivateTest_acc': PrivateTest_acc,
    	    'best_PublicTest_acc_epoch': best_PublicTest_acc_epoch,
            'best_PrivateTest_acc_epoch': epoch,
        }
        if not os.path.isdir(path):
            os.mkdir(path)
        torch.save(state, os.path.join(path,'PrivateTest_ad251_model.t7'))
        best_PrivateTest_acc = PrivateTest_acc
        best_PrivateTest_acc_epoch = epoch

for epoch in range(start_epoch, total_epoch):
    train(epoch)
    PublicTest(epoch)
    PrivateTest(epoch)
    
#储存训练日志

with open(log_path, 'w') as f:
    f.write('train_loss: ' + str(temp_train_loss))
    f.write('\n\nputest_loss: ' + str(temp_putest_loss))
    f.write('\n\nprtest_loss: ' + str(temp_prtest_loss))
    f.write('\n\ntrain_acc: ' + str(temp_train_acc))
    f.write('\n\nputest_acc: ' + str(temp_putest_acc))
    f.write('\n\nprtest_acc: ' + str(temp_prtest_acc))

print("best_PublicTest_acc: %0.3f" % best_PublicTest_acc)
print("best_PublicTest_acc_epoch: %d" % best_PublicTest_acc_epoch)
print("best_PrivateTest_acc: %0.3f" % best_PrivateTest_acc)
print("best_PrivateTest_acc_epoch: %d" % best_PrivateTest_acc_epoch)