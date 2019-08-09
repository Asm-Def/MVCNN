from Data.classnames import classnames
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from Model.Model import Model

class SVCNN(Model):
    def __init__(self, name, nclasses=40, pretraining=True, cnn_name='vgg11'):
        super(SVCNN, self).__init__(name)
        self.nclasses = nclasses
        self.pretraining = pretraining
        self.cnn_name = cnn_name
        self.use_resnet = cnn_name.startswith('resnet')

        if self.use_resnet:
            if self.cnn_name == 'resnet18':
                self.net = models.resnet18(pretrained=self.pretraining)
                self.net.fc = nn.Linear(512,nclasses)
            elif self.cnn_name == 'resnet34':
                self.net = models.resnet34(pretrained=self.pretraining)
                self.net.fc = nn.Linear(512,nclasses)
            elif self.cnn_name == 'resnet50':
                self.net = models.resnet50(pretrained=self.pretraining)
                self.net.fc = nn.Linear(2048,nclasses)
        else:
            if self.cnn_name == 'alexnet':
                alexnet = models.alexnet(pretrained=self.pretraining)
                self.net_1 = alexnet.features
                self.net_2 = alexnet.classifier
            elif self.cnn_name == 'vgg11':
                vgg11 = models.vgg11(pretrained=self.pretraining)
                self.net_1 = vgg11.features
                self.net_2 = vgg11.classifier
            elif self.cnn_name == 'vgg16':
                vgg16 = models.vgg16(pretrained=self.pretraining)
                self.net_1 = vgg16.features
                self.net_2 = vgg16.classifier

            self.net_2._modules['6'] = nn.Linear(4096,nclasses)

    def forward(self, x):
        if self.use_resnet:
            return self.net(x)
        else:
            y = self.net_1(x)
            y = F.adaptive_avg_pool2d(y, (7, 7))
            return self.net_2(y.view(y.shape[0],-1))


class MVCNN(Model):
    def __init__(self, name, model, nclasses=40, cnn_name='vgg11', num_views = 12):
        super(MVCNN, self).__init__(name)
        self.nclasses = nclasses
        self.num_views = num_views
        self.use_resnet = cnn_name.startswith('resnet')

        if self.use_resnet:
            self.net_1 = nn.Sequential(*list(model.net.children())[:-1])
            self.net_2 = model.net.fc
        else:
            self.net_1 = model.net_1
            self.net_2 = model.net_2

    def forward(self, x):
        y = self.net_1(x)
        y = y.view((int(x.shape[0]/self.num_views), self.num_views, y.shape[-3], y.shape[-2], y.shape[-1]))
        return self.net_2(F.adaptive_avg_pool2d(self.max(y, 1)[0].view(y.shape[0], -1), (7, 7)))

class MVCNN_CNN(Model):
    def __init__(self, name, model, nclasses=40, cnn_name='vgg11', num_views = 12):
        super(MVCNN_CNN, self).__init__(name)
        self.nclasses = nclasses
        self.num_views = num_views
        self.use_resnet = cnn_name.startswith('resnet')

        if self.use_resnet:
            self.net_1 = nn.Sequential(*list(model.net.children())[:-1])
            self.net_2 = model.net.fc
            raise Exception('not tested')
        else:
            self.net_1 = nn.Sequential(model.net_1, nn.AdaptiveAvgPool2d((4,4)))
            self.net_2 = nn.Sequential(nn.Conv1d(16 * 512, 4096, 12), nn.MaxPool1d(12))
            self.net_3 = nn.Sequential(
                *list(model.net_2.children())[1:-1],
                nn.Linear(4096, 40)
            )

    def forward(self, x):
        y = self.net_1(x)  # (patch*num_views, 512, 4, 4)
        y = y.reshape((y.shape[0], y.shape[1], -1)).transpose(0, 2)  # (16, 512, patch*num_views)
        y = y.reshape((y.shape[0], y.shape[1], -1, self.num_views))
        y = torch.cat((y, y[:, :, :, :-1]), 3)  # (16, 512, patch, num_views+num_views-1)
        y = y.transpose(0, 2)  # (patch, 16, 512, ...)
        y = y.reshape(y.shape[0], y.shape[1]*y.shape[2], -1)  # (patch, 16*512, ...)
        y = self.net_2(y).squeeze(2)  # (patch, 4096)
        return self.net_3(y)
