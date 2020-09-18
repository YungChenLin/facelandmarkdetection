## Define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I

def conv3x3(c1, c2, stride=1):
    return nn.Conv2d(c1, c2, kernel_size = 3, stride = stride, padding = 1, bias = False)

class ResBlk(nn.Module):
    ch_expansion = 1
    def __init__(self, ichs, chs, stride =1, downsample = None):
        super(ResBlk, self).__init__()
        
        self.conv1 = conv3x3(ichs, chs, stride)
        self.bn1 = nn.BatchNorm2d(chs)
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = conv3x3(chs, chs)
        self.bn2 = nn.BatchNorm2d(chs)
        self.stride = stride
        self.downsample = downsample

    def forward(self, x):
        res = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)

        if self.downsample is not None:
            res = self.downsample(res)   
        x = x + res

        x = self.relu(x)
        return x

class ResBlk3(nn.Module):
    ch_expansion = 4
    def __init__(self,  ichs, chs, stride = 1, downsample = None):
        super(ResBlk3, self).__init__()
        
        self.conv1 = nn.Conv2d(ichs, chs, kernel_size = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(chs)

        self.conv2 = nn.Conv2d(chs, chs, kernel_size = 3, stride = stride, padding =1, bias =False)
        self.bn2 = nn.BatchNorm2d(chs)

        self.conv3 = nn.Conv2d(chs, 4*chs, kernel_size = 1, bias = False)
        self.bn3 = nn.BatchNorm2d(4*chs)

        self.relu = nn.ReLU(inplace =True)

        self.downsample = downsample

    def forward(self, x):
        res = x
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu(y)

        y = self.conv2(y)
        y = self.bn2(y)
        y = self.relu(y)

        y = self.conv3(y)
        y = self.bn3(y)

        if self.downsample is not None:
            res = self.downsample(res)        

        y = y + res
        y = self.relu(y)
        return y

class Resnet(nn.Module):
    def __init__(self, resblk, repeats_list, img_ch=3, num_class = 3):
        super(Resnet, self).__init__()
        
        self.conv1 = conv3x3(img_ch, 64, stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(64, 64)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.inchs = 64
        self.layer1 = self._make_layer(resblk, repeats_list[0], 64)
        self.layer2 = self._make_layer(resblk, repeats_list[1], 128, blk_stride=2)
        self.layer3 = self._make_layer(resblk, repeats_list[2], 256, blk_stride=2)
        self.layer4 = self._make_layer(resblk, repeats_list[3], 512, blk_stride=2)

        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * resblk.ch_expansion, num_class)

    def _make_layer(self, resblk, num_repeat, blkchs, blk_stride = 1):
        
        downsample = None
        out_chs = blkchs*resblk.ch_expansion
        if blk_stride != 1 or self.inchs != blkchs * resblk.ch_expansion:
            # make down sample operation for the residual element 'x'
            # the down sample is implemented with 1x1 convolution
            downsample = nn.Sequential(
                nn.Conv2d(self.inchs, out_chs
                        ,kernel_size=1, stride=blk_stride, bias=False),
                nn.BatchNorm2d(out_chs),
            )

        resblks = []
        resblks.append(resblk(self.inchs, blkchs, stride = blk_stride, downsample= downsample))
        self.inchs = out_chs
        for i in range(1, num_repeat):
            resblks.append( resblk(self.inchs, blkchs) )

        return nn.Sequential(*resblks)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.net = Resnet(ResBlk, [2,2,2,2], 3, 68*2)

    def forward(self, x):
        return self.net(x)
# class Net(nn.Module):
#     def __init__(self, in_chs, out_chs):
#         super(Net, self).__init__()
        
#         self.conv1 = nn.Conv2d(3, 32, 5)
#         self.pool = nn.MaxPool2d(2, 2)
        
#         self.conv2 = nn.Conv2d(32, 64, 3) 
#         self.bn2 = nn.BatchNorm2d(64)
        
#         self.conv3 = nn.Conv2d(64, 128, 3)
#         self.bn3 = nn.BatchNorm2d(128)
        
#         self.conv4 = nn.Conv2d(128, 256, 3)
#         self.bn4 = nn.BatchNorm2d(256)
        
#         self.conv5 = nn.Conv2d(256, 512, 1)
#         self.bn5 = nn.BatchNorm2d(512)        
        
#         # Fully-connected (linear) layers
#         self.fc1 = nn.Linear(512*6*6, 1024)
#         self.fc2 = nn.Linear(1024, 512)
#         self.fc3 = nn.Linear(512, 68*2)
        

#     def forward(self, x):

#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.bn2(self.conv2(x))))
#         x = self.pool(F.relu(self.bn3(self.conv3(x))))
#         x = self.pool(F.relu(self.bn4(self.conv4(x))))
#         x = self.pool(F.relu(self.bn5(self.conv5(x))))
        
#         # Flatten
#         x = x.view(x.size(0), -1)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x




class ModelInitilizer:
    @staticmethod
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find('BatchNorm2d') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)

