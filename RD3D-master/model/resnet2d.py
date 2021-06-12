import torch.nn as nn
import math
import torch


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


# class BasicBlock(nn.Module):
#     expansion = 1

#     def __init__(self, inplanes, planes, stride=1, downsample=None):
#         super(BasicBlock, self).__init__()
#         self.conv1 = conv3x3(inplanes, planes, stride)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = conv3x3(planes, planes)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.downsample = downsample
#         self.stride = stride

#     def forward(self, x):
#         residual = x

#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)

#         out = self.conv2(out)
#         out = self.bn2(out)

#         if self.downsample is not None:
#             residual = self.downsample(x)

#         out += residual
#         out = self.relu(out)

#         return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

# class BottleneckKDA(nn.Module):
#     expansion = 4

#     def __init__(self, inplanes, planes, stride=1, downsample=None):
#         super(BottleneckKDA, self).__init__()
#         self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(planes)

#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
#                                padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(planes)

#         self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(planes * 4)

#         self.relu = nn.ReLU(inplace=True)
#         self.downsample = downsample
#         self.stride = stride
#         inp, oup = inplanes, planes * 4
#         self.squeeze = inp // 16
#         self.dim = int(math.sqrt(inp))
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)#通道前后不变，尺寸变为1，b,c,h,w->b,c,1,1
#         self.fc = nn.Sequential(
#             nn.Linear(inp, self.squeeze, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Linear(self.squeeze, 4, bias=False),
#         )#全连接层，输出为4
#         self.sf = nn.Softmax(dim=1)
#         #Φ
#         self.conv_s1 = nn.Conv2d(planes, planes, 1, stride=stride, padding=0, bias=False)
#         self.conv_s2 = nn.Conv2d(planes, planes, 1, stride=stride, padding=0, bias=False)
#         self.conv_s3 = nn.Conv2d(planes, planes, 1, stride=stride, padding=0, bias=False)
#         self.conv_s4 = nn.Conv2d(planes, planes, 1, stride=stride, padding=0, bias=False)

#     def forward(self, x):
#         #attention branch
#         #b,4,1,1,1
#         b, c, h, w = x.size()
#         y = self.fc(self.avg_pool(x).view(b, c)).view(b, 4, 1, 1, 1)
#         y = self.sf(y)

#         residual = x
#         #经过一个1×1的卷积，W0
#         out = self.conv1(x)#b,c,h,w-->b,planes,h,w
#         out = self.bn1(out)
#         out = self.relu(out)

#         #conv_si（）求的是Φi
#         #subspace routing 
#         #▲W0
#         dyres = self.conv_s1(out)*y[:,0] + self.conv_s2(out)*y[:,1] + \
#             self.conv_s3(out)*y[:,2] + self.conv_s4(out)*y[:,3]
#         #▲W0+W0
#         out = dyres + self.conv2(out)#b,planes,h,w

#         out = self.bn2(out)
#         out = self.relu(out)

#         out = self.conv3(out)#b,4*planes,h,w
#         out = self.bn3(out)
        
#         #如果downsample不为空，就执行
#         if self.downsample is not None:
#             # import ipdb;ipdb.set_trace()
#             residual = self.downsample(x)

#         out += residual #+ dyres
#         out = self.relu(out)

#         return out

class ResNet50(nn.Module):
    # ResNet with two branches
    def __init__(self,resnet):
        self.inplanes = 64
        super(ResNet50, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(Bottleneck, 64, 3)
        self.layer2 = self._make_layer(Bottleneck, 128, 4, stride=2)
        self.layer3 = self._make_layer(Bottleneck, 256, 6, stride=2)
        self.layer4 = self._make_layer(Bottleneck, 512, 3, stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
        
        self.new_param, self.init_param= [], []
        if resnet:
            print ('loading pretrain model from %s' %(resnet))
            model = torch.load(resnet)#['state_dict']
            # import ipdb;ipdb.set_trace()
            # prefix = 'module.features.'
            new_params = self.state_dict().copy()
            for x in new_params:
                if x in model:
                    new_params[x] = model[x]
                    self.init_param.append(x)
                else:
                    self.new_param.append(x)
                    print (x)
            self.load_state_dict(new_params)
        # import ipdb;ipdb.set_trace()


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x
