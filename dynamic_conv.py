from turtle import xcor
import torch
import torch.nn as nn
import torch.nn.functional as F


from utils import *
class attention2d(nn.Module):
    def __init__(self, in_planes, ratios, K, temperature, init_weight=True):
        super(attention2d, self).__init__()
        assert temperature%3==1
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        if in_planes!=3:
            hidden_planes = int(in_planes*ratios)+1
        else:
            hidden_planes = K
        self.fc1 = nn.Conv2d(in_planes, hidden_planes, 1, bias=False)
        # self.bn = nn.BatchNorm2d(hidden_planes)
        self.fc2 = nn.Conv2d(hidden_planes, K, 1, bias=True)
        self.temperature = temperature
        if init_weight:
            self._initialize_weights()
        self.mean = Averager()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m ,nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def updata_temperature(self):
        if self.temperature!=1:
            self.temperature -=3
            print('Change temperature to:', str(self.temperature))


    def forward(self, x):
        x = self.avgpool(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x).view(x.size(0), -1)
      

        return F.softmax(x/self.temperature, 1)


class Dynamic_conv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, ratio=0.25, stride=1, padding=0, dilation=1, groups=1, bias=True, K=4, temperature=34, init_weight=True, **kwargs):
        # self.args = kwargs['argu']
        super(Dynamic_conv2d, self).__init__()
        assert in_planes%groups==0
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.K = 4#self.args.n_clusters
        self.attention = attention2d(in_planes, ratio, self.K, temperature)
        # self.middle = self.args.middle
        self.weight1 = nn.Parameter(torch.randn(self.K, out_planes, in_planes//groups, kernel_size, kernel_size), requires_grad=True)
        # self.weight2 = nn.Parameter(torch.randn(self.K, out_planes, self.middle, kernel_size, kernel_size), requires_grad=True)
        # self.weight3 = nn.Parameter(torch.randn(self.K, out_planes, out_planes, kernel_size, kernel_size), requires_grad=True)
        # self.weight4 = nn.Parameter(torch.randn(self.K, out_planes, out_planes, kernel_size, kernel_size), requires_grad=True)

        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(out_planes)
        # self.bn2 = nn.BatchNorm2d(out_planes)
        # self.bn3 = nn.BatchNorm2d(out_planes)
        # self.bn4 = nn.BatchNorm2d(out_planes)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.K, out_planes))
        else:
            self.bias = None
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for i in range(self.K):
            nn.init.kaiming_uniform_(self.weight1[i])
            # nn.init.kaiming_uniform_(self.weight2[i])
            # nn.init.kaiming_uniform_(self.weight3[i])
            # nn.init.kaiming_uniform_(self.weight4[i])
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def update_temperature(self):
        self.attention.updata_temperature()

    def forward(self, x):

        softmax_attention = self.attention(x)

        # !!!!!!!!!!abandon the manipulating module
        # a = softmax_attention.shape
        # softmax_attention = torch.ones([a[0], a[1]]).cuda()

        batch_size, in_planes, height, width = x.size()
        identity = x
        x = x.view(1, -1, height, width)
        weight1 = self.weight1.view(softmax_attention.shape[-1], -1)
        # weight2 = self.weight2.view(softmax_attention.shape[-1], -1)
        # weight3 = self.weight3.view(softmax_attention.shape[-1], -1)
        # weight4 = self.weight4.view(softmax_attention.shape[-1], -1)
        # weight = self.weight.view(self.K, -1)


        # aggregate_weight1 = torch.mm(softmax_attention, weight1).unsqueeze(1).repeat(1,c,1,1,1,1).view(-1, self.in_planes, self.kernel_size, self.kernel_size)
        aggregate_weight1 = torch.mm(softmax_attention, weight1).view(-1, self.in_planes, self.kernel_size, self.kernel_size)
        # aggregate_weight2 = torch.mm(softmax_attention, weight2).view(-1, self.middle, self.kernel_size,
        #                                                               self.kernel_size)
        # aggregate_weight3 = torch.mm(softmax_attention, weight3).view(-1, self.out_planes, self.kernel_size,
        #                                                               self.kernel_size)
        # aggregate_weight4 = torch.mm(softmax_attention, weight4).view(-1, self.out_planes, self.kernel_size,
        #                                                               self.kernel_size)
        if self.bias is not None:
            aggregate_bias = torch.mm(softmax_attention, self.bias).view(-1)
            output = F.conv2d(x, weight=aggregate_weight1, bias=aggregate_bias, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups*batch_size)

        else:
            output = F.conv2d(x, weight=aggregate_weight1, bias=None, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups * batch_size)

        output = output.view(batch_size, self.out_planes, output.size(-2), output.size(-1))
        output = self.bn1(output)
        if identity.shape == output.shape:
            output = identity + output
        output = self.relu(output)

        # output = output.view(1, -1, height, width)#2
        # if self.bias is not None:
        #     aggregate_bias = torch.mm(softmax_attention, self.bias).view(-1)
        #     output = F.conv2d(output, weight=aggregate_weight2, bias=aggregate_bias, stride=self.stride,
        #                       padding=self.padding,
        #                       dilation=self.dilation, groups=self.groups * batch_size)
        # else:
        #     output = F.conv2d(output, weight=aggregate_weight2, bias=None, stride=self.stride, padding=self.padding,
        #                       dilation=self.dilation, groups=self.groups * batch_size)

        # output = output.view(batch_size, self.out_planes, output.size(-2), output.size(-1))
        # output = self.bn2(output)
        # # output = identity + output
        # output = self.relu(output)
        #
        # output = output.view(1, -1, height, width)#3
        # if self.bias is not None:
        #     aggregate_bias = torch.mm(softmax_attention, self.bias).view(-1)
        #     output = F.conv2d(output, weight=aggregate_weight3, bias=aggregate_bias, stride=self.stride, padding=self.padding,
        #                       dilation=self.dilation, groups=self.groups*batch_size)
        #
        # else:
        #     output = F.conv2d(output, weight=aggregate_weight3, bias=None, stride=self.stride, padding=self.padding,
        #                       dilation=self.dilation, groups=self.groups * batch_size)
        #
        # output = output.view(batch_size, self.out_planes, output.size(-2), output.size(-1))
        # output = self.bn3(output)
        # output = identity + output
        # output = self.relu(output)
        # # #
        # output = output.view(1, -1, height, width)#4
        # if self.bias is not None:
        #     aggregate_bias = torch.mm(softmax_attention, self.bias).view(-1)
        #     output = F.conv2d(output, weight=aggregate_weight4, bias=aggregate_bias, stride=self.stride, padding=self.padding,
        #                       dilation=self.dilation, groups=self.groups*batch_size)
        #
        # else:
        #     output = F.conv2d(output, weight=aggregate_weight4, bias=None, stride=self.stride, padding=self.padding,
        #                       dilation=self.dilation, groups=self.groups * batch_size)
        # output = output.view(batch_size, self.out_planes, output.size(-2), output.size(-1))
        # output = self.bn4(output)
        # output = identity + output
        # output = self.relu(output)


        return softmax_attention, output

