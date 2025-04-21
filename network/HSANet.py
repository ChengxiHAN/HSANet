import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import matplotlib.pyplot as plt
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# self attention--spatial
class Atten_Spa(nn.Module):
    def __init__(self, in_dim):
        super(Atten_Spa, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):  # [2,512,16,16] [2,1,128,128]
        m_batchsize, C, height, width = x.size()

        query = self.query_conv(x)
        proj_query = query.view(m_batchsize, -1, width*height).permute(0, 2, 1)
        key = self.key_conv(x)
        proj_key = key.view(m_batchsize, -1, width*height)

        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        self.energy = energy
        self.attention = attention

        value = self.value_conv(x)
        proj_value = value.view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x

        return out


# cross attention--spatial w/ change guide map
class Atten_Cross(nn.Module):
    def __init__(self, in_dim):
        super(Atten_Cross, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, cond):  # [2,512,16,16] [2,1,128,128]
        m_batchsize, C, height, width = x.size()

        # guiding_map0 = F.interpolate(guiding_map0, x.size()[2:], mode='bilinear', align_corners=True) # map 2,1,128,128 -> 2,1,16,16
        #
        # guiding_map = F.sigmoid(guiding_map0)

        query = self.query_conv(x) # query from x
        proj_query = query.view(m_batchsize, -1, width*height).permute(0, 2, 1)
        key = self.key_conv(cond)  # key from cond
        proj_key = key.view(m_batchsize, -1, width*height)

        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        self.energy = energy
        self.attention = attention

        value = self.value_conv(cond) # value from cond
        proj_value = value.view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x

        return out

#增加decoder解码器，不concat多尺度特征生成guide map,也concat多尺度特征生成最后输出
class HSANet(nn.Module):
    def __init__(self,):
        super(HSANet, self).__init__()
        vgg16_bn = models.vgg16_bn(pretrained=True)
        self.inc = vgg16_bn.features[:5]  # 64
        self.down1 = vgg16_bn.features[5:12]  # 128
        self.down2 = vgg16_bn.features[12:22]  # 256
        self.down3 = vgg16_bn.features[22:32]  # 512
        self.down4 = vgg16_bn.features[32:42]  # 512

        self.conv_reduce_1 = BasicConv2d(128*2,128,3,1,1)
        self.conv_reduce_2 = BasicConv2d(256*2,256,3,1,1)
        self.conv_reduce_3 = BasicConv2d(512*2,512,3,1,1)
        self.conv_reduce_4 = BasicConv2d(512*2,512,3,1,1)

        self.up_layer4 = BasicConv2d(512,512,3,1,1)
        self.up_layer3 = BasicConv2d(512,512,3,1,1)
        self.up_layer2 = BasicConv2d(256,256,3,1,1)

        self.decoder = nn.Sequential(BasicConv2d(512,64,3,1,1),
                                     nn.Conv2d(64,1,3,1,1))

        self.decoder_final = nn.Sequential(BasicConv2d(128, 64, 3, 1, 1),
                                           nn.Conv2d(64, 1, 1))

        self.cgm_1 = Atten_Cross(128)
        self.cgm_2 = Atten_Cross(256)
        self.cgm_3 = Atten_Cross(512)
        self.cgm_4 = Atten_Cross(512)

        self.sa_1 = Atten_Spa(128)
        self.sa_2 = Atten_Spa(256)
        self.sa_3 = Atten_Spa(512)
        self.sa_4 = Atten_Spa(512)

        #相比v2 额外的模块
        self.upsample2x=nn.UpsamplingBilinear2d(scale_factor=2)
        self.decoder_module4 = BasicConv2d(1024,512,3,1,1)
        self.decoder_module3 = BasicConv2d(768,256,3,1,1)
        self.decoder_module2 = BasicConv2d(384,128,3,1,1)

    def forward(self,A,B):

        size = A.size()[2:]
        layer1_pre = self.inc(A) # 2,64,256,256
        layer1_A = self.down1(layer1_pre) # 2,128,128,128
        layer2_A = self.down2(layer1_A) # 2,256,64,64
        layer3_A = self.down3(layer2_A) # 2,512,32,32
        layer4_A = self.down4(layer3_A) # 2,512,16,16

        layer1_pre = self.inc(B)
        layer1_B = self.down1(layer1_pre)
        layer2_B = self.down2(layer1_B)
        layer3_B = self.down3(layer2_B)
        layer4_B = self.down4(layer3_B)

        # Concatenate features from A and B
        layer1 = self.conv_reduce_1(torch.cat((layer1_B, layer1_A), dim=1)) # 2,128,128,128
        layer2 = self.conv_reduce_2(torch.cat((layer2_B, layer2_A), dim=1)) # 2,256,64,64
        layer3 = self.conv_reduce_3(torch.cat((layer3_B, layer3_A), dim=1)) # 2,512,32,32
        layer4 = self.conv_reduce_4(torch.cat((layer4_B, layer4_A), dim=1)) # 2,512,16,16

        # change semantic guiding map 这部分没用到
        layer4_1 = F.interpolate(layer4, layer1.size()[2:], mode='bilinear', align_corners=True) # 2,512,128,128
        feature_fuse=layer4_1 #需要注释！
        change_map = self.decoder(feature_fuse)  #2,1,128,128

        # self attention
        layer4_4 = self.sa_4(layer4)
        layer4_5 = self.cgm_4(layer4_4,layer4)
        feature4 = self.decoder_module4(torch.cat([self.upsample2x(layer4_5), layer3], 1))

        layer3_3 = self.sa_3(feature4)
        layer3_4 = self.cgm_3(layer3_3,layer3)
        feature3 = self.decoder_module3(torch.cat([self.upsample2x(layer3_4),layer2],1))

        layer2_3 = self.sa_2(feature3)
        layer2_4 = self.cgm_2(layer2_3,layer2)
        feature2 = self.decoder_module2(torch.cat([self.upsample2x(layer2_4), layer1], 1))

        change_map = F.interpolate(change_map, size, mode='bilinear', align_corners=True)
        final_map = self.decoder_final(feature2)

        final_map = F.interpolate(final_map, size, mode='bilinear', align_corners=True)

        return change_map, final_map


if __name__=='__main__':
    #测试热图
    net = HSANet().cuda()
    out, _ = net(torch.rand((2, 3, 256, 256)).cuda(), torch.rand((2, 3, 256, 256)).cuda())
    print(out.size())

