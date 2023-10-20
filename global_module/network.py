import torch
from torch import nn
from attontion import PAM_Module, CAM_Module,PAM_X_Module,PAM_Y_Module
import math

import sys
sys.path.append('../global_module/')
from activation import mish, gelu, gelu_new, swish


import sys
sys.path.append('../D3D_code/')
from dcn.modules.deform_conv import *
import functools

class ResBlock_3d(nn.Module):
    def __init__(self, nf):
        super(ResBlock_3d, self).__init__()
        self.dcn0 = DeformConvPack_d(nf, nf, kernel_size=(3,3,3), stride=1, padding=(1,1,1), dimension='HW')
        self.dcn1 = DeformConvPack_d(nf, nf, kernel_size=(3,3,3), stride=1, padding=(1,1,1), dimension='HW')
        # self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.mish=mish()

    def forward(self, x):
        return self.dcn1(self.mish(self.dcn0(x))) + x

class ResBlock_3d_2(nn.Module):
    def __init__(self, nf):
        super(ResBlock_3d_2, self).__init__()
        self.dcn0 = DeformConvPack_d(nf, nf, kernel_size=(3,3,1), stride=1, padding=(1,1,0), dimension='HW')
        self.dcn1 = DeformConvPack_d(nf, nf, kernel_size=(3,3,1), stride=1, padding=(1,1,0), dimension='HW')
        # self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.mish=mish()

    def forward(self, x):
        return self.dcn1(self.mish(self.dcn0(x))) + x



####################################################################################################
####################################################################################################
#                                        OUR NET                                                   #
####################################################################################################
####################################################################################################

class D3DTBTA_network(nn.Module):
    def __init__(self, band, classes, nf=8):
        super(D3DTBTA_network, self).__init__()

        self.name = 'D3DTBTA'

        self.input = nn.Sequential(nn.Conv3d(in_channels=1, out_channels=nf,kernel_size=(1, 1, 7), stride=(1, 1, 2)),
                                    nn.BatchNorm3d(nf,  eps=0.001, momentum=0.1, affine=True), # 动量默认值为0.1
                                    mish()) 

        self.residual_layer = self.make_layer(functools.partial(ResBlock_3d, nf), 1)

        self.residual_spec = self.make_layer(functools.partial(ResBlock_3d_2, 60), 1)
        self.residual_spa_x = self.make_layer(functools.partial(ResBlock_3d_2, 60), 1)
        self.residual_spa_y = self.make_layer(functools.partial(ResBlock_3d_2, 60), 1)



        self.conv_feature=nn.Sequential(nn.Conv3d(in_channels=nf, out_channels=24,padding=(0, 0, 3),kernel_size=(1, 1, 7), stride=(1, 1, 1)),
                                        nn.BatchNorm3d(24,  eps=0.001, momentum=0.1, affine=True), # 动量默认值为0.1
                                        mish()) 

        # spectral Branch
        self.conv11 =nn.Sequential(nn.Conv3d(in_channels=24, out_channels=12, padding=(0, 0, 3),kernel_size=(1, 1, 7), stride=(1, 1, 1)),
                                    nn.BatchNorm3d(12,  eps=0.001, momentum=0.1, affine=True), # 动量默认值为0.1
                                    mish())

        self.conv12 =nn.Sequential(nn.Conv3d(in_channels=36, out_channels=12, padding=(0, 0, 3),kernel_size=(1, 1, 7), stride=(1, 1, 1)),
                                    nn.BatchNorm3d(12,  eps=0.001, momentum=0.1, affine=True), # 动量默认值为0.1
                                    mish())
        
        self.conv13 =nn.Sequential(nn.Conv3d(in_channels=48, out_channels=12, padding=(0, 0, 3),kernel_size=(1, 1, 7), stride=(1, 1, 1)),
                                    nn.BatchNorm3d(12,  eps=0.001, momentum=0.1, affine=True), # 动量默认值为0.1
                                    mish())

        kernel_3d = math.ceil((band - 6) / 2)

        self.conv14 =nn.Sequential(nn.Conv3d(in_channels=60, out_channels=60, padding=(0, 0, 0),kernel_size=(1, 1, kernel_3d), stride=(1, 1, 1)),
                                    nn.BatchNorm3d(60,  eps=0.001, momentum=0.1, affine=True), # 动量默认值为0.1
                                    # mish()
                                    )

        # Spatial Branch x
        self.conv21 =nn.Sequential(nn.Conv3d(in_channels=24, out_channels=12, padding=(1, 0, 0),kernel_size=(3, 1, 1), stride=(1, 1, 1)),
                                    nn.BatchNorm3d(12,  eps=0.001, momentum=0.1, affine=True), # 动量默认值为0.1
                                    mish())

        self.conv22 =nn.Sequential(nn.Conv3d(in_channels=36, out_channels=12, padding=(1, 0, 0),kernel_size=(3, 1, 1), stride=(1, 1, 1)),
                                    nn.BatchNorm3d(12,  eps=0.001, momentum=0.1, affine=True), # 动量默认值为0.1
                                    mish())
        
        self.conv23 =nn.Sequential(nn.Conv3d(in_channels=48, out_channels=12, padding=(1, 0, 0),kernel_size=(3, 1, 1), stride=(1, 1, 1)),
                                    nn.BatchNorm3d(12,  eps=0.001, momentum=0.1, affine=True), # 动量默认值为0.1
                                    mish())

        self.conv24 =nn.Sequential(nn.Conv3d(in_channels=60, out_channels=60, padding=(0, 0, 0),kernel_size=(1, 1, kernel_3d), stride=(1, 1, 1)),
                                    nn.BatchNorm3d(60,  eps=0.001, momentum=0.1, affine=True), # 动量默认值为0.1
                                    # mish()
                                    )

        # Spatial Branch y
        self.conv31 =nn.Sequential(nn.Conv3d(in_channels=24, out_channels=12, padding=(0, 1, 0),kernel_size=(1, 3, 1), stride=(1, 1, 1)),
                                    nn.BatchNorm3d(12,  eps=0.001, momentum=0.1, affine=True), # 动量默认值为0.1
                                    mish())

        self.conv32 =nn.Sequential(nn.Conv3d(in_channels=36, out_channels=12, padding=(0, 1, 0),kernel_size=(1, 3, 1), stride=(1, 1, 1)),
                                    nn.BatchNorm3d(12,  eps=0.001, momentum=0.1, affine=True), # 动量默认值为0.1
                                    mish())
        
        self.conv33 =nn.Sequential(nn.Conv3d(in_channels=48, out_channels=12, padding=(0, 1, 0),kernel_size=(1, 3, 1), stride=(1, 1, 1)),
                                    nn.BatchNorm3d(12,  eps=0.001, momentum=0.1, affine=True), # 动量默认值为0.1
                                    mish())

        self.conv34 =nn.Sequential(nn.Conv3d(in_channels=60, out_channels=60, padding=(0, 0, 0),kernel_size=(1, 1, kernel_3d), stride=(1, 1, 1)),
                                    nn.BatchNorm3d(60,  eps=0.001, momentum=0.1, affine=True), # 动量默认值为0.1
                                    # mish()
                                    )

        self.batch_norm_spectral = nn.Sequential(
                                    nn.BatchNorm3d(60,  eps=0.001, momentum=0.1, affine=True), # 动量默认值为0.1
                                    #gelu_new(),
                                    #swish(),
            mish(),
                                    nn.Dropout(p=0.5)
        )
        self.batch_norm_spatial_x = nn.Sequential(
                                    nn.BatchNorm3d(60,  eps=0.001, momentum=0.1, affine=True), # 动量默认值为0.1
                                    #gelu_new(),
                                    #swish(),
            mish(),
                                    nn.Dropout(p=0.5)
        )
        self.batch_norm_spatial_y = nn.Sequential(
                                    nn.BatchNorm3d(60,  eps=0.001, momentum=0.1, affine=True), # 动量默认值为0.1
                                    #gelu_new(),
                                    #swish(),
            mish(),
                                    nn.Dropout(p=0.5)
        )


        self.global_pooling = nn.AdaptiveAvgPool3d(1)
        self.full_connection = nn.Sequential(
                                nn.Dropout(p=0.5),
                                nn.Linear(180, classes),
                                # nn.Softmax() ,
        )

        self.attention_spectral = CAM_Module(60)
        self.attention_spatial_x = PAM_X_Module(60)
        self.attention_spatial_y = PAM_Y_Module(60)
    
    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, X):
        X = self.input(X)
        X = self.residual_layer(X)

        X = self.conv_feature(X) # n*24*9*9*97

        # X = self.residual_layer(X)

        # spectral
        x11 = self.conv11(X)
        x12 = torch.cat((X, x11), dim=1)
        x12 = self.conv12(x12)
        x13 = torch.cat((X, x11, x12), dim=1)
        x13 = self.conv13(x13)
        x14 = torch.cat((X, x11, x12, x13), dim=1)
        x14 = self.conv14(x14)

        # 光谱注意力通道
        x1 = self.attention_spectral(x14)
        x1 = torch.mul(x1, x14)

        x1 = self.residual_spec(x1)
        # print(x1.shape)

        # spatial x
        x21 = self.conv21(X)
        x22 = torch.cat((X, x21), dim=1)
        x22 = self.conv22(x22)
        x23 = torch.cat((X, x21, x22), dim=1)
        x23 = self.conv23(x23)
        x24 = torch.cat((X, x21, x22, x23), dim=1)
        x24 = self.conv24(x24)
        # print(x24.shape)

        # 空间x注意力机制 
        x2 = self.attention_spatial_x(x24)
        x2 = torch.mul(x2, x24)

        x2 = self.residual_spa_x(x2)
        # print(x2.shape)

        # spatial y
        x31 = self.conv31(X)
        x32 = torch.cat((X, x31), dim=1)
        x32 = self.conv32(x32)
        x33 = torch.cat((X, x31, x32), dim=1)
        x33 = self.conv33(x33)
        x34 = torch.cat((X, x31, x32, x33), dim=1)
        x34 = self.conv34(x34)

        # 空间y注意力机制 
        x3 = self.attention_spatial_y(x34)
        x3 = torch.mul(x3, x34)

        x3 = self.residual_spa_x(x3)
        # print(x3.shape)

        # model1
        x1 = self.batch_norm_spectral(x1)
        x1 = self.global_pooling(x1)
        x1 = x1.squeeze(-1).squeeze(-1).squeeze(-1)
        
        x2=self.batch_norm_spatial_x(x2)
        x2= self.global_pooling(x2)
        x2 = x2.squeeze(-1).squeeze(-1).squeeze(-1)

        x3=self.batch_norm_spatial_x(x3)
        x3= self.global_pooling(x3)
        x3 = x3.squeeze(-1).squeeze(-1).squeeze(-1)

        x_pre = torch.cat((x1, x2, x3), dim=1)

        output = self.full_connection(x_pre)

        return output












if __name__=='__main__':
    net=D3DTBTA_network(200,16).cuda()
    x=torch.randn((8,1,9,9,200)).cuda()
    y=net(x)
    print(y.shape)