import torch
import torch.nn as nn


class BasicConv3x3(nn.Module):
    def __init__(self,Cin,Cout):
        super().__init__()
        self.BasicConv3x3 = nn.Sequential(
                            nn.Conv2d(Cin,Cout,kernel_size=3,padding=1),
                            nn.BatchNorm2d(Cout),
                            nn.ReLU(inplace=True)
                            )
    def forward(self,x):
        return self.BasicConv3x3(x)
class DilatedConvolution(nn.Module):
    def __init__(self,Cin,Cout,dilation):
        super().__init__()
        self.DilatedConvl = nn.Sequential(
                            nn.Conv2d(Cin,Cout,kernel_size=3,padding=dilation,dilation=dilation),
                            nn.BatchNorm2d(Cout),
                            nn.ReLU(inplace=True)
                            )
    def forward(self,x):
        return self.DilatedConvl(x)
class ConvnSigm():
    def __init__(self):
        super().__init__()
        self.convnsigm = nn.Sequential(
                        nn.Conv2d(64,1,kernel_size=3,padding=1),
                        nn.Sigmoid()
                        )
    def forward(self,x):
        return self.convnsigm(x)

class Rsu7(nn.Module):
    def __init__(self,en):
        super().__init__()
        self.downsample = nn.MaxPool2d(kernel_size=2,stride=2)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        if en == True:
            self.conv_1 = BasicConv3x3(Cin=3,Cout=64)
        else:
            self.conv_1 = BasicConv3x3(Cin=128,Cout=64)
        self.conv_2 = BasicConv3x3(Cin=64,Cout=16)
        self.conv_3_1 = BasicConv3x3(Cin=16,Cout=16)
        self.conv_3_2 = BasicConv3x3(Cin=16,Cout=16)
        self.conv_3_3 = BasicConv3x3(Cin=16,Cout=16)
        self.conv_3_4 = BasicConv3x3(Cin=16,Cout=16)
        self.conv_3_5 = BasicConv3x3(Cin=16,Cout=16)
        self.conv_4_1 = BasicConv3x3(Cin=32,Cout=16)
        self.conv_4_2 = BasicConv3x3(Cin=32,Cout=16)
        self.conv_4_3 = BasicConv3x3(Cin=32,Cout=16)
        self.conv_4_4 = BasicConv3x3(Cin=32,Cout=16)
        self.conv_4_5 = BasicConv3x3(Cin=32,Cout=16)
        self.conv_5 = BasicConv3x3(Cin=32,Cout=64)
        self.deconv = DilatedConvolution(Cin=16,Cout=16,dilation=2)
    def forward(self,x):
        out1 = self.conv_1(x)
        out2 = self.conv_2(out1)
        out2_ds = self.downsample(out2)
        out3 = self.conv_3_1(out2_ds)
        out3_ds = self.downsample(out3)
        out4 = self.conv_3_2(out3_ds)
        out4_ds = self.downsample(out4)
        out5 = self.conv_3_3(out4_ds)
        out5_ds = self.downsample(out5)
        out6 = self.conv_3_4(out5_ds)
        out6_ds = self.downsample(out6)
        out7 = self.conv_3_5(out6_ds)
        # print(out7.shape,self.deconv(out7).shape)
        out8 = torch.cat((out7,self.deconv(out7)),dim=1)
        out9 = self.conv_4_1(out8)
        out9_us = self.upsample(out9)
        # print(out6.shape,out6_ds.shape,out7.shape,out9_us.shape)
        out10 = torch.cat((out6,out9_us),dim=1)
        out11 = self.conv_4_2(out10)
        out11_us = self.upsample(out11)
        out12 = torch.cat((out5,out11_us),dim=1)
        out13 = self.conv_4_3(out12)
        out13_us = self.upsample(out13)
        # print(out4.shape,out13_us.shape)
        out14 = torch.cat((out4,out13_us),dim=1)
        out15 = self.conv_4_4(out14)
        out15_us = self.upsample(out15)
        out16 = torch.cat((out3,out15_us),dim=1)
        out17 = self.conv_4_5(out16)
        out17_us = self.upsample(out17)
        out18 = torch.cat((out2,out17_us),dim=1)
        out19 = self.conv_5(out18)
        out20 = torch.add(out1,out19)
        return out20

class Rsu6(nn.Module):
    def __init__(self,en):
        super().__init__()
        self.downsample = nn.MaxPool2d(kernel_size=2,stride=2)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        if en ==True:
            self.conv_1 = BasicConv3x3(Cin=64,Cout=64)
        else:
            self.conv_1 = BasicConv3x3(Cin=128,Cout=64)
        self.conv_2 = BasicConv3x3(Cin=64,Cout=16)
        self.conv_3_1 = BasicConv3x3(Cin=16,Cout=16)
        self.conv_3_2 = BasicConv3x3(Cin=16,Cout=16)
        self.conv_3_3 = BasicConv3x3(Cin=16,Cout=16)
        self.conv_3_4 = BasicConv3x3(Cin=16,Cout=16)
        self.conv_4_1 = BasicConv3x3(Cin=32,Cout=16)
        self.conv_4_2 = BasicConv3x3(Cin=32,Cout=16)
        self.conv_4_3 = BasicConv3x3(Cin=32,Cout=16)
        self.conv_4_4 = BasicConv3x3(Cin=32,Cout=16)
        self.conv_5 = BasicConv3x3(Cin=32,Cout=64)
        self.deconv = DilatedConvolution(Cin=16,Cout=16,dilation=2)
    def forward(self,x):
        out1 = self.conv_1(x)
        out2 = self.conv_2(out1)
        out2_ds = self.downsample(out2)
        out3 = self.conv_3_1(out2_ds)
        out3_ds = self.downsample(out3)
        out4 = self.conv_3_2(out3_ds)
        out4_ds = self.downsample(out4)
        out5 = self.conv_3_3(out4_ds)
        out5_ds = self.downsample(out5)
        out6 = self.conv_3_4(out5_ds)
        # out6_ds = self.downsample(out6)
        # out7 = self.conv_3_5(out6_ds)
        out8 = torch.cat((out6,self.deconv(out6)),dim=1)
        out9 = self.conv_4_1(out8)
        out9_us = self.upsample(out9)
        out10 = torch.cat((out5,out9_us),dim=1)
        out11 = self.conv_4_2(out10)
        out11_us = self.upsample(out11)
        out12 = torch.cat((out4,out11_us),dim=1)
        out13 = self.conv_4_3(out12)
        out13_us = self.upsample(out13)
        out14 = torch.cat((out3,out13_us),dim=1)
        out15 = self.conv_4_4(out14)
        out15_us = self.upsample(out15)
        # out16 = torch.cat((out3,out15_us),dim=1)
        # out17 = self.conv_4_5(out16)
        # out17_us = self.upsample(out17)
        out18 = torch.cat((out2,out15_us),dim=1)
        out19 = self.conv_5(out18)
        out20 = torch.add(out1,out19)
        return out20

class Rsu5(nn.Module):
    def __init__(self,en):
        super().__init__()
        self.downsample = nn.MaxPool2d(kernel_size=2,stride=2)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        if en == True:
            self.conv_1 = BasicConv3x3(Cin=64,Cout=64)
        else:
            self.conv_1 = BasicConv3x3(Cin=128,Cout=64)
        self.conv_2 = BasicConv3x3(Cin=64,Cout=16)
        self.conv_3_1 = BasicConv3x3(Cin=16,Cout=16)
        self.conv_3_2 = BasicConv3x3(Cin=16,Cout=16)
        self.conv_3_3 = BasicConv3x3(Cin=16,Cout=16)
        self.conv_4_1 = BasicConv3x3(Cin=32,Cout=16)
        self.conv_4_2 = BasicConv3x3(Cin=32,Cout=16)
        self.conv_4_3 = BasicConv3x3(Cin=32,Cout=16)
        self.conv_5 = BasicConv3x3(Cin=32,Cout=64)
        self.deconv = DilatedConvolution(Cin=16,Cout=16,dilation=2)
    def forward(self,x):
        out1 = self.conv_1(x)
        out2 = self.conv_2(out1)
        out2_ds = self.downsample(out2)
        out3 = self.conv_3_1(out2_ds)
        out3_ds = self.downsample(out3)
        out4 = self.conv_3_2(out3_ds)
        out4_ds = self.downsample(out4)
        out5 = self.conv_3_3(out4_ds)
        # out5_ds = self.downsample(out5)
        # out6 = self.conv_3_4(out5_ds)
        # out6_ds = self.downsample(out6)
        # out7 = self.conv_3_5(out6_ds)
        out8 = torch.cat((out5,self.deconv(out5)),dim=1)
        out9 = self.conv_4_1(out8)
        out9_us = self.upsample(out9)
        out10 = torch.cat((out4,out9_us),dim=1)
        out11 = self.conv_4_2(out10)
        out11_us = self.upsample(out11)
        out12 = torch.cat((out3,out11_us),dim=1)
        out13 = self.conv_4_3(out12)
        out13_us = self.upsample(out13)
        # out14 = torch.cat((out3,out13_us),dim=1)
        # out15 = self.conv_4_4(out14)
        # out15_us = self.upsample(out15)
        # out16 = torch.cat((out3,out15_us),dim=1)
        # out17 = self.conv_4_5(out16)
        # out17_us = self.upsample(out17)
        out18 = torch.cat((out2,out13_us),dim=1)
        out19 = self.conv_5(out18)
        out20 = torch.add(out1,out19)
        return out20
        
class Rsu4(nn.Module):
    def __init__(self,en):
        super().__init__()
        self.downsample = nn.MaxPool2d(kernel_size=2,stride=2)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        if en == True:
            self.conv_1 = BasicConv3x3(Cin=64,Cout=64)
        else:
            self.conv_1 = BasicConv3x3(Cin=128,Cout=64)
        self.conv_2 = BasicConv3x3(Cin=64,Cout=16)
        self.conv_3_1 = BasicConv3x3(Cin=16,Cout=16)
        self.conv_3_2 = BasicConv3x3(Cin=16,Cout=16)
        self.conv_4_1 = BasicConv3x3(Cin=32,Cout=16)
        self.conv_4_2 = BasicConv3x3(Cin=32,Cout=16)
        self.conv_5 = BasicConv3x3(Cin=32,Cout=64)
        self.deconv = DilatedConvolution(Cin=16,Cout=16,dilation=2)
    def forward(self,x):
        out1 = self.conv_1(x)
        out2 = self.conv_2(out1)
        out2_ds = self.downsample(out2)
        out3 = self.conv_3_1(out2_ds)
        out3_ds = self.downsample(out3)
        out4 = self.conv_3_2(out3_ds)
        # out4_ds = self.downsample(out4)
        # out5 = self.conv_3_3(out4_ds)
        # out5_ds = self.downsample(out5)
        # out6 = self.conv_3_4(out5_ds)
        # out6_ds = self.downsample(out6)
        # out7 = self.conv_3_5(out6_ds)
        out8 = torch.cat((out4,self.deconv(out4)),dim=1)
        out9 = self.conv_4_1(out8)
        out9_us = self.upsample(out9)
        out10 = torch.cat((out3,out9_us),dim=1)
        out11 = self.conv_4_2(out10)
        out11_us = self.upsample(out11)
        # out12 = torch.cat((out4,out11_us),dim=1)
        # out13 = self.conv_4_3(out12)
        # out13_us = self.upsample(out13)
        # out14 = torch.cat((out3,out13_us),dim=1)
        # out15 = self.conv_4_4(out14)
        # out15_us = self.upsample(out15)
        # out16 = torch.cat((out3,out15_us),dim=1)
        # out17 = self.conv_4_5(out16)
        # out17_us = self.upsample(out17)
        out18 = torch.cat((out2,out11_us),dim=1)
        out19 = self.conv_5(out18)
        out20 = torch.add(out1,out19)
        return out20

class Rsu4F(nn.Module):
    def __init__(self,en):
        super().__init__()
        # self.downsample = nn.MaxPool2d(kernel_size=2,stride=2)
        # self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        if en == True:
            self.conv_1 = BasicConv3x3(Cin=64,Cout=64)
        else:
            self.conv_1 = BasicConv3x3(Cin=128,Cout=64)
        self.conv_2 = BasicConv3x3(Cin=64,Cout=16)
        self.deconv1 = DilatedConvolution(Cin=16,Cout=16,dilation=2)
        self.deconv2 = DilatedConvolution(Cin=16,Cout=16,dilation=4)
        self.deconv3 = DilatedConvolution(Cin=16,Cout=16,dilation=8)
        self.deconv4 = DilatedConvolution(Cin=32,Cout=16,dilation=4)
        self.deconv5 = DilatedConvolution(Cin=32,Cout=16,dilation=2)
        self.conv_5 = BasicConv3x3(Cin=32,Cout=64)
        
    def forward(self,x):
        out1 = self.conv_1(x)
        out2 = self.conv_2(out1)
        out3 = self.deconv1(out2)
        out4 = self.deconv2(out3)
        out5 = self.deconv3(out4)
        out6 = self.deconv4(torch.cat((out4,out5),dim=1))
        out7 = self.deconv5(torch.cat((out4,out6),dim=1))
        out8 = self.conv_5(torch.cat((out2,out7),dim=1))
        return out8
