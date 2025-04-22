
import torch
from torch import nn
import math

class BN(nn.Module):
    def __init__(self,feature) -> None:
        super().__init__()
        self.bn = nn.BatchNorm1d(feature)
    def forward(self,x):
        return self.bn(x.permute(0,2,1)).permute(0,2,1)
    
# 定义ECANet的类
class eca_block(nn.Module):
    # 初始化, in_channel代表特征图的输入通道数, b和gama代表公式中的两个系数
    def __init__(self, in_channel,nhead=4):
        # 继承父类初始化
        super().__init__()
        
        # 根据输入通道数自适应调整卷积核大小
        self.channel = in_channel
        self.nhead = nhead

        # 全局平均池化，输出的特征图的宽高=1
        self.avg_pool = nn.AdaptiveAvgPool1d(output_size=1)
        self.max_pool = nn.AdaptiveMaxPool1d(output_size=1)
        self.linear1 = nn.Sequential(
            nn.Linear(in_features=in_channel//nhead,out_features=in_channel//nhead,bias=False),
            BN(feature=in_channel//nhead)
        )
        self.linear2 = nn.Sequential(
            nn.Linear(in_features=in_channel//nhead,out_features=in_channel//nhead,bias=False),
            BN(feature=in_channel//nhead)
        )
        # softmax激活函数，权值归一化
        # self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()
    # 前向传播
    def forward(self, inputs):
        # 获得输入图像的shape
        b, c, s = inputs.shape
        
        avg_x = self.avg_pool(inputs).reshape((inputs.size(0),self.nhead,self.channel//self.nhead))
        avg_x = self.linear1(avg_x)
        max_x = self.max_pool(inputs).reshape((inputs.size(0),self.nhead,self.channel//self.nhead))
        max_x = self.linear2(max_x)
        score = self.sigmoid(max_x+avg_x)
        score = score.reshape((b,c,-1)).view([b,c,1])
        # 维度调整，变成序列形式 [b,c,1,1]==>[b,1,c]
        # 维度调整 [b,1,c]==>[b,c,1,1]
        
        # 将输入特征图和通道权重相乘[b,c,h,w]*[b,c,1,1]==>[b,c,h,w]
        outputs = score * inputs 
        return outputs
    