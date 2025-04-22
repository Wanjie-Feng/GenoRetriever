import torch
import torch.nn as nn
from MECA_block import eca_block

class Linear(nn.Module):
    def __init__(self,inp,oup,bias=False) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features=inp,out_features=oup,bias=bias)
    def forward(self,x):
        out = self.linear(x.permute(0,2,1)).permute(0,2,1)
        return out
    
class ConvBlock(nn.Module):
    def __init__(self, inp, oup,seq_len,dilation_ratio=2,expand_ratio=2,kernel_size = 9,dropout=0.3,MHA=False):
        super(ConvBlock, self).__init__()
        hidden_dim = round(inp * expand_ratio)
        self.MHA = MHA
        self.inp = inp
        self.oup = oup
        self.fn_bn = nn.BatchNorm1d(oup)
        self.atten = nn.MultiheadAttention(embed_dim=inp,num_heads=2,dropout=dropout) if self.MHA else eca_block(in_channel=inp,nhead=1)
        self.conv_up= nn.Sequential(
            Linear(inp=inp,oup=hidden_dim,bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
        )
        self.conv_hidden = nn.Sequential(
            nn.Conv1d(hidden_dim,hidden_dim,kernel_size=kernel_size,stride=1,padding=(kernel_size-1)//2,bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Dropout1d(p=dropout)
        )
        self.conv_down = nn.Sequential(
            Linear(inp=hidden_dim,oup=oup,bias=False),
            nn.BatchNorm1d(oup),
            nn.SiLU(),
        )
        self.act = nn.SiLU()
    def forward(self, x):
        if self.MHA:
            inp = x.permute(0,2,1)
            atten = self.atten(inp,inp,inp)[0].permute(0,2,1) + x
        else:
            atten = self.atten(x) + x
        up = self.conv_up(atten)
        hidden = self.conv_hidden(up)
        down = self.conv_down(hidden) 
        out = self.fn_bn(down+atten)
        return out


class up_layer(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,scale_factor) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=scale_factor,mode='linear')
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels,out_channels,kernel_size,stride=1,padding=(kernel_size-1)//2),
            nn.BatchNorm1d(out_channels),
            )
        self.act = nn.SiLU()
    def forward(self,x,feature):
        out = self.act(self.conv(self.up(x)) + feature)
        return out

class down_layer(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride):
        super().__init__()
        self.down = nn.Conv1d(in_channels,out_channels,kernel_size,stride,padding = (kernel_size - 1) // 2,bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.SiLU()
    def forward(self,x):
        out = self.act(self.bn(self.down(x)))
        return out




