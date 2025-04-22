import glob
from GenoRT_s1 import GenoRT_S1
import torch
from matplotlib import pyplot as plt
import logomaker
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
prop_cycle = plt.rcParams['axes.prop_cycle']
itercolor =  prop_cycle()
import seaborn as sns
import os
sns.set_style("white")
color_scheme = {
    'A' :'#f23b27',
    'C' : '#e9963e',
    'G' : '#304f9e',
    'T' : '#65a9d7',
}
def plotfun(motifpwm, title=None, ax=None,ylabel=True):
    # print(motifpwm.shape,np.mean(motifpwm,axis=1).shape)
    offset = (motifpwm.shape[0]-1)/2 - np.round((np.arange(motifpwm.shape[0]) * np.square((motifpwm - motifpwm.mean(axis=1,keepdims=True))).sum(axis=1)).sum()/  np.square((motifpwm - motifpwm.mean(axis=1,keepdims=True))).sum(axis=1).sum())
    try:
        if  offset > 0:
            motifpwm[int(offset):, :] = motifpwm[:-int(offset),:]
            motifpwm[:int(offset), :] = 0
        else:
            motifpwm[:-int(np.abs(offset)), :] = motifpwm[int(np.abs(offset)):,:]
            motifpwm[-int(np.abs(offset)):, :] = 0
    except ValueError:
        pass
        
    motifpwm = motifpwm - np.mean(motifpwm,axis=1,keepdims=True)
    motifpwm = motifpwm - np.abs(motifpwm).mean(1,keepdims=True)*0.4
    motifpwm = np.where(np.abs(motifpwm)<motifpwm[motifpwm>0].mean() + motifpwm[motifpwm>0].std()*0.1,0,motifpwm)
    print(motifpwm[motifpwm>0].min())
    motifpwm = pd.DataFrame(motifpwm,columns=['A','C','G','T'])
    crp_logo = logomaker.Logo(motifpwm,
                              shade_below=.8,
                              fade_below=.8,
                            #   font_name='Arial Rounded MT Bold',
                              color_scheme=color_scheme,
                             ax=ax)

    # style using Logo methods
    crp_logo.style_spines(visible=False)
    crp_logo.style_spines(spines=['left', 'bottom'], visible=True)
    crp_logo.style_xticks(rotation=90, fmt='%d', anchor=0)
    if title is not None:
        crp_logo.ax.set_title(title,x=1.2, y=0.5)

    # style using Axes methods
    if ylabel:
        crp_logo.ax.set_ylabel("Motif score", labelpad=-1)
    crp_logo.ax.xaxis.set_ticks_position('none')
    crp_logo.ax.yaxis.set_ticks_position('left')

    crp_logo.ax.set_xticks([])
    crp_logo.ax.tick_params(axis='both', which='major', pad=-3)
    return crp_logo

import torch 
import torch.nn as nn


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)
    elif isinstance(m, nn.Conv1d):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)
            
class attention(nn.Module):
    def __init__(self,feature_num):
        super(attention, self).__init__()
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.linear = nn.Linear(feature_num,feature_num)
        self.act = nn.Softmax(dim=1)
    def forward(self,x):
        pool = self.pool(x).permute(0,2,1)
        oup = self.linear(pool).permute(0,2,1)
        atten_out = self.act(oup)
        return atten_out + x

class trans(nn.Module):
    def __init__(self):
        super(trans, self).__init__()
        # c = nn.Conv1d(in_channels=4,out_channels=64,kernel_size=4)
        pass
    def forward(self,x):
        return x.permute(0,2,1)
    
class GenoRT(nn.Module):
    def __init__(self):
        super(GenoRT, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(4, 40, kernel_size=51, padding=25),
            )
        self.conv_inr = nn.Sequential(nn.Conv1d(4, 10, kernel_size=15, padding=7))

        self.atten = attention(feature_num=40)
        
        self.activation = nn.Sigmoid()

        self.deconv = nn.Sequential(
            nn.Conv1d(80, 10, kernel_size=601, padding=300),
            nn.BatchNorm1d(10),
            )
        
        self.deconv_inr = nn.Sequential(nn.ConvTranspose1d(20, 10, kernel_size=15, padding=7))

        self.softplus = nn.Softplus()
        
        
    def forward(self, x):
        
        y = torch.cat([self.atten(self.conv(x)), 
                       self.atten(self.conv(x.flip([1, 2])).flip([2]))], dim=1)
        y_inr = torch.cat(
            [self.conv_inr(x), self.conv_inr(x.flip([1, 2])).flip([2])], 1
        )               
        y_act = self.activation(y) * y
        y_inr_act = self.activation(y_inr) * y_inr

        y_pred = self.softplus(self.deconv(y_act)+self.deconv_inr(y_inr_act))
    
        return y_pred
net = GenoRT()
tissue_list = ['Flower','Leaf','Nodule','Pod','Root','Seed','Shoot','Stemtip','merge']
stage = '1'
for tissue in tissue_list:
    save_path = rf"/Data5/pfGao/xtwang/TSS/tss/analyze/new_model_result/motif_img/PI46"
    os.makedirs(save_path, exist_ok=True)
    print(net)
    weight = np.load(rf"/Data5/pfGao/xtwang/TSS/tss/analyze/new_model_result/PI46_stage1/{tissue}/t_conv_weight.npy")
    de_weight = np.load(rf"/Data5/pfGao/xtwang/TSS/tss/analyze/new_model_result/PI46_stage1/{tissue}/t_deconv_weight.npy")
    print(weight.shape)
    print(de_weight.shape)
    fig,axes = plt.subplots(figsize=(10,weight.shape[0]), nrows=weight.shape[0],ncols=2, dpi=600)
    for idx in range(weight.shape[0]):
        inp = weight[idx].T
        print(inp.max(),inp[inp>0].std())
        axes[idx,0].set_xticks([])
        if idx != 0:
            plotfun(inp,ax=axes[idx,0],ylabel=False)
        else:
            plotfun(inp,ax=axes[idx,0])
        print(de_weight[idx,0].shape)
        axes[idx,1].plot(np.arange(-(de_weight[idx,0].shape[0]-1)/2,(de_weight[idx,0].shape[0]+1)/2), de_weight[idx,0],color='#65a9d7')
        axes[idx,1].plot(np.arange(-(de_weight[idx,0].shape[0]-1)/2,(de_weight[idx,0].shape[0]+1)/2), de_weight[idx+weight.shape[0],0,::-1],color='#e9963e',linestyle='--')
        axes[idx,1].get_xaxis().set_visible(False)
        sns.despine()
    fig.suptitle(f'Motif in {tissue}',fontsize=30)
    plt.savefig(os.path.join(save_path,f'{tissue}_motif.svg'),format='svg',dpi=800)
    plt.close()


#对筛选后的motif进行可视化
weight = np.load(r"analyze/result/sorted_cl_motif.npy")
de_weight = np.load(r"analyze/result/sorted_cl_deconv.npy")
print(weight.shape)
fig,axes = plt.subplots(figsize=(10,weight.shape[0]), nrows=weight.shape[0],ncols=2, dpi=600)
for idx in range(weight.shape[0]):
    inp = weight[idx].T
    print(inp.max(),inp[inp>0].std())

    axes[idx,0].set_xticks([])
    if idx != 0:
        plotfun(inp,ax=axes[idx,0],ylabel=False)
    else:
        plotfun(inp,ax=axes[idx,0])
    print(de_weight[idx,0].shape)
    axes[idx,1].plot(np.arange(-(de_weight[idx,0].shape[0]-1)/2,(de_weight[idx,0].shape[0]+1)/2), de_weight[idx,0],color='#65a9d7')
    axes[idx,1].plot(np.arange(-(de_weight[idx,0].shape[0]-1)/2,(de_weight[idx,0].shape[0]+1)/2), de_weight[idx+weight.shape[0],0,::-1],color='#e9963e',linestyle='--')
    axes[idx,1].get_xaxis().set_visible(False)

    sns.despine()
plt.savefig(os.path.join('analyze/new_model_result/motif_img/final_motif.svg'),format='svg',dpi=800)
plt.close()