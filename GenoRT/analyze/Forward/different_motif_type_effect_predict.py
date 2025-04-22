import os
import sys
import time
import pandas as pd
import numpy as np
import pyBigWig
import tabix
import torch
from torch import nn
from selene_sdk.targets import Target
import selene_sdk
from v3_lstm import *
# 用于统计不同尺度motif Inr的总效应
tissue = 'merge'
dir_path = 'new_model/model_for_predict'
nets_ = os.listdir(dir_path)
nets_ = [os.path.join(dir_path,i) for i in nets_ if i.startswith(f'predict',0)]
motifs = np.load(rf"analyze/result/sorted_cl_motif.npy")

other_motifs = np.load(rf"/Data5/pfGao/xtwang/TSS/tss/analyze/result/pi_w82_other.npy")
n_motif = np.load("analyze/result/sorted_cl_motif.npy").shape[0]
n_others = np.load("analyze/result/pi_w82_other.npy").shape[0]
print(n_motif,n_others)
motif_idx = [ i for i in range(0,18)]
long_inr_idx = [ i for i in range(18,24)]
short_inr_idx = [ i for i in range(24,27)]


w82_tsses = pd.read_table(
   rf"/Data5/pfGao/xtwang/TSS/tss/data_process/Unidirection_result.txt",
    sep="\t",
)
w82_genome = selene_sdk.sequences.Genome(
    input_path=r"/Data5/pfGao/xtwang/TSS/W82-NJAU/genome/Wm82-NJAU.fasta",
    )


class GenomicSignalFeatures(Target):
    def __init__(
        self,
        input_paths,
        features,
        shape,
        blacklists=None,
        blacklists_indices=None,
        replacement_indices=None,
        replacement_scaling_factors=None,
    ):
        self.input_paths = input_paths
        self.initialized = False
        self.blacklists = blacklists
        self.blacklists_indices = blacklists_indices
        self.replacement_indices = replacement_indices
        self.replacement_scaling_factors = replacement_scaling_factors

        self.n_features = len(features)
        self.feature_index_dict = dict(
            [(feat, index) for index, feat in enumerate(features)]
        )
        self.shape = (len(input_paths), *shape)

    def get_feature_data(
        self, chrom, start, end, nan_as_zero=True, feature_indices=None
    ):
        if not self.initialized:
            self.data = [pyBigWig.open(path) for path in self.input_paths]
            if self.blacklists is not None:
                self.blacklists = [
                    tabix.open(blacklist) for blacklist in self.blacklists
                ]
            self.initialized = True
        if feature_indices is None:
            feature_indices = np.arange(len(self.data))
        wigmat = np.zeros((len(feature_indices), end - start), dtype=np.float32)
        for i in feature_indices:
            try:
                wigmat[i, :] = self.data[i].values(chrom, start, end, numpy=True)
            except:
                print(chrom, start, end, self.input_paths[i], flush=True)
                raise

        if self.blacklists is not None:
            if self.replacement_indices is None:
                if self.blacklists_indices is not None:
                    for blacklist, blacklist_indices in zip(
                        self.blacklists, self.blacklists_indices
                    ):
                        for _, s, e in blacklist.query(chrom, start, end):
                            wigmat[
                                blacklist_indices,
                                np.fmax(int(s) - start, 0) : int(e) - start,
                            ] = 0
                else:
                    for blacklist in self.blacklists:
                        for _, s, e in blacklist.query(chrom, start, end):
                            wigmat[:, np.fmax(int(s) - start, 0) : int(e) - start] = 0
            else:
                for (
                    blacklist,
                    blacklist_indices,
                    replacement_indices,
                    replacement_scaling_factor,
                ) in zip(
                    self.blacklists,
                    self.blacklists_indices,
                    self.replacement_indices,
                    self.replacement_scaling_factors,
                ):
                    for _, s, e in blacklist.query(chrom, start, end):
                        wigmat[
                            blacklist_indices,
                            np.fmax(int(s) - start, 0) : int(e) - start,
                        ] = (
                            wigmat[
                                replacement_indices,
                                np.fmax(int(s) - start, 0) : int(e) - start,
                            ]
                            * replacement_scaling_factor
                        )
        if nan_as_zero:
            wigmat[np.isnan(wigmat)] = 0
        wigmat = np.log(wigmat+1)
        return wigmat

class Encoder(nn.Module):
    def __init__(self,n_motifs,n_others):
        super(Encoder, self).__init__()
        feature_dim = (n_motifs+n_others) * 2
        self.conv_motif = nn.Conv1d(4, n_motifs, kernel_size=51, padding=25)
        self.conv_other = nn.Conv1d(4, n_others, kernel_size=3, padding=1)


        self.conv_motif.weight.data = torch.FloatTensor(motifs)
        self.conv_other.weight.data = torch.FloatTensor(other_motifs)
        self.sigmoid = nn.Sigmoid()


        self.conv_motif.weight.requires_grad = False
        self.conv_other.weight.requires_grad = False

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=feature_dim,out_channels=feature_dim,kernel_size=1,bias=False),
            nn.BatchNorm1d(feature_dim),
            nn.SiLU()            
        )

        self.avgpool = nn.Sequential(
            nn.AdaptiveAvgPool1d(4650),
            nn.BatchNorm1d(feature_dim),
            nn.SiLU()
        )
        self.maxpool = nn.Sequential(
            nn.AdaptiveMaxPool1d(4650),
            nn.BatchNorm1d(feature_dim),
            nn.SiLU()
        )
        self.atten = eca_block(in_channel=2*n_motifs,nhead=1)

    def forward(self, x,KO_effect=None):
        y_motif = torch.cat([self.conv_motif(x),self.conv_motif(x.flip([1, 2])).flip([2])], 1)

        y_other = torch.cat(
            [self.conv_other(x),self.conv_other(x.flip([1, 2])).flip([2])], 1)
        
        motif_act = self.sigmoid(y_motif)
        motif_act = self.atten(motif_act)

        y_other_act = self.sigmoid(y_other)
        
        # #训练时随机丢弃10%的motif激活层
        # if self.training:
        #     # num_channels = self.conv_motif.weight.size(0)
        #     num_channels = 18
        #     drop_idx = torch.randperm(num_channels)[:int(num_channels * 0.1)]
        #     mask = torch.ones(self.conv_motif.out_channels*2, device=x.device)
        #     mask[drop_idx] = 0
        #     mask[drop_idx+num_channels] = 0
        #     motif_act= motif_act * mask.view(1, -1, 1)
            
        if KO_effect != None:
            if KO_effect == 'motif':
                motif_act[:,torch.tensor(motif_idx),:] = 0
                motif_act[:,torch.tensor(motif_idx) + y_motif.shape[1]//2,:] = 0
            elif KO_effect == 'short_inr':
                motif_act[:,torch.tensor(short_inr_idx),:] = 0
                motif_act[:,torch.tensor(short_inr_idx) + y_motif.shape[1]//2,:] = 0
            elif KO_effect == 'long_inr':
                motif_act[:,torch.tensor(long_inr_idx),:] = 0
                motif_act[:,torch.tensor(long_inr_idx) + y_motif.shape[1]//2,:] = 0
            else:
                y_other_act *= 0 
        feature = self.conv(torch.cat([motif_act,y_other_act],dim=1))
        oup = self.avgpool(feature) + self.maxpool(feature)
        return oup + feature
    
    
class Decoder(nn.Module):
    def __init__(self,inp_dim):
        super(Decoder,self).__init__()
        
        self.c = nn.Sequential(
                nn.Conv1d(in_channels=inp_dim,out_channels=inp_dim + 32,kernel_size=25,padding=12,bias=False),
                nn.BatchNorm1d(inp_dim + 32),
                nn.SiLU(),
                ConvBlock(inp_dim + 32,inp_dim + 32,seq_len=100000,kernel_size=25,dilation_ratio=1,MHA=False),
                nn.BatchNorm1d(inp_dim + 32),

        )
        
        self.d1 =nn.Sequential(
            down_layer(inp_dim + 32,(inp_dim + 32)*2,stride=5,kernel_size=25),
            ConvBlock((inp_dim + 32)*2,(inp_dim + 32)*2,seq_len=20000,kernel_size=25,dilation_ratio=1),
        )
        self.d2 = nn.Sequential(
            down_layer((inp_dim + 32)*2,(inp_dim + 32)*2,stride=5,kernel_size=25),
            ConvBlock((inp_dim + 32)*2,(inp_dim + 32)*2,seq_len=4000,kernel_size=25,dilation_ratio=1),

        )

        self.d3 = nn.Sequential(
            down_layer((inp_dim + 32)*2,(inp_dim + 32)*2,stride=3,kernel_size=25),
            ConvBlock((inp_dim + 32)*2,(inp_dim + 32)*2,seq_len=800,kernel_size=25,dilation_ratio=1,MHA=False),
        )
        self.d4 = nn.Sequential(
            down_layer((inp_dim + 32)*2,(inp_dim + 32)*2,stride=2,kernel_size=25),
            ConvBlock((inp_dim + 32)*2,(inp_dim + 32)*2,seq_len=800,kernel_size=25,dilation_ratio=1,MHA=False),

        )
        self.u4 =  up_layer((inp_dim + 32)*2,(inp_dim + 32)*2,kernel_size=25,scale_factor=2)
        self.c4 = nn.Sequential(
            ConvBlock((inp_dim + 32)*2,(inp_dim + 32)*2,seq_len=5000,kernel_size=25,dilation_ratio=1),

        )
        self.u3 =  up_layer((inp_dim + 32)*2,(inp_dim + 32)*2,kernel_size=25,scale_factor=3)
        self.c3 = nn.Sequential(
            ConvBlock((inp_dim + 32)*2,(inp_dim + 32)*2,seq_len=5000,kernel_size=25,dilation_ratio=1),

        )
        self.u2 = up_layer((inp_dim + 32)*2,(inp_dim + 32)*2,kernel_size=25,scale_factor=5)
        self.c2 = nn.Sequential(
            ConvBlock((inp_dim + 32)*2,(inp_dim + 32)*2,seq_len=25000,kernel_size=25,dilation_ratio=1),
        )
        
        self.u1 = up_layer((inp_dim + 32)*2,inp_dim + 32,kernel_size=25,scale_factor=5)
        self.c1 = nn.Sequential(
            ConvBlock(inp_dim + 32,inp_dim + 32,seq_len=100000,kernel_size=25,dilation_ratio=1),

        )
        self.reg_final = nn.Sequential(
            nn.Conv1d(inp_dim + 32,inp_dim + 32,kernel_size=25,stride=1,padding=12,bias=False),
            nn.BatchNorm1d(inp_dim + 32),
            nn.SiLU(),
            nn.Conv1d(inp_dim + 32,inp_dim,kernel_size=25,stride=1,padding=12,bias=False),
            nn.BatchNorm1d(inp_dim),
            nn.SiLU(),
        )
    def forward(self,x,mask=0):
        o1 = self.c(x)
        o2 = self.d1(o1)
        o3 = self.d2(o2) 
        o4 = self.d3(o3) 
        o5 = self.d4(o4) 
        u4 = self.c4(self.u4(o5,o4)) 
        u3 = self.c3(self.u3(u4,o3)) 
        u2 = self.c2(self.u2(u3,o2)) 
        u1 = self.c1(self.u1(u2,o1))
        out = self.reg_final(u1)
        return out

class Geno_RT(nn.Module):
    def __init__(self,n_motifs,n_others):
        super(Geno_RT,self).__init__()
        inp_dim = (n_motifs + n_others)*2
        self.encoder = Encoder(n_motifs,n_others)
        self.decoder1 = Decoder(inp_dim=inp_dim)
        self.decoder2 = Decoder(inp_dim=inp_dim)
        self.act =  nn.Sequential(
            nn.Conv1d(inp_dim,2,kernel_size=1,stride=1,padding=0,bias=False),
            nn.Softplus(),
        )
    def forward(self,x,KO_effect=None,mask=0):
        encode_feature = self.encoder(x,KO_effect=KO_effect)
        decode_feature = self.decoder1(encode_feature,mask=mask)
        decode_feature = self.decoder2(decode_feature,mask=mask)
        out = self.act(decode_feature)
        return out

tfeature = GenomicSignalFeatures(
  [
    rf"data_process/filter_bw/merge.plus.bw",
    rf"data_process/filter_bw/merge.minus.bw",
],
    [
        f"{tissue}_plus",
        f"{tissue}_minus",
    ],
    (4000,),
)
window_size = 4650
seqs = []
tars = []
print('select thick.start')
w82_tsses = w82_tsses[w82_tsses['support'] >1]
promoter = w82_tsses[w82_tsses['txType'] =='promoter']
promoter = promoter[promoter['peakTxType'] =='promoter']

UTR = w82_tsses[w82_tsses['txType'] =='fiveUTR']
UTR = UTR[UTR['peakTxType'] == 'fiveUTR']

TSS_hc = pd.concat([promoter,UTR],axis=0)
TSS_hc = TSS_hc[ ~TSS_hc['seqnames'].str.contains('scaffold') ]
TSS_hc = TSS_hc.sort_values(by='score',ascending=False)
tsses = TSS_hc.iloc[:,:]
n_tsses = len(tsses)
print(n_tsses)

all_motif_cor = []
all_diff_cor = []
all_other_cor = []
all_long_inr_cor = []
all_short_inr_cor = []

all_sum_eff = []
all_motif_eff = []
all_other_eff = []
all_long_inr_eff = []
all_short_inr_eff = []
print(len(nets_))
pos_cor_list = []
pos_mse_list = []
print(nets_)
for model_path in nets_:
    net = torch.load(model_path,map_location='cuda:0').to('cuda:0')
    net.eval()
    cor = []
    chr = []
    pos_list = []
    strand_list = []
    score_list = []
    singal_strand = []
    mse = []
    KO_dict = {
        'motif':[],
        'long_inr':[],
        'short_inr':[],
        'other':[],
        'sum':[]
    }
    for randi in range(n_tsses):
        # if tsses["seqnames"].values[randi] not in [ "GWHCAYC00000010",'GWHCAYC00000007']:
        seqnamesm, pos, strand = (
            w82_tsses["seqnames"].values[randi],
            w82_tsses["thick.start"].values[randi],
            w82_tsses["strand"].values[randi],
        )
        
        try:
            if strand == '-':
                seq = w82_genome.get_encoding_from_coords(
                    w82_tsses["seqnames"].values[randi],
                    w82_tsses["thick.start"].values[randi] - 825,
                    w82_tsses["thick.start"].values[randi] + 3825,
                    w82_tsses["strand"].values[randi],
                )
                
                tar = tfeature.get_feature_data(
                    w82_tsses["seqnames"].values[randi],
                    w82_tsses["thick.start"].values[randi] - 825 ,
                    w82_tsses["thick.start"].values[randi] + 3825,
                )
                tar = tar[::-1, ::-1]
                
            else:
                seq = w82_genome.get_encoding_from_coords(
                    w82_tsses["seqnames"].values[randi],
                    w82_tsses["thick.start"].values[randi] - 3825,
                    w82_tsses["thick.start"].values[randi] + 825,
                    w82_tsses["strand"].values[randi],
                )
                
                tar = tfeature.get_feature_data(
                    w82_tsses["seqnames"].values[randi],
                    w82_tsses["thick.start"].values[randi] - 3825,
                    w82_tsses["thick.start"].values[randi] + 825,
                )

        except:
            continue
        window_size = 5
        tar_smoothed = []
        # 创建平滑后的数组
        for i in range(tar.shape[0]):
            smoothed_data = np.convolve(tar[i], np.ones(window_size)/window_size, mode='same')
            tar_smoothed.append(smoothed_data)
        tar = np.array(tar_smoothed)
        # print(torch.tensor(seq).shape)
        with torch.no_grad():
            pred = net(torch.tensor(seq).permute(1,0).cuda()[None,:,:],KO_effect = None).detach().cpu()
            cor = np.corrcoef(pred[0,0,325:-325],tar[0,325:-325])[0][1]
            print(cor)
            for ko_idx in ['motif','long_inr','short_inr','other']:
                KO_pred = net(torch.tensor(seq).permute(1,0).cuda()[None,:,:],KO_effect = ko_idx).detach().cpu()
                ko_effect = (pred - KO_pred)[0,0]

                KO_dict[ko_idx].append(ko_effect)
        singal_strand.append(tar[0])
        KO_dict['sum'].append(pred.detach().cpu()[0,0])   

    pos_cor_list.append(cor)
    pos_mse_list.append(mse)

import matplotlib.pyplot as plt
import seaborn as sns 
palette = sns.color_palette("husl", 5)
motif_eff = np.array(KO_dict['motif']).mean(axis=0)[-1325:-325]
long_inr_eff = np.array(KO_dict['long_inr']).mean(axis=0)[-1325:-325]
short_inr_eff = np.array(KO_dict['short_inr']).mean(axis=0)[-1325:-325]
other_eff = np.array(KO_dict['other']).mean(axis=0)[-1325:-325]
sum_eff = np.array(KO_dict['sum']).mean(axis=0)[-1325:-325]
all_sum_eff.append(sum_eff)
all_motif_eff.append(motif_eff)
all_other_eff.append(other_eff)
all_long_inr_eff.append(long_inr_eff)
all_short_inr_eff.append(short_inr_eff)

import matplotlib.cm  as cm

x = list(range(4650))
avg_sum_eff = np.array(all_sum_eff).mean(0)

avg_motif_eff = np.array(all_motif_eff).mean(0)
avg_other_eff = np.array(all_other_eff).mean(0) 
avg_long_inr_eff = np.array(all_long_inr_eff).mean(0) 
avg_short_inr_eff = np.array(all_short_inr_eff).mean(0) 

plt.plot(np.arange(-500,500),avg_sum_eff,c=palette[0],label='Sum',linewidth=0.3)
plt.plot(np.arange(-500,500),avg_motif_eff,c=palette[1],label='Motif',linewidth=0.3)
plt.plot(np.arange(-500,500),avg_long_inr_eff,c=palette[2],label='Long Inr',linewidth=0.3)
plt.plot(np.arange(-500,500),avg_short_inr_eff,c=palette[3],label='Short Inr',linewidth=0.3)
plt.plot(np.arange(-500,500),avg_other_eff,c=palette[4],label='Other',linewidth=0.3)

plt.xlabel('Relative Position')
plt.ylabel('Effect of Sequential Pattern(log)')
plt.title('Different Sequential Pattern Effect In Model')
plt.legend()
plt.savefig('/Data5/pfGao/xtwang/TSS/tss/analyze/new_model_result/Different Sequential Pattern Effect In Model.svg',dpi=800,format='svg')
plt.close()

df = np.vstack([avg_sum_eff,avg_motif_eff,avg_long_inr_eff,avg_short_inr_eff,avg_other_eff])
print(df.shape)
df = pd.DataFrame(df.T,columns=['sum','motif','long_inr','short_inr','others'])
df.to_csv("/Data5/pfGao/xtwang/TSS/tss/analyze/new_model_result/diff_type_motif_effect.csv",index=None)
# plt.plot(np.arange(-500,500),avg_sum_eff,c=palette[0],label='Sum',linewidth=0.3)
print(avg_long_inr_eff)
plt.plot(np.arange(-500,500),avg_motif_eff / 18 ,c=palette[1],label='Motif',linewidth=0.3) 
plt.plot(np.arange(-500,500),avg_long_inr_eff / 6, c=palette[2],label='Long Inr',linewidth=0.3) 
plt.plot(np.arange(-500,500),avg_short_inr_eff / 3,c=palette[3],label='Short Inr',linewidth=0.3)
plt.plot(np.arange(-500,500),avg_other_eff / 12,c=palette[4],label='Other',linewidth=0.3)
plt.xlabel('Relative Position')
plt.ylabel('Effect of Sequential Pattern(log)')
plt.title('Different Sequential Pattern Effect In Model')
plt.legend()
plt.savefig('/Data5/pfGao/xtwang/TSS/tss/analyze/new_model_result/Different Sequential Pattern Effect In Model_avg.svg',dpi=800,format='svg')
plt.close()

        
