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
import random
from reset import reset
import sys
from torch.utils.data import Dataset,DataLoader
import re
#统计所有位点的位置、信号值、不同motif对该位点的效应以及预测值与真实值之间的相关系数大小
tissue = 'Root'
device = 'cuda:2'
motifs = np.load(rf"analyze/result/sorted_cl_motif.npy")
other_motifs = np.load(rf"/Data5/pfGao/xtwang/TSS/tss/analyze/result/pi_w82_other.npy")
# tri_motifs = np.load(rf"/Data5/pfGao/xtwang/TSS/tss/analyze/result/tri.npy")
print(motifs.shape)
n_motifs = motifs.shape[0]
n_others = other_motifs.shape[0] 
# n_tris = tri_motifs.shape[0] 
n_tsses = 40000
sys.path.append("../utils/")

w82_tsses = pd.read_table(
   rf"KO_YY1/930-3/Unidirection_result.txt",
    sep="\t",
)
w82_genome = selene_sdk.sequences.Genome(
    input_path=r"/Data5/pfGao/xtwang/TSS/W82-NJAU/genome/Wm82-NJAU.fasta",
    )

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

    def forward(self, x, KO_idx = None):

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
            
        if KO_idx != None:
            motif_act[:,KO_idx,:] = 0
            motif_act[:,KO_idx+motifs.shape[0],:] = 0

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
        # self.decoder3 = Decoder(inp_dim=inp_dim)

        self.act =  nn.Sequential(
            nn.Conv1d(inp_dim,2,kernel_size=1,stride=1,padding=0,bias=False),
            nn.Softplus(),
        )

    def forward(self,x,mask=0,KO_idx=None):
        encode_feature = self.encoder(x,KO_idx=KO_idx)
        decode_feature = self.decoder1(encode_feature,mask=mask)
        decode_feature = self.decoder2(decode_feature,mask=mask)
        out = self.act(decode_feature)
        return out
    
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




w82_tfeature = GenomicSignalFeatures(
  [
    rf"KO_YY1/930-3/Wm82-NJAU.930-3.plus.bw",
    rf"KO_YY1/930-3/Wm82-NJAU.930-3.minus.bw",
],
    [
        f"{tissue}_plus",
        f"{tissue}_minus",
    ],
    (4000,),
)



seqs = []
tars = []

predict_dict = {
    'chrom':[],
    'pos':[],
    'strand':[],
    'gene':[],
    'seq':[],
    'tar':[],
    'TSR_max_level':[],
    'TSR_sum_level':[],
}
print('select thick.start')

motif_name = 'YY1'
KO_motif = '[A][T][G][G][C]'


w82_tsses = w82_tsses[w82_tsses.iloc[:,-1] >1]


promoter = w82_tsses[w82_tsses['txType'] =='promoter']
promoter = promoter[promoter['peakTxType'] =='promoter']

UTR = w82_tsses[w82_tsses['txType'] =='fiveUTR']
UTR = UTR[UTR['peakTxType'] == 'fiveUTR']

TSS_hc = pd.concat([promoter,UTR],axis=0)
TSS_hc = TSS_hc[ ~TSS_hc['seqnames'].str.contains('scaffold') ]
TSS_hc = TSS_hc.sort_values(by='score',ascending=False)
w82_tsses = TSS_hc.iloc[:,:]
w82_n_tsses = len(w82_tsses)
print(w82_n_tsses)
for randi in range(w82_n_tsses):
    
    seqnamesm, pos, strand ,gene = (
        w82_tsses["seqnames"].values[randi],
        w82_tsses["thick.start"].values[randi],
        w82_tsses["strand"].values[randi],
        w82_tsses["txID"].values[randi],

    )
    
    try:
        if strand == '-':
            seq = w82_genome.get_encoding_from_coords(
                w82_tsses["seqnames"].values[randi],
                w82_tsses["thick.start"].values[randi] - 825,
                w82_tsses["thick.start"].values[randi] + 3825,
                w82_tsses["strand"].values[randi],
            )
            str_seq = w82_genome.get_sequence_from_coords(
                w82_tsses["seqnames"].values[randi],
                w82_tsses["thick.start"].values[randi] - 825,
                w82_tsses["thick.start"].values[randi] + 3825,
                w82_tsses["strand"].values[randi],)
            
            tar = w82_tfeature.get_feature_data(
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
            
            tar = w82_tfeature.get_feature_data(
                w82_tsses["seqnames"].values[randi],
                w82_tsses["thick.start"].values[randi] - 3825,
                w82_tsses["thick.start"].values[randi] + 825,
            )
            str_seq = w82_genome.get_sequence_from_coords(
                w82_tsses["seqnames"].values[randi],
                w82_tsses["thick.start"].values[randi] - 3825,
                w82_tsses["thick.start"].values[randi] + 825,
                w82_tsses["strand"].values[randi],)
    except:
        continue
    # window_size = 5
    # tar_smoothed = []
    # # 创建平滑后的数组
    # for i in range(tar.shape[0]):
    #     smoothed_data = np.convolve(tar[i], np.ones(window_size)/window_size, mode='same')
    #     tar_smoothed.append(smoothed_data)
    if re.search(KO_motif, str_seq[-1325:-825]):
        # print(re.match(KO_motif, str_seq[325:-325]))
        pass
    else:
        continue

    # tar = np.array(tar_smoothed)
    predict_dict['chrom'].append(seqnamesm)
    predict_dict['pos'].append(pos)
    predict_dict['strand'].append(strand)
    predict_dict['gene'].append(gene)
    predict_dict['seq'].append(seq)
    predict_dict['tar'].append(tar[0,325:-325].flatten())
    predict_dict['TSR_max_level'].append(tar.max())
    predict_dict['TSR_sum_level'].append(tar.sum())

predict_dataframe = pd.DataFrame(predict_dict,dtype=object) 
print(predict_dataframe)
class myDataset(Dataset):
    def __init__(self,predict_dict):
        super().__init__()
        self.pred_dict = predict_dict
        self.pred_dataframe = pd.DataFrame(predict_dict)
    def __len__(self):
        return len(self.pred_dict['seq'])
    def __getitem__(self, index):
        seq = torch.tensor(self.pred_dict['seq'][index]).permute(1,0)
        tar = np.array(self.pred_dict['tar'][index])
        return seq,tar,index
batchsize = 1024
dataloader = DataLoader(myDataset(predict_dict),batch_size=batchsize,shuffle=False)


dir_path = 'new_model/930-3'
nets_ = os.listdir(dir_path)
nets_ = [torch.load(os.path.join(dir_path,i)).to(device).eval() for i in nets_ if i.startswith(f'predict',0)]

#获取motif列表
motif_list = []
for idx,motif in enumerate(motifs[:18,:,:]):
    npy = reset(motif.T).astype(object)
    npy[:,0] = np.where(npy[:,0] > 0 ,'A',0)
    npy[:,1] = np.where(npy[:,1] > 0 ,'C',0)
    npy[:,2] = np.where(npy[:,2] > 0 ,'G',0)
    npy[:,3] = np.where(npy[:,3] > 0 ,'T',0)
    npy = npy.T
    idx_= np.argwhere(np.count_nonzero(npy =='0',axis=0) != 4)
    start,end = idx_[0][0],idx_[-1][0]
    motif_str = r''
    for col in range(start,end + 1):
        count = np.count_nonzero(npy[:,col] =='0')
        if count != 4:
            position_value = ''
            for element in npy[:,col]:
                if element != '0':
                    position_value += element
            position_value = f'[{position_value}]'
            motif_str += position_value
        else:
            motif_str+='.'
    motif_list.append(motif_str)



KO_idx = motif_list.index(KO_motif)
print(KO_idx)
# os.makedirs()

for batch_id,(seq,target,idx) in enumerate(dataloader):
    idx = list(idx)
    seq = seq.to(device)
    info = predict_dataframe.iloc[idx,:].drop(['seq','tar'],axis=1).reset_index(drop=True)

    with torch.no_grad():
        tss_pos = torch.argmax(target,dim=1)
        tss_pos = torch.clamp(tss_pos, min=200, max=3800)
        pred_list = [ net(seq,KO_idx=None).detach().cpu().numpy()[:,0,325:-325] for net in nets_]

        pred = torch.tensor(np.array(pred_list).mean(axis=0))
        ko_pred_list = [ net(seq,KO_idx=KO_idx).detach().cpu().numpy()[:,0,325:-325] for net in nets_]
        KO_pred = torch.tensor(np.array(ko_pred_list).mean(axis=0))
        print(pred.shape)

        KO_pred = torch.tensor(np.array([row[pos-200:pos+200] for pos,row in zip(tss_pos,KO_pred)]))
        pred = torch.tensor(np.array([row[pos-200:pos+200] for pos,row in zip(tss_pos,pred)]))
        
    result = torch.cat([KO_pred.mean(dim=1,keepdim=True),
                        KO_pred.max(dim=1,keepdim=True).values,
                        KO_pred.sum(dim=1,keepdim=True),
                        pred.mean(dim=1,keepdim=True),
                        pred.max(dim=1,keepdim=True).values,
                        pred.sum(dim=1,keepdim=True)],dim=1).numpy()

    result = pd.DataFrame(result,columns=['KO_mean','KO_max','KO_sum','Pred_mean','Pred_max','Pred_sum'])
    result = pd.concat([info,result],axis=1)
    result.to_csv(rf'analyze/new_model_result/net_predict_KO_{motif_name}_W82_info_root.csv',header=False if batch_id!=0 else True,index=None,mode='a')
    print(f"net: {(batch_id+1) * batchsize}/{len(predict_dataframe)}")

