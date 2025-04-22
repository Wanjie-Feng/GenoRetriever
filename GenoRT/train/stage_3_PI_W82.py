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
from Module import *
import random
import sys
from scipy.stats import gaussian_kde
from torch.optim.lr_scheduler import CosineAnnealingLR
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
    motifpwm = motifpwm / motifpwm.max()
    return motifpwm
#统计所有位点的位置、信号值、不同motif对该位点的效应以及预测值与真实值之间的相关系数大小
tissue = 'merge'
device = torch.device('cuda:3')
motifs = np.load(rf"analyze/result/sorted_cl_motif.npy")

other_motifs = np.load(rf"/Data5/pfGao/xtwang/TSS/tss/analyze/result/pi_w82_other.npy")
# tri_motifs = np.load(rf"/Data5/pfGao/xtwang/TSS/tss/analyze/result/tri.npy")
print(motifs.shape)
n_motifs = motifs.shape[0]
n_others = other_motifs.shape[0] 
# n_tris = tri_motifs.shape[0] 
n_tsses = 40000
sys.path.append("../utils/")
pi_tsses = pd.read_table(
    rf"/Data6/wanjie/other_data_archives/STRIPE/PI46/Total//Unidirection_result.txt",
    sep="\t",
)
pi_genome = selene_sdk.sequences.Genome(
    input_path=r"PI48916/PI46.t2t.final.fa",
)

w82_tsses = pd.read_table(
   rf"data_process/Unidirection_result.txt",
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
        wigmat = np.log(wigmat + 1)
        return wigmat   


pi_tfeature = GenomicSignalFeatures(input_paths=[
    rf"PI48916/filter_bw/plus.filter.bw",
    rf"PI48916/filter_bw/minus.filter.bw",
]            
                                       
 ,features=[f'merge.plus',f'merge.minus'],shape=(4000,))


w82_tfeature = GenomicSignalFeatures(
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
pi_tsses = pi_tsses[pi_tsses['support'] >1]


promoter = pi_tsses[pi_tsses['txType'] =='promoter']
promoter = promoter[promoter['peakTxType'] =='promoter']

UTR = pi_tsses[pi_tsses['txType'] =='fiveUTR']
UTR = UTR[UTR['peakTxType'] == 'fiveUTR']

TSS_hc = pd.concat([promoter,UTR],axis=0)
TSS_hc = TSS_hc[ ~TSS_hc['seqnames'].str.contains('scaffold') ]
pi_n_tsses = len(pi_tsses)

print(pi_n_tsses)
predict_dict = {
    'chrom':[],
    'pos':[],
    'strand':[],
    'seq':[],
    'tar':[],
    'TSR_max_level':[],
    'TSR_sum_level':[],
}


for randi in range(pi_n_tsses):
    
    seqnamesm, pos, strand = (
        pi_tsses["seqnames"].values[randi],
        pi_tsses["thick.start"].values[randi],
        pi_tsses["strand"].values[randi],
    )
    
    try:
        if strand == '-':
            seq = pi_genome.get_encoding_from_coords(
                pi_tsses["seqnames"].values[randi],
                pi_tsses["thick.start"].values[randi] - 825,
                pi_tsses["thick.start"].values[randi] + 3825,
                pi_tsses["strand"].values[randi],
            )
            
            tar = pi_tfeature.get_feature_data(
                pi_tsses["seqnames"].values[randi],
                pi_tsses["thick.start"].values[randi] - 825 ,
                pi_tsses["thick.start"].values[randi] + 3825,
            )
            tar = tar[::-1, ::-1]
            
        else:
            seq = pi_genome.get_encoding_from_coords(
                pi_tsses["seqnames"].values[randi],
                pi_tsses["thick.start"].values[randi] - 3825,
                pi_tsses["thick.start"].values[randi] + 825,
                pi_tsses["strand"].values[randi],
            )
            
            tar = pi_tfeature.get_feature_data(
                pi_tsses["seqnames"].values[randi],
                pi_tsses["thick.start"].values[randi] - 3825,
                pi_tsses["thick.start"].values[randi] + 825,
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
    predict_dict['chrom'].append(seqnamesm)
    predict_dict['pos'].append(pos)
    predict_dict['strand'].append(strand)
    predict_dict['seq'].append(seq)
    predict_dict['tar'].append(tar)
    predict_dict['TSR_max_level'].append(tar.max())
    predict_dict['TSR_sum_level'].append(tar.sum())



w82_tsses = w82_tsses[w82_tsses['support'] >1]


promoter = w82_tsses[w82_tsses['txType'] =='promoter']
promoter = promoter[promoter['peakTxType'] =='promoter']

UTR = w82_tsses[w82_tsses['txType'] =='fiveUTR']
UTR = UTR[UTR['peakTxType'] == 'fiveUTR']

TSS_hc = pd.concat([promoter,UTR],axis=0)
TSS_hc = TSS_hc[ ~TSS_hc['seqnames'].str.contains('scaffold') ]
w82_n_tsses = len(w82_tsses)
print(w82_n_tsses)

for randi in range(w82_n_tsses):
    
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

    except:
        continue
    window_size = 5
    tar_smoothed = []
    # 创建平滑后的数组
    for i in range(tar.shape[0]):
        smoothed_data = np.convolve(tar[i], np.ones(window_size)/window_size, mode='same')
        tar_smoothed.append(smoothed_data)
    tar = np.array(tar_smoothed)
    predict_dict['chrom'].append(seqnamesm)
    predict_dict['pos'].append(pos)
    predict_dict['strand'].append(strand)
    predict_dict['seq'].append(seq)
    predict_dict['tar'].append(tar)
    predict_dict['TSR_max_level'].append(tar.max())
    predict_dict['TSR_sum_level'].append(tar.sum())




batchsize = 64


def KL(pred, target):
    
    return target * (torch.log(target + 1e-10) - torch.log(pred + 1e-10)) + (pred - target).abs()


for epoch in range(0,15):

    predict_dataframe = pd.DataFrame.from_dict(predict_dict) 
    w82_chrom = [  i for i in set(predict_dataframe['chrom']) if 'GWHCAYC' in i ]
    pi_chrom =  [  i for i in set(predict_dataframe['chrom']) if 'Gs' in i ]
    pi_chrom.sort()
    w82_chrom.sort()
    chr_id = random.sample([i for i in range(len(w82_chrom))],2)
    chr_id = [7,9]
    test_chrom = []
    for idx in chr_id:
        test_chrom.append(pi_chrom[idx])
        test_chrom.append(w82_chrom[idx])


    test_dataframe = predict_dataframe[predict_dataframe['chrom'].isin(test_chrom)]
    train_dataframe = predict_dataframe[~predict_dataframe['chrom'].isin(test_chrom)]
    del predict_dataframe

    train_seqs = np.stack(train_dataframe['seq']).transpose([0, 2, 1])
    train_tars = np.stack(train_dataframe['tar'])
    test_seqs = np.stack(test_dataframe['seq']).transpose([0, 2, 1])
    test_tars = np.stack(test_dataframe['tar'])
    del train_dataframe,test_dataframe

    net = Geno_RT(n_motifs,n_others)
    weights = torch.ones(2).to(device)
    net.to(device)
    net.train()

    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=0.0001,weight_decay=0.001)
    stime = time.time()
    i = 0
    bestcor = 0
    past_losses = []
    past_l2 = []
    testation_loss_stage3 = []
    bestcor = 0
    scheduler = CosineAnnealingLR(optimizer, T_max=100)
    flag = False
    counter = 0
    for iter in range(20):

        randinds = np.random.permutation(np.arange(len(train_seqs)))
        train_seqs = train_seqs[randinds,:,:]
        train_tars = train_tars[randinds,:,:]

        for j in range(train_seqs.shape[0] // batchsize):
            net.train()
            sequence = train_seqs[j * batchsize : (j + 1) * batchsize, :, :]
            target = train_tars[j * batchsize : (j + 1) * batchsize, :, :]

            sequence = torch.FloatTensor(sequence)
            target = torch.FloatTensor(target)
            if torch.rand(1) < 0.5:
                sequence = sequence.flip([1, 2])
                target = target.flip([1, 2])
            optimizer.zero_grad()

            pred = net(torch.Tensor(sequence.float()).to(device),mask=0)
    
            loss0 = (
                KL(pred[:, :, 325:-325], target.to(device)[:, :, 325:-325])
                * weights[None, :, None]
            ).mean()

            loss = loss0

            loss.backward()
            past_losses.append(loss0.detach().cpu().numpy())
            optimizer.step()
            scheduler.step()

            if i % 100 == 0:
                print(f"{iter} train loss:" + str(np.mean(past_losses[-100:])), flush=True)
                past_losses = []
                past_l2 = []
            if i % 200 == 0:
                with torch.no_grad():
                    net.eval()
                    cor = []
                    past_losses = []

                    batchsize = 1
                    predict_chr = []
                    label_chr = []
                    for j in range(test_seqs.shape[0] // batchsize):
                        sequence = test_seqs[j * batchsize : (j + 1) * batchsize, :, :]
                        target = test_tars[j * batchsize : (j + 1) * batchsize, :, :]
                        sequence = torch.FloatTensor(sequence)
                        target = torch.FloatTensor(target)

                        if torch.rand(1) < 0.5:
                            sequence = sequence.flip([1, 2])
                            target = target.flip([1, 2])

                        optimizer.zero_grad()
                        pred = net(torch.Tensor(sequence.float()).to(device),mask=0)

                        predict_chr.append(pred[:, :, 325:-325].detach().cpu().numpy())
                        label_chr.append(target[:, :, 325:-325].numpy())
                        
                        loss0 = (
                            KL(pred[:, :, 325:-325], target.to(device)[:, :, 325:-325])
                            * weights[None, :, None]
                        ).mean()
                        cor.append(np.corrcoef(pred[:, :, 325:-325].detach().cpu().numpy().flatten(), target[:, :, 325:-325].numpy().flatten())[0,1])
                        
                        past_losses.append(loss0.detach().cpu().numpy())
                        testloss = np.mean(past_losses)
                        testcor = np.mean(cor)
                        maxcor = np.max(cor)
                    batchsize = 64
                    net.train()
                predict_chr = np.array(predict_chr).flatten()
                label_chr = np.array(label_chr).flatten()
                all_cor = np.corrcoef(predict_chr,label_chr)[0,1]
                print(iter,"test loss:" + str(np.mean(past_losses)) + " test cor:" +str(np.mean(cor)) + ' max cor:' + str(maxcor) + ' all cor:' + str(all_cor), flush=True)
                if testcor > bestcor:
                    try:
                        os.remove(f'new_model/W82_for_draw/fine-tune_predict_pi_w82_merge_stage_3_for_{tissue}_{epoch}_avgcor_{"{:.7f}".format(bestcor)}.best.pt')
                    except FileNotFoundError:
                        pass
                    torch.save(net,f'new_model/W82_for_draw/fine-tune_predict_pi_w82_merge_stage_3_for_{tissue}_{epoch}_avgcor_{"{:.7f}".format(testcor)}.best.pt')
                    bestcor = testcor
                    counter = 0
                else:
                    counter += 1

            i = i + 1


