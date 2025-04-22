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
import re

motifs = np.load(rf"analyze/result/sorted_cl_motif.npy")
other_motifs = np.load(rf"/Data5/pfGao/xtwang/TSS/tss/analyze/result/pi_w82_other.npy")
strand = '-'
pos = 4465781
chrom = 'GWHCAYC00000001'
gene_id = 'GmW82.01G042700'
motif = 'TATAAA'
one_hot_mapping = { 'A': [1, 0, 0, 0],
                    'C': [0, 1, 0, 0],
                    'G': [0, 0, 1, 0],
                    'T': [0, 0, 0, 1],
                } 


w82_genome = selene_sdk.sequences.Genome(
    input_path=r"/Data5/pfGao/xtwang/TSS/W82-NJAU/genome/Wm82-NJAU.fasta",
    )
w82_tsses = pd.read_table(
   rf"/Data5/pfGao/xtwang/TSS/tss/W82_bw/TSR/W82.Leaf.Unidirection_result0.txt",
    sep="\t",
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
    
w82_tfeature = GenomicSignalFeatures(
  [
    rf"/Data5/pfGao/xtwang/TSS/tss/data_process/filter_bw/W82.Leaf.plus.filter.bw",
    rf"/Data5/pfGao/xtwang/TSS/tss/data_process/filter_bw/W82.Leaf.minus.filter.bw",
],
    [
        f"plus",
        f"minus",
    ],
    (4000,),
)

map_dict = str.maketrans({
    'A':'T',
    'C':'G',
    'G':'C',
    'T':'A'
} )

padding = len(motif)
# padding = 0
if strand == '-':
    seq = w82_genome.get_encoding_from_coords(chrom,pos-825,pos+3825 - padding,strand)
    tar = w82_tfeature.get_feature_data(chrom,pos - 825,pos+3825,)
    tar = tar[::-1,::-1].copy()
    str_seq = w82_genome.get_sequence_from_coords(chrom,pos-825,pos+3825-padding,strand)
    orgin_seq = w82_genome.get_encoding_from_coords(chrom,pos-825,pos+3825,strand)
else:
    seq = w82_genome.get_encoding_from_coords(chrom,pos-3825 + padding,pos+825,strand)
    tar = w82_tfeature.get_feature_data(chrom , pos-3825, pos+825)
    str_seq = w82_genome.get_sequence_from_coords(chrom,pos-3825 + padding,pos+825,strand)
    orgin_seq = w82_genome.get_encoding_from_coords(chrom,pos-3825,pos+825,strand)
# print(str_seq)
print(len(seq))

import os
device = 'cuda:1'
net_dir = r'/Data5/pfGao/xtwang/TSS/tss/new_model/W82_tissues'
net_list = [ torch.load(os.path.join(net_dir,i)).to(device).eval() for i in os.listdir(net_dir) if i.startswith(r'1w82_fine-tune',0)]
# net_list = [torch.load(r"merge_stage_3_model/model_for_predict/plus_predict_pi_w82_merge_stage_3_for_merge_0_cor_0.6869734.best.pt").to(device).eval()]
# print(len(net_list))

seqs = []
# # ----------------单碱基扰动----------------#
tss_pos = 3825 - len(motif)
ATG_pos = np.array([ m.end() - tss_pos for m in re.finditer('ATG',str_seq) if m.end() - tss_pos >0 ]).min()
# start_pos = tss_pos - 1200
end_pos = tss_pos + ATG_pos - 3
start_pos = end_pos - 1000
print(ATG_pos)
print(start_pos,end_pos)
print(gene_id,chrom,pos,strand)
print(str_seq[start_pos:end_pos + 3])

import copy
import matplotlib.pyplot as plt
with torch.no_grad():
    orig_seq = torch.FloatTensor(orgin_seq.copy()[None,:,:]).permute(0,2,1).repeat(end_pos-start_pos,1,1).to(device)
    pred = torch.stack([ net(orig_seq).detach().cpu() for net in net_list ]).mean(dim=0)[:,0,3825-200:3825+200]
    print(pred.argmax(dim=1))
    reseqs=[]
    for pos_id,pos_ in enumerate(range(start_pos,end_pos)):
        reseq = str_seq[:pos_] + motif + str_seq[pos_:]
        reseq = np.vstack(pd.Series(list(reseq)).map(one_hot_mapping)).T
        reseqs.append(list(reseq))
    reseqs = np.array(reseqs)
    print(reseqs.shape)
    reseqs = torch.FloatTensor(reseqs).to(device)
    re_pred = torch.stack([ net(reseqs).detach().cpu() for net in net_list ]).mean(dim=0)[:,0,3825-200:3825+200]
    print(re_pred.argmax(dim=1))
    print(re_pred.shape,pred.shape)
    effect = np.array((re_pred.sum(dim=1) - pred.sum(dim=1)) / pred.sum(dim=1))
    result = pd.DataFrame(effect,columns=['effect'])
    label = np.zeros(len(effect))
    label[tss_pos - start_pos] = 1
    result['label'] = label
    result['seq'] = list(str_seq[start_pos:end_pos])
    result.to_csv(rf'analyze/new_model_result/1test_{gene_id}_{chrom}_{pos}_{strand}_1000_insert_{motif}.csv',index=None)
    print(result)


















# # #————————————————变异序列预测——————————————————#

# orign = '''GTGGACTATGTAGACTCTAAACTCGATAAACTCTAAAATTTACTAACGAGTCAGCAAATTACAAAATTAGACCAAAATATATATCATTTTGGATTAATTTTTATGCATATAAGAATCATCATACTTCTATTTAGTGTGTTATTCCTGAATAAAAAAACACTCGTCTCTAAAAATATTAAACACTTGTTTTTAAACAACATCAAATCCTCAATGAGAATGTTCTACCATTAATATCACCTCCGTAAGTTATATTATTGTTATCTAATACTAGACTTGATTATTTAAGTAAGTAAAATTTATATTTTACTATTTTGCTTTATTATGTTGTTAAAATATTTAACTGATATGTTACTTATAAATATTTTATTATTATTTTTATATAAAGTATAAGTTTACTAATGGAGTCCATGAGTTAAATTTATAAAACTCTCACAAGTTTATATATAGACTTTCAAGTTTGATACATGCTAATAATAATGCTTCTTAATCCTTATTTATAAATAACTGATTTTGGAGCAAAAAGTTTATATACTTTTTGTCAATTTATAAAATAAAAGAATAGAAAAAGAAAACTATATTCTTCCATGTAAATTGTAACAAAGATTAAGGATATATGCCTAATAAAAAGACAAAAGCAAATGAAAATACTAGAAAAATTAACCCGAGGAAGTAAACACGCAACGAGAAATTCGTACATGAAGACTGGGAACAACGGACTGTTCCGATAATGCCTTTGTCCAATCCGAACTCGACAACGTTTTTGACGAAACCGCTCTGGTTTCACAAAACCACCTTCTCACTTTCACTTTTCCCTTTTCTTTTCGTGGAATATAATTACAATAACCAATGATTTTTCGTCCTCTTGTTTTGTTTTATGTTTTATAATGCCCACTTTATAAATAAAATAATAAAAAAATGAATGAATGAATCAAGGGGTGAGACTCACGGTCATGACTCATGTAAATTCGTCTCCCACAACAACACTAGCTCTCACGTACATAAATATAAAGATGTTGATAGCATATTATAAATCATAGAGTTCATTCCATACACAGAAAGAAGAACAACACTAAGAAAGCAGAATAGGGAAAAATACATCGATAGAGGAAAATTAATTGAAGGGTGCTACTACTACGACCTTCTATATCATTAATGATTGAAATTCGTGTTGGAATTGTCTTGTAGGTGTAGTAGTGATAATAAGAGAGGATCATATATATATTAGTAGTGCTAGAAATTAATTGAAGTGTGTAATAGAATAGAGTGTGATAGAGAGAGAT'''
# var = '''GTGGACTATGTAGACTCTAAACTCGATAAATTCTAGAATTTACTAACGAGTCAGCAAATTACAAAATTAGACCAAAATATATATCATTTTGGATTAATTTTTATGCATATAAGAATCATCATACTTCTATTTAGTGTGTTATTCCTGAATAAAAAAACACTCGTCTCTAAAAACATTAAACACTTGTTTTTAAACAACATCAAATTCTCAATGAGAATGTTCTACCATTAATATCACCTCCGTAAGTTATATTATTGTTATCTAATACTAGACTTGATTATTTAAGTATGTAAAATTTATATTTTACTATTTTGCTTTATTATGTTGTTAAAATATTTAACTGATATGTTACTTATAAATATTTTATTATTATTTTTATATAAAGTATAAGTTTACTAATGGAGTCCATGAGTTAAATTTATAAAACTCTCACAAGTTTATATATAGACTTTCAAGTTTGATACATGCTAATAATAATGCTTCTTAATCCTTATTTATAAATAACTGATTTTGGAGCAAAAAGTTTATATACTTTTTGTCAATTTATAAAATAAAAGAATAGAAAAAGAAAACTATATTCTTCCATGTAAATTGTAACAAAGATTAAGGATATATGCCTAATAAAAAGACAAAAGCAAATGAAAATACTAGAAAAATTAACCCGAGGAAGTAAACACGCAACGAGAAATTCGTACATGAAGACTGGGAACAACGGACTGTTCCGATAATGCCTTTGTCCAATCCGAACTCGACAACGTTTTTGACGAAACCGCTCTGGTTTTACAAAACCACCTTCTCACTTTCACTTTTCCCTTTTCTTTTCGTGGAATATAATTACAATAACCAATGATTTTTCGTCCTCTTGTTTTGTTTTATGTTTTATAATGCCCACTTTATAAATAAAATAATAAAAAAATGAATGAATGAATCAAGGGGTGAGACTCACGGTCATGACTCATGTAAATTCGTCTCCCACAACAACAATAGCTCTCACGTACATAAATATAAAGATGTTGATAGCATATTATAAATCATAGAGTTCATTCCATACACAGAAAGAAGAACAACACTAAGAAAGCAGAATAGGGAAAAATACATCGATAGAGGAAAATTAATTGAAGGGTGCTACTACTACGACCTTCTATATCATTAATGATTGAAATTCGTGTTGGAATTGTCTTGTAGGTGTAGTAGTGATAATAAGAGAGGATCATATATATATTAGTAGTGCTAGAAATTAATTGAAGTGTGTAATAGAATAGAGTGTGATAGAGAGAAAT'''

# var = var.replace("\n", "")
# orign = orign.replace("\n", "")

# var_seq = str_seq[:end_pos-len(var)] + var + str_seq[end_pos:]
# orign_seq = str_seq[:end_pos-len(orign)] + orign + str_seq[end_pos:]
# # orign_seq = str_seq
# # print(orign)
# # print(orign_seq == str_seq)
# orign_seq = torch.FloatTensor(np.vstack(pd.Series(list(orign_seq)).map(one_hot_mapping)).T[None,:,:]).to(device)
# # print((seq == np.array(orign_seq[0].T.cpu())).all())
# var_seq = torch.FloatTensor(np.vstack(pd.Series(list(var_seq)).map(one_hot_mapping)).T[None,:,:]).to(device)

# import matplotlib.pyplot as plt
# from scipy.stats import ttest_ind

# # inp = torch.cat([orign_seq,var_seq],dim=0)
# # print(inp.shape)
# with torch.no_grad():
#     org_pred = np.array(torch.stack([net(orign_seq).detach().cpu() for net in net_list ]).mean(dim=0)[:,0,-1325:-325])
#     var_pred = np.array(torch.stack([net(var_seq).detach().cpu() for net in net_list ]).mean(dim=0)[:,0,-1325:-325])
#     plt.plot(np.arange(1000),org_pred.flatten(),c='orange')
#     plt.plot(np.arange(1000,2000),var_pred.flatten(),c='blue')
#     plt.savefig('test.png')
#     t_stat, p_value = ttest_ind(org_pred.flatten(),var_pred.flatten())

#     print(p_value)
#     exp_change = (var_pred-org_pred).sum() / org_pred.sum()
#     print(exp_change)