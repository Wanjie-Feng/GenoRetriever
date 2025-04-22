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
import logomaker
import matplotlib.pyplot as plt

tissue = 'Stemtip'
motifs = np.load(rf"/Data5/pfGao/xtwang/TSS/tss/analyze/result/sorted_motif.npy")
motifs_deconv = np.load(rf"/Data5/pfGao/xtwang/TSS/tss/analyze/result/sorted_deconv.npy")
print(motifs_deconv.shape)
n_motifs = motifs.shape[0]

n_tsses = 40000
sys.path.append("../utils/")
tsses = pd.read_table(
    rf"/Data6/wanjie/other_data_archives/STRIPE/PI46/{tissue}/Unidirection_result.txt",
    sep="\t",
)
genome = selene_sdk.sequences.Genome(
    input_path=r"PI48916/PI46.t2t.final.fa",
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


tfeature = GenomicSignalFeatures(input_paths=[
    rf"PI48916/PI46.{tissue}.plus.bw",
    rf"PI48916/PI46.{tissue}.minus.bw",
]
                                 
 ,features=[f'{tissue}.plus',f'{tissue}.minus'],shape=(4000,))


window_size = 4650
seqs = []
tars = []
print('select thick.start')
tsses = tsses[tsses[f'{tissue}.1'] >=1]

promoter = tsses[tsses['txType'] =='promoter']
promoter = promoter[promoter['peakTxType'] =='promoter']

UTR = tsses[tsses['txType'] =='fiveUTR']
UTR = UTR[UTR['peakTxType'] == 'fiveUTR']

TSS_hc = pd.concat([promoter,UTR],axis=0)
TSS_hc = TSS_hc[ ~TSS_hc['seqnames'].str.contains('scaffold') ]
n_tsses = len(tsses)
print(n_tsses)

for randi in range(n_tsses):
    
    seqnamesm, pos, strand = (
        tsses["seqnames"].values[randi],
        tsses["thick.start"].values[randi],
        tsses["strand"].values[randi],
    )
    try:
        if strand == '-':
            seq = genome.get_encoding_from_coords(
                tsses["seqnames"].values[randi],
                tsses["thick.start"].values[randi] - 825,
                tsses["thick.start"].values[randi] + 3825,
                tsses["strand"].values[randi],
            )
            
            tar = tfeature.get_feature_data(
                tsses["seqnames"].values[randi],
                tsses["thick.start"].values[randi] - 825 ,
                tsses["thick.start"].values[randi] + 3825,
            )
            
            tar = tar[::-1, ::-1]
            
        else:
            seq = genome.get_encoding_from_coords(
                tsses["seqnames"].values[randi],
                tsses["thick.start"].values[randi] - 3825,
                tsses["thick.start"].values[randi] + 825,
                tsses["strand"].values[randi],
            )
            
            tar = tfeature.get_feature_data(
                tsses["seqnames"].values[randi],
                tsses["thick.start"].values[randi] - 3825,
                tsses["thick.start"].values[randi] + 825,
            )
    except:
        continue
    seqs.append(seq)
    tars.append(tar)
print('make dataset')

num_seq = len(seqs)
seqs = np.dstack(seqs)
tars = np.dstack(tars)
seqs = seqs.transpose([2, 1, 0])
tars = tars.transpose([2, 0, 1])
np.random.seed(1)
randinds = np.random.permutation(np.arange(num_seq))
seqs = seqs[randinds, :]
tars = tars[randinds, :]
tsses_rand = tsses.iloc[randinds, :]
train_seqs = seqs[~tsses_rand["seqnames"].isin([ "Gs10",'Gs07']).values, :]
valid_seqs = seqs[tsses_rand["seqnames"].isin(["Gs10",'Gs07']).values, :]
train_tars = tars[~tsses_rand["seqnames"].isin(["Gs10",'Gs07']).values, :]
valid_tars = tars[tsses_rand["seqnames"].isin(["Gs10",'Gs07']).values, :]

class GenoRT_s2(nn.Module):
    def __init__(self,n_motifs):
        super(GenoRT_s2, self).__init__()
        
        self.conv = nn.Conv1d(4, n_motifs, kernel_size=51, padding=25)
        self.conv_inr = nn.Conv1d(4, 40, kernel_size=3, padding=1)
        self.deconv = nn.Sequential(
            nn.Conv1d(in_channels=n_motifs * 2,out_channels=2, kernel_size=601, padding=300),
            nn.BatchNorm1d(2))
        self.deconv_inr = nn.Sequential(
            nn.Conv1d(80, 2, kernel_size=15, padding=7),
            nn.BatchNorm1d(2))

        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()
        self.scaler = nn.Parameter(torch.ones(1))
        self.scaler2 = nn.Parameter(torch.ones(1))

    def forward(self, x,withscaler=True):
        y = torch.cat([self.conv(x), self.conv(x.flip([1, 2])).flip([2])], 1)
        y_inr = torch.cat(
            [self.conv_inr(x), self.conv_inr(x.flip([1, 2])).flip([2])], 1
        )
        if withscaler:
            yact = self.sigmoid(y * self.scaler**2)

            y_inr_act = self.sigmoid(y_inr * self.scaler2**2)
        else:
            yact = self.sigmoid(y)
            y_inr_act = self.sigmoid(y_inr)
        y_pred = self.softplus(
            self.deconv(yact) + self.deconv_inr(y_inr_act)
        )
        
        return y_pred



def PoissonKL(lpred, ltarget):
    return ltarget * torch.log((ltarget + 1e-10) / (lpred + 1e-10)) + lpred - ltarget


def KL(pred, target):
    pred = (pred + 1e-10) / ((pred + 1e-10).sum(2)[:, :, None])
    target = (target + 1e-10) / ((target + 1e-10).sum(2)[:, :, None])
    return target * (torch.log(target + 1e-10) - torch.log(pred + 1e-10))


def std2(x, axis, dim):
    return ((x - x.mean(axis=axis, keepdims=True)) ** dim).mean(axis=axis) ** (1 / dim)


batchsize = 16
for epoch in range(10,15):

    net = GenoRT_s2(n_motifs=n_motifs).cpu()
    net.conv.weight.data = torch.FloatTensor(motifs).cuda(2)
    net.deconv[0].weight.data = torch.FloatTensor(motifs_deconv).permute(1,0,2).cuda(2)


    weights = torch.ones(2).cuda(2)
    net.cuda(2)
    net.train()
    net.conv.weight.requires_grad = False
    net.conv.bias.requires_grad = True
    net.deconv[0].weight.requires_grad = False
    net.deconv[0].bias.requires_grad = True
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=0.002, weight_decay=0.01)
    
    stime = time.time()
    i = 0
    bestcor = 0
    past_losses = []
    past_l2 = []
    past_l1act = []
    validation_loss_stage3 = []
    bestloss = 100
    flag = 0
    for iter in range(150):
        for j in range(train_seqs.shape[0] // batchsize):
            sequence = train_seqs[j * batchsize : (j + 1) * batchsize, :, :]
            target = train_tars[j * batchsize : (j + 1) * batchsize, :, :]
            sequence = torch.FloatTensor(sequence)
            target = torch.FloatTensor(target)
            if torch.rand(1) < 0.5:
                sequence = sequence.flip([1, 2])
                target = target.flip([1, 2])
            optimizer.zero_grad()
            pred = net(torch.Tensor(sequence.float()).cuda(2))
            loss0 = (
                KL(pred[:, :, 325:-325], target.cuda(2)[:, :, 325:-325])
                * weights[None, :, None]
            ).mean()

            l1inract = (
                net.deconv_inr[0].weight.abs() * (weights**1)[:, None, None]
            ).mean()
            l1inrmotif = (net.conv_inr.weight.abs()).mean()
            l2inr = (
                (
                    (
                        (
                            (
                                net.deconv_inr[0].weight[:, :, :-1]
                                - net.deconv_inr[0].weight[:, :, 1:]
                            )
                        )
                        ** 2
                    ).mean(2)
                    / (std2(net.deconv_inr[0].weight, axis=2, dim=2) ** 2 + 1e-10)
                )
                * (weights**1)[:, None]
            ).mean()
            loss = (
                loss0
                + 5e-4
                * (
                    PoissonKL(
                        pred[:, :, 325:-325] / np.log(10), target.cuda(2)[:, :, 325:-325]
                    )
                    * weights[None, :, None]
                ).mean()
                + l2inr * 5e-4
                + l1inract * 4e-5
                + l1inrmotif * 2e-5
            )

            loss.backward()
            past_losses.append(loss0.detach().cpu().numpy())
            optimizer.step()
            if i % 100 == 0:
                print(f"{iter} train loss:" + str(np.mean(past_losses[-100:])), flush=True)
                past_losses = []

            if i % 1000 == 0:
                with torch.no_grad():
                    past_losses = []
                    for j in range(valid_seqs.shape[0] // batchsize):
                        sequence = valid_seqs[j * batchsize : (j + 1) * batchsize, :, :]
                        target = valid_tars[j * batchsize : (j + 1) * batchsize, :, :]
                        sequence = torch.FloatTensor(sequence)
                        target = torch.FloatTensor(target)
                        if torch.rand(1) < 0.5:
                            sequence = sequence.flip([1, 2])
                            target = target.flip([1, 2])
                        optimizer.zero_grad()
                        pred = net(torch.Tensor(sequence.float()).cuda(2))
                        loss0 = (
                            KL(pred[:, :, 325:-325], target.cuda(2)[:, :, 325:-325])
                            * weights[None, :, None]
                        ).mean()
                        past_losses.append(loss0.detach().cpu().numpy())
                        validloss = np.mean(past_losses)
                print("valid loss:" + str(np.mean(past_losses)), flush=True)
            if validloss < bestloss:
                try:
                    os.remove(f'PI_models/test_{tissue}_stage_2_{epoch}_loss_{"{:.9f}".format(bestloss)}.best.pt')
                except FileNotFoundError:
                    pass
                torch.save(net,f'PI_models/test_{tissue}_stage_2_{epoch}_loss_{"{:.9f}".format(validloss)}.best.pt')
                bestloss = validloss
            i = i + 1

            
