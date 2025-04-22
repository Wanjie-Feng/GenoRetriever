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
from torch.optim.lr_scheduler import ReduceLROnPlateau
n_tsses = 40000
# suffix = sys.argv[1]
modelstr = "stage1_"
print(modelstr)
os.makedirs("./PI_models", exist_ok=True)
# sys.path.append("../utils/")
tissues = 'Stemtip'
tsses = pd.read_table(
    rf"/Data6/wanjie/other_data_archives/STRIPE/PI46/{tissues}/Unidirection_result.txt",
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
        self, seqnamesom, start, end, nan_as_zero=True, feature_indices=None
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
                wigmat[i, :] = self.data[i].values(seqnamesom, start, end, numpy=True)
            except:
                print(seqnamesom, start, end, self.input_paths[i], flush=True)
                raise

        if self.blacklists is not None:
            if self.replacement_indices is None:
                if self.blacklists_indices is not None:
                    for blacklist, blacklist_indices in zip(
                        self.blacklists, self.blacklists_indices
                    ):
                        for _, s, e in blacklist.query(seqnamesom, start, end):
                            wigmat[
                                blacklist_indices,
                                np.fmax(int(s) - start, 0) : int(e) - start,
                            ] = 0
                else:
                    for blacklist in self.blacklists:
                        for _, s, e in blacklist.query(seqnamesom, start, end):
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
                    for _, s, e in blacklist.query(seqnamesom, start, end):
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


tfeature = GenomicSignalFeatures(input_paths=[
    rf"PI48916/PI46.{tissues}.plus.bw",
    rf"PI48916/PI46.{tissues}.minus.bw",
]
                                 
 ,features=[f'{tissues}.plus',f'{tissues}.minus'],shape=(4000,))

window_size = 4650
seqs = []
tars = []
print('select thick.start')
tsses = tsses[tsses[f'{tissues}.1'] >=1]


promoter = tsses[tsses['txType'] =='promoter']
promoter = promoter[promoter['peakTxType'] =='promoter']

UTR = tsses[tsses['txType'] =='fiveUTR']
UTR = UTR[UTR['peakTxType'] == 'fiveUTR']

TSS_hc = pd.concat([promoter,UTR],axis=0)
TSS_hc = TSS_hc[ ~TSS_hc['seqnames'].str.contains('scaffold') ]
TSS_hc = TSS_hc.sort_values(by='score',ascending=False)
tsses = TSS_hc.iloc[:40000,:]
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
    except RuntimeError:
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
        self.weight_score = nn.Parameter(torch.randn((feature_num,1)))
        self.act = nn.Softmax(dim=0)
    def forward(self,x):
        atten = self.act(self.weight_score) * x
        return atten + x
    
class GenoRT(nn.Module):
    def __init__(self):
        super(GenoRT, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(4, 40, kernel_size=51, padding=25),
            )
        
        self.activation = nn.Sigmoid()

        self.deconv = nn.Sequential(
            nn.Conv1d(80, 2, kernel_size=601, padding=300),
            nn.BatchNorm1d(2),
            )
        
        self.softplus = nn.Softplus()
        
        
    def forward(self, x):
        
        y = torch.cat([self.conv(x), 
                       self.conv(x.flip([1, 2])).flip([2])], dim=1)
                
        y_act = self.activation(y) * y
        y_pred = self.softplus(self.deconv(y_act))
    
        return y_pred

    
def PseudoPoissonKL(pred, target):
    return target * torch.log((target + 1e-10) / (pred + 1e-10)) + pred - target


def KL(pred, target):
    pred = (pred + 1e-10) / ((pred + 1e-10).sum(2)[:, :, None])
    target = (target + 1e-10) / ((target + 1e-10).sum(2)[:, :, None])
    return target * (torch.log(target + 1e-10) - torch.log(pred + 1e-10))



def std2(x, axis, dim):
    return ((x - x.mean(axis=axis, keepdims=True)) ** dim).mean(axis=axis) ** (1 / dim)

print('Training')
for epoch in range(12,15):
    batchsize = 16
    stime = time.time()
    i = 0
    bestcor = 0
    past_losses = []
    past_l2 = []
    past_l1act = []
    train_losses_stage1 = []
    valid_losses_stage1 = []
    weights = torch.ones(2).cuda(2)
    bestloss = np.inf
    # modelstr = f"soybean_mutit_tissues_stage1_test"
    net = GenoRT()
    # net = SimpleNet()

    net.cuda(2)
    net.train()
    params = [p for p in net.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(params, lr=0.002, weight_decay=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=10, threshold=0)
    
    while True:
        if i // 1000 <= 200 : 
            for j in np.random.permutation(range(train_seqs.shape[0] // batchsize)):
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

                l2 = (
                    (
                        (
                            (net.deconv[0].weight[:, :, :-1] - net.deconv[0].weight[:, :, 1:]) ** 2
                        ).mean(2)
                        / (std2(net.deconv[0].weight, axis=2, dim=4) ** 2 + 1e-10)
                    )
                    * (weights)[:, None]
                ).mean()

                l1act = (net.deconv[0].weight.abs() * (weights)[:, None, None]).mean()
                l1motif = net.conv[0].weight.abs().mean()
                loss = (
                    loss0
                    + l2 * 2e-3
                    + 1e-3
                    * (
                        PseudoPoissonKL(
                            pred[:, :, 325:-325] , target.cuda(2)[:, :, 325:-325]
                        )
                        * weights[None, :, None]
                    ).mean()
                    + l1act * 5e-5
                    + l1motif * 4e-5
                )
                loss.backward()
                past_losses.append(loss0.detach().cpu().numpy())
                past_l2.append(l2.detach().cpu().numpy())
                past_l1act.append(l1act.item())
                optimizer.step()
                
                i += 1
                
                if i % 100 == 0:
                    print("train loss:" + str(np.mean(past_losses[-100:])), flush=True)
                    train_losses_stage1.append(np.mean(past_losses[-100:]))
                    past_losses = []
                if i % 1000 == 0:
                    with torch.no_grad():
                        past_losses = []
                        past_cor = []
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
                            cor = np.corrcoef(pred[:, :, 325:-325].detach().cpu().reshape(-1,200).numpy(),
                                            target[:, :, 325:-325].detach().cpu().reshape(-1,200).numpy())[0][1]
                            
                            past_losses.append(loss0.detach().cpu().numpy())
                            past_cor.append(cor)
                    validcor = np.mean(past_cor)
                    validloss = np.mean(past_losses)
                    print(f"{epoch},{i},valid loss:" + str(validloss),"valid cor:" + str(validcor), flush=True)
                    valid_losses_stage1.append(validloss)
                    # scheduler.step(validloss)
                    if validloss < bestloss:
                        try:
                            os.remove(f'PI_models/test_{tissues}_stage_1_{epoch}_loss_{"{:.7f}".format(bestloss)}.best.pt')
                        except FileNotFoundError:
                            pass
                        torch.save(net,f'PI_models/test_{tissues}_stage_1_{epoch}_loss_{"{:.7f}".format(validloss)}.best.pt')
                        bestloss = validloss
        else:
            break       
            
