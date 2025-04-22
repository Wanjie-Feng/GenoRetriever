#***********************************绘制热图查看YY1效应模式*********************************#
import pyBigWig
import selene_sdk 
import tabix
from selene_sdk.targets import Target
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import fftconvolve
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
    
motif = 'YY1'

exp_bg_bw = GenomicSignalFeatures(input_paths=[
    rf"KO_YY1/930-3/Wm82-NJAU.930-3.plus.filter.bw",
    rf"KO_YY1/930-3/Wm82-NJAU.930-3.minus.filter.bw",
]
                                 
 ,features=[f'bg_root.plus',f'bg_root.minus'],shape=(4000,))

exp_ko_bw = GenomicSignalFeatures(input_paths=[
    rf"KO_YY1/YY1-12/Wm82-NJAU.YY1-12.plus.filter.bw",
    rf"KO_YY1/YY1-12/Wm82-NJAU.YY1-12.minus.filter.bw",
]
                                 
 ,features=[f'ko_root.plus',f'ko_root.minus'],shape=(4000,))


tss_dict = {
    'tss_info':[],
    'change_exp':[],
    'distance':[]
}
background_result = pd.read_csv(rf"KO_YY1/net_predict/KO_YY1_sorted.csv")
background_result.index = background_result['chrom'] + '_' + background_result['pos'].map(str) + '_' + background_result['strand']
for tss_idx in range(background_result.shape[0]):
    chrom,pos,strand = background_result.index[tss_idx].split('_')
    pos = int(pos)

    if strand == '-':
        bg_tar = exp_bg_bw.get_feature_data(
            chrom,
            pos - 500 ,
            pos + 3500,
        )
        ko_tar = exp_ko_bw.get_feature_data(
            chrom,
            pos - 500 ,
            pos + 3500,
        )
        # continue
        bg_tar = bg_tar[::-1, ::-1]
        ko_tar = ko_tar[::-1, ::-1]
    else:
        bg_tar = exp_bg_bw.get_feature_data(
            chrom,
            pos - 3500 ,
            pos + 500,
        )
        ko_tar = exp_ko_bw.get_feature_data(
            chrom,
            pos - 3500 ,
            pos + 500,
        )     

    bg_tar = bg_tar[0]
    ko_tar = ko_tar[0]
    tss_pos = np.argmax(bg_tar,axis=0)
    tss_pos = np.clip(tss_pos,a_min=500,a_max=3500)
    bg_tar = bg_tar[tss_pos-500:tss_pos+500]
    ko_tar = ko_tar[tss_pos-500:tss_pos+500]

    print(chrom,pos,strand,(ko_tar-bg_tar).max())
    tss_dict['tss_info'].append(background_result.index[tss_idx])
    tss_dict['change_exp'].append(ko_tar-bg_tar)
    tss_dict['distance'].append(np.argmax(ko_tar) - np.argmax(bg_tar))


tss_dict = np.vstack(pd.DataFrame(tss_dict)['change_exp'].values)
tss_dict = fftconvolve(tss_dict,np.ones((5,3))/15,mode='valid')
max_values = np.abs(tss_dict).max(axis=1, keepdims=True)
non_zero_mask = max_values != 0

# 只有当最大值不为0时才进行除法
tss_dict = np.where(non_zero_mask, tss_dict / max_values, 0)


from scipy.signal import fftconvolve

print(tss_dict)
plt.figure(figsize=(7,10),dpi=600)
print(tss_dict.shape)
g = sns.heatmap(tss_dict,cmap='RdBu_r',center=0,yticklabels=False)
g.set_ylabel(rf'{tss_dict.shape[0]} Promoters')
g.set_title(f'KD-{motif} - Control')
g.set_xticks(ticks=[0,500,990],labels=['-500','0','500'])
plt.savefig(f'KD-{motif} - Control.png',format='png',dpi=600)
