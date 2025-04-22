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
import torch.utils.data.dataloader as DataLoader
import torch.utils.data.dataset as Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR
import glob
import re
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.nonparametric.api as smnp

map_dict = str.maketrans({
    'A':'T',
    'C':'G',
    'G':'C',
    'T':'A'
} )
motifs = np.load(rf"analyze/result/sorted_cl_motif.npy")
other_motifs = np.load(rf"/Data5/pfGao/xtwang/TSS/tss/analyze/result/pi_w82_other.npy")
genomic_path_list = [
    r'/Data5/pfGao/xtwang/TSS/tss/brassica_napus/stripe_result/ZS11/Brassica_napus.ZS11.v0.genome.fa', # 油菜
    r'/Data5/pfGao/xtwang/TSS/tss/cotton/stripe_result/TM-1/TM-1_HAU_v2.fa', # 棉花
    r'/Data5/pfGao/xtwang/TSS/tss/maize/stripe_result/B73/Zm-B73-REFERENCE-NAM-5.0.fa', # 玉米
    r'/Data5/pfGao/xtwang/TSS/tss/rice/stripe_result/ZS97/ZS97RS3.rename.fa', # 水稻ZS97
    r'/Data5/pfGao/xtwang/TSS/tss/tomato/stripe_result/LA1589/LA1589.genome.fa', # 番茄
    r'/Data5/pfGao/xtwang/TSS/tss/wheat/stripe_result/Svevo/genomic_1.fa', # 小麦
    r"/Data5/pfGao/xtwang/TSS/W82-NJAU/genome/Wm82-NJAU.fasta", # 大豆W82
]
TSR_path_list = [
    r"/Data5/pfGao/xtwang/TSS/tss/brassica_napus/stripe_result/ZS11/ZS11.Unidirection_result.txt", # 油菜
    r'/Data5/pfGao/xtwang/TSS/tss/cotton/stripe_result/TM-1/TM-1.Unidirection_result.txt', # 棉花
    r'/Data5/pfGao/xtwang/TSS/tss/maize/stripe_result/B73/B73.Unidirection_result.txt', # 玉米
    r'/Data5/pfGao/xtwang/TSS/tss/rice/stripe_result/ZS97/ZS97.Unidirection_result.txt', # 水稻ZS97
    r"/Data5/pfGao/xtwang/TSS/tss/tomato/stripe_result/LA1589/LA1589.Unidirection_result.txt", # 番茄
    r'/Data5/pfGao/xtwang/TSS/tss/wheat/stripe_result/Svevo/Svevo.Unidirection_result.txt', # 小麦
    r'/Data5/pfGao/xtwang/TSS/tss/data_process/W82.Unidirection_result.txt', # 大豆W82
]

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
print(motif_list)

result_dict = {}
plant_list = [
'Brassica_napus',
'Cotton',
'Maize',
'Rice',
'Tomato',
'Wheat',
'Soybean'
]
single = ['Maize','Rice','Wheat']

GC_dict = {

}
for plant,TSR_path,Genome_path in zip(plant_list,TSR_path_list,genomic_path_list):
        result_dict[plant] = {}
        plant_dict ={}
        GC_dict[plant] = []
        pos_count = 0
        TSR_file = pd.read_csv(TSR_path, sep="\t").iloc[:500,:]
        genome = selene_sdk.sequences.Genome(input_path=Genome_path)
        for randi in range(TSR_file.shape[0]):
            seqnamesm, pos, strand = (
                TSR_file["seqnames"].values[randi],
                TSR_file["thick.start"].values[randi],
                TSR_file["strand"].values[randi],
            )
            
            try:
                if strand == '-':
                    seq = genome.get_sequence_from_coords(
                        TSR_file["seqnames"].values[randi],
                        TSR_file["thick.start"].values[randi] - 500,
                        TSR_file["thick.start"].values[randi] + 3500,
                        TSR_file["strand"].values[randi],
                    )
                    seq = seq.translate(map_dict)
                else:
                    seq = genome.get_sequence_from_coords(
                        TSR_file["seqnames"].values[randi],
                        TSR_file["thick.start"].values[randi] - 3500,
                        TSR_file["thick.start"].values[randi] + 500,
                        TSR_file["strand"].values[randi],
                    )

            except:
                continue
            if len(seq) !=4000 or 'N' in seq:
                print(seqnamesm,strand,pos)
                continue
            else:
                pos_count += 1
            for motif in motif_list:
                motif_iter = re.finditer(motif, seq)
                if plant_dict.get(motif) is None:
                    plant_dict[motif] = 0
                else:
                    plant_dict[motif] += len(list(motif_iter))
            GC_ratio = (seq.count('G')+seq.count('C')) / len(seq)
            GC_dict[plant].append(GC_ratio)
            print(GC_ratio)
        for key, value in plant_dict.items():
            plant_dict[key] = value / pos_count
        result_dict[plant] = plant_dict
        GC_dict[plant]= np.mean(GC_dict[plant])

print(GC_dict)
avg_effect_dataframe = pd.read_csv("analyze/new_model_result/motif_avg_effect_in_diff_plant.csv")
avg_effect_dataframe = avg_effect_dataframe.set_index('plant')
avg_effect_dataframe.to_csv(r'analyze/new_model_result/Motif_Effect_Change.csv')
frequnce_result_dataframe = pd.DataFrame(result_dict).T
frequnce_result_dataframe.to_csv(r'analyze/new_model_result/Motif_Frequence_Change.csv')
print(frequnce_result_dataframe)
print(avg_effect_dataframe)
frequnce_result_dataframe = frequnce_result_dataframe / frequnce_result_dataframe.sum(axis=1).values.reshape(-1,1)
frequnce_result_dataframe = frequnce_result_dataframe / frequnce_result_dataframe.max(axis=0).values
# avg_effect_dataframe = avg_effect_dataframe / avg_effect_dataframe.sum(axis=1).values.reshape(-1,1)
avg_effect_dataframe = avg_effect_dataframe / avg_effect_dataframe.max(axis=1).values.reshape(-1,1)
print(avg_effect_dataframe)
mean_effect = avg_effect_dataframe.mean(axis=0).values
frequnce_result_dataframe = frequnce_result_dataframe.iloc[:,np.argsort(mean_effect)[::-1]].T
avg_effect_dataframe = avg_effect_dataframe.iloc[:,np.argsort(mean_effect)[::-1]].T
print(mean_effect.shape)
print(frequnce_result_dataframe)


plt.figure(figsize=(14,7))
# frequnce_result_dataframe = frequnce_result_dataframe.loc[avg_effect_dataframe.index]
cg = sns.clustermap(frequnce_result_dataframe,row_cluster=False,col_cluster=True,cmap="RdBu_r", vmin=0, vmax=1,method='average')
heatmap_ax = cg.ax_heatmap
heatmap_bbox = heatmap_ax.get_position()
x0 = heatmap_bbox.x0
y0 = heatmap_bbox.y0
width = heatmap_bbox.width
height = heatmap_bbox.height
col_reordered_ind = cg.dendrogram_col.reordered_ind
print(col_reordered_ind)
change = frequnce_result_dataframe.iloc[:,col_reordered_ind]
# print(change)
# change.to_csv(r'analyze/new_model_result/Motif_Frequence_Change.csv')

change = change.iloc[:,3:].mean(axis=1) - change.iloc[:,:3].mean(axis=1)

print(change)
plt.setp(cg.ax_heatmap.get_xticklabels(), rotation=45, horizontalalignment='right')
fig = cg._figure
ax_line = fig.add_axes([x0-width/10,y0,width/10,height])
ax_line.plot(-change[::-1],np.arange(0,len(change),1),c='gray',linewidth=1,linestyle=(0, (8, 3)), label='Custom Dash',marker='D',markersize=5)
ax_line.set_yticks([])
ax_line.set_xticks([0])
ax_line.xlim = (-0.5,0.5)
ax_line.spines['right'].set_visible(False)
ax_line.spines['top'].set_visible(False)
ax_line.axvline(0, color='gray', linestyle=(0, (3, 5)), label='Custom Dash', linewidth=1)
ax_line.set_xlabel('Cluster2 - Cluster1',fontsize=5,labelpad=5)
# plt.tight_layout()
plt.savefig('analyze/new_model_result/frequence.svg', format='svg',dpi=800)
plt.close()

frequnce_change = (change - change.min()) / (change.max()-change.min())



cg = sns.clustermap(avg_effect_dataframe,row_cluster=False,col_cluster=True,cmap="RdBu_r", vmin=0, vmax=1,method='average')
heatmap_ax = cg.ax_heatmap
heatmap_bbox = heatmap_ax.get_position()
x0 = heatmap_bbox.x0
y0 = heatmap_bbox.y0
width = heatmap_bbox.width
height = heatmap_bbox.height
col_reordered_ind = cg.dendrogram_col.reordered_ind
print(col_reordered_ind)
change = avg_effect_dataframe.iloc[:,col_reordered_ind]
# print(change)
change = change.iloc[:,3:].mean(axis=1) - change.iloc[:,:3].mean(axis=1)

print(change)
plt.setp(cg.ax_heatmap.get_xticklabels(), rotation=45, horizontalalignment='right')
fig = cg._figure
ax_line = fig.add_axes([x0-width/10,y0,width/10,height])
ax_line.plot(-change[::-1],np.arange(0,len(change),1),c='gray',linewidth=1,linestyle=(0, (8, 3)), label='Custom Dash',marker='D',markersize=5)
ax_line.set_yticks([])
ax_line.set_xticks([0])
ax_line.xlim = (-0.5,0.5)
ax_line.spines['left'].set_visible(False)
ax_line.spines['top'].set_visible(False)
ax_line.axvline(0, color='gray', linestyle=(0, (3, 5)), label='Custom Dash', linewidth=1)
ax_line.set_xlabel('Cluster2 - Cluster1',fontsize=5,labelpad=5)
# plt.tight_layout()
plt.savefig('analyze/new_model_result/effect.svg', format='svg',dpi=800)
plt.close()

effect_change = (change - change.min()) / (change.max() - change.min())

cor = np.corrcoef(frequnce_change,effect_change)[0,1]
t = np.linspace(0, 1, 100)  # 参数 t 的范围从0到1
x = t
y = t
plt.figure(figsize=(5,5))
plt.scatter(frequnce_change,effect_change,s=5,c='black')
plt.plot(x,y,color='black',linewidth=0.5)
plt.xlabel('Frequency change')
plt.ylabel('Effect change')
plt.legend([f'Correlation: {cor:.2f}'])
plt.savefig('analyze/new_model_result/frequence_effect_correlation.svg', format='svg',dpi=800)
print(cor)
plt.close()



#位置效应
motif_posision_dict = {}
motif_posision_dict_forheat = {}

for plant,TSR_path,Genome_path in zip(plant_list,TSR_path_list,genomic_path_list):
        r_dict = {}
        r_heat_dict = {}
        plant_dict ={}
        GC_dict[plant] = []
        pos_count = 0
        TSR_file = pd.read_csv(TSR_path, sep="\t")
        genome = selene_sdk.sequences.Genome(input_path=Genome_path)
        for randi in range(TSR_file.shape[0]):
            seqnamesm, pos, strand = (
                TSR_file["seqnames"].values[randi],
                TSR_file["thick.start"].values[randi],
                TSR_file["strand"].values[randi],
            )
            
            try:
                if strand == '-':
                    seq = genome.get_sequence_from_coords(
                        TSR_file["seqnames"].values[randi],
                        TSR_file["thick.start"].values[randi] - 500,
                        TSR_file["thick.start"].values[randi] + 3500,
                        TSR_file["strand"].values[randi],
                    )
                    seq = seq.translate(map_dict)
                else:
                    seq = genome.get_sequence_from_coords(
                        TSR_file["seqnames"].values[randi],
                        TSR_file["thick.start"].values[randi] - 3500,
                        TSR_file["thick.start"].values[randi] + 500,
                        TSR_file["strand"].values[randi],
                    )

            except:
                continue
            if len(seq) !=4000 or 'N' in seq:
                print(seqnamesm,strand,pos)
                continue
            else:
                pos_count += 1
            for motif in motif_list:
                if r_dict.get(motif) is None:
                    r_dict[motif] = []
                    r_heat_dict[motif] = []
                else:
                    motif_posision = [ np.abs(i.start() - 3500) for i in re.finditer(motif, seq)]
                    motif_posision = [ i for i in motif_posision if np.abs(i) < 2000]
                    motif_posision_heat = [ np.abs(i.start() - 3500) for i in re.finditer(motif, seq)]
                    motif_posision_heat = [ i for i in motif_posision_heat if np.abs(i) < 2000]
                    r_dict[motif] += motif_posision
                    r_heat_dict[motif] += motif_posision_heat
        motif_posision_dict[plant] = r_dict
        motif_posision_dict_forheat[plant] = r_heat_dict
        # break

fig,ax = plt.subplots(ncols=2,nrows=int(len(motif_list)/2),figsize=(10,20))
for idx,motif in enumerate(motif_list):
    single_list = []
    double_list = []
    for plant in plant_list:
        if plant in single:
            single_list+=motif_posision_dict[plant][motif]
        else:
            double_list+=motif_posision_dict[plant][motif]
    kde = smnp.KDEUnivariate(np.array(single_list))
    kde.fit(kernel="gau", bw="normal_reference", fft=True, gridsize=1000, adjust=0.1, cut=3, clip=(-np.inf, np.inf))  # 使用默认参数
    kde_x = kde.support
    kde_y = kde.density
    single_xmax = kde_x[np.argmax(kde_y)] # 获取单子叶最大峰值对应横坐标
    single_ymax = np.max(kde_y)

    ax[int(idx/2),int(idx%2)].plot(kde_x[50:-50],kde_y[50:-50],label='single',linewidth=0.5)

    kde = smnp.KDEUnivariate(np.array(double_list))
    kde.fit(kernel="gau", bw="normal_reference", fft=True, gridsize=1000, adjust=0.1, cut=3, clip=(-np.inf, np.inf))  # 使用默认参数
    kde_x = kde.support
    kde_y = kde.density
    double_xmax = kde_x[np.argmax(kde_y)] # 获取双子叶最大峰值对应横坐标
    double_ymax = np.max(kde_y)
    ax[int(idx/2),int(idx%2)].plot(kde_x[50:-50],kde_y[50:-50],label='double',linewidth=1)
    ax[int(idx/2),int(idx%2)].set_title(motif)
    ax[int(idx/2),int(idx%2)].spines['right'].set_visible(False)
    ax[int(idx/2),int(idx%2)].spines['top'].set_visible(False)

    x1x2 = [double_xmax,double_ymax]
    x1x2.sort()
    y1y2 = [max(double_ymax,single_ymax) *1.02] * len(x1x2)
    # ax[int(idx/2),int(idx%2)].plot(x1x2,y1y2,color='black')
ax[0,0].legend()
plt.tight_layout()
plt.savefig(r"analyze/new_model_result/single_double_motif_position_adjust5.svg",format='svg',dpi=800)
#plt.show()
plt.close()


result_dict = {}
plot_dict = {}

for plant in motif_posision_dict.keys():
    motif_dict = {}
    motif_dict_2 = {}
    for k,v in motif_posision_dict[plant].items():
        # 计算 KDE 曲线的 x 和 y 值
        print(v)
        kde = smnp.KDEUnivariate(v)
        kde.fit(kernel="gau", bw="normal_reference", fft=True, gridsize=1000, adjust=5, cut=3, clip=(-np.inf, np.inf))  # 使用默认参数
        kde_x = kde.support
        kde_y = kde.density
        # print(len(kde_x),len(kde_y))
        # 找到频率最高的 x 坐标
        max_density_idx = np.argmax(kde_y)
        max_x = kde_x[max_density_idx]
        motif_dict[k] = max_x
        motif_dict_2[k] =(kde_x,kde_y)
        # print(kde_x)
    result_dict[plant] = motif_dict
    plot_dict[plant] = motif_dict_2
result_df = pd.DataFrame(result_dict).T
result_df.to_csv(r'analyze/new_model_result/Motif_Position_Change.csv')

result_df = (result_df - result_df.min(axis=0)) / (result_df.max(axis=0) - result_df.min(axis=0))
result_df = result_df / result_df.max(axis=0)
result_df = result_df.iloc[:,np.argsort(mean_effect)[::-1]].T
print(result_df)
cg = sns.clustermap(result_df,row_cluster=False,col_cluster=True,cmap="RdBu_r", vmin=0, vmax=1,method="average",metric="euclidean")
heatmap_ax = cg.ax_heatmap
heatmap_bbox = heatmap_ax.get_position()
x0 = heatmap_bbox.x0
y0 = heatmap_bbox.y0
width = heatmap_bbox.width
height = heatmap_bbox.height
col_reordered_ind = cg.dendrogram_col.reordered_ind
print(col_reordered_ind)
change = result_df.iloc[:,col_reordered_ind]
# print(change)
# change.to_csv(r'analyze/new_model_result/Motif_Position_Change.csv')

change = change.iloc[:,3:].mean(axis=1) - change.iloc[:,:3].mean(axis=1)

print(change)
plt.setp(cg.ax_heatmap.get_xticklabels(), rotation=45, horizontalalignment='right')
fig = cg._figure
ax_line = fig.add_axes([x0-width/10,y0,width/10,height])
ax_line.plot(-change[::-1],np.arange(0,len(change),1),c='gray',linewidth=1,linestyle=(0, (8, 3)), label='Custom Dash',marker='D',markersize=5)
ax_line.set_yticks([])
ax_line.set_xticks([0])
ax_line.xlim = (-0.5,0.5)
ax_line.spines['left'].set_visible(False)
ax_line.spines['top'].set_visible(False)
ax_line.axvline(0, color='gray', linestyle=(0, (3, 5)), label='Custom Dash', linewidth=1)
ax_line.set_xlabel('Cluster2 - Cluster1',fontsize=5,labelpad=5)
# plt.tight_layout()
plt.savefig('analyze/new_model_result/motif_position_heatmap.svg', format='svg',dpi=800)
plt.close()
pos_change = (change - change.min()) / (change.max()-change.min())
cor = np.corrcoef(effect_change,pos_change)[0,1]


t = np.linspace(0, 1, 100)  # 参数 t 的范围从0到1
x = t
y = t
plt.figure(figsize=(5,5))
plt.scatter(pos_change,effect_change,s=5,c='black')
plt.plot(x,y,color='black',linewidth=0.5)
plt.xlabel('Posision change')
plt.ylabel('Effect change')
plt.legend([f'Correlation: {cor:.2f}'])
plt.savefig('analyze/new_model_result/posistion_effect_correlation.svg', format='svg',dpi=800)
print(cor)
plt.close()
# fig,ax = plt.subplots(ncols=2,nrows=int(len(motif_list)/2),figsize=(10,20))
# for idx,motif in enumerate(motif_list):
#     for plant in plot_dict.keys():
#         data = plot_dict[plant][motif]
#         ax[int(idx/2),int(idx%2)].plot(data[0][50:-50],data[1][50:-50],label=plant,linewidth=0.5)
#         ax[int(idx/2),int(idx%2)].spines['right'].set_visible(False)
#         ax[int(idx/2),int(idx%2)].spines['top'].set_visible(False)
#     ax[int(idx/2),int(idx%2)].set_title(motif)
# ax[0,1].legend()
# plt.tight_layout()
# plt.savefig('analyze/new_model_result/motif_position.svg', format='svg',dpi=800)
# #plt.show()
# plt.close()




data = np.column_stack((pos_change,effect_change,frequnce_change))

import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# 构建特征矩阵
X = np.column_stack((frequnce_change, pos_change))
X = sm.add_constant(X)  # 添加常数项（截距）
# 添加常数项
X = sm.add_constant(X)
# scaler = StandardScaler()
# X = scaler.fit_transform(X)
print(X.sum())
print(effect_change.sum())
# y_reg = np.concatenate((effect_change, np.zeros(X_reg.shape[1] - 1)))
# 拟合多元线性回归模型
# model = sm.OLS(effect_change, X)
df = pd.DataFrame({
    'y': effect_change,
    'X1': frequnce_change,
    'X2': pos_change
})
model = smf.ols('y ~ X1 + X2 + X1:X2', data=df).fit()
print(model.summary())

motif_effect_change = pd.concat([effect_change,pos_change,frequnce_change],axis=1)
print(motif_effect_change.shape)
motif_effect_change = pd.DataFrame(motif_effect_change)
motif_effect_change.columns = ['Effect Change','Position Change','Frequence Change']
motif_effect_change.to_csv(r"analyze/new_model_result/motif_effect_change.csv")