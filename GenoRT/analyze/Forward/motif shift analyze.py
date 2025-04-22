import pandas  as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import ast
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from matplotlib.cm import get_cmap
import matplotlib.colors as colors
import matplotlib.cm as cm



pred_path = r"PI_W82_merge/data"
pred_path = [ os.path.join(pred_path,i) for i in os.listdir(pred_path) if '._' not in i and i.startswith("model_for_predict_info_by_fft.fliter.csv",0)]
for idx,path in enumerate(pred_path):

    predict_file = pd.read_csv(path).sort_values(by=['chrom','pos','strand'])
    if idx==0:
       result = predict_file.iloc[:,5:-1].values[None,:,:]
    else:
        result = np.concatenate([result,predict_file.iloc[:,5:-1].values[None,:,:]],axis=0)
        print('a',result.shape,predict_file.iloc[:,5:-1].values.shape)

# print(result.shape)
all_predict_info = predict_file
all_predict_info.iloc[:,5:-1] = result.mean(axis=0)
# all_predict_info = pd.read_csv(r'PI_W82_merge/data/predcit_info_by_fft.csv')
all_predict_info['charge_motif'] = all_predict_info.iloc[:,5:-1].values.argmax(axis=1)
# all_predict_info = pd.read_csv(r'PI_W82_merge/data/predcit_info_by_fft.csv')
all_predict_info.index = all_predict_info['chrom'] + '-' + all_predict_info['pos'].map(str) + '-' + all_predict_info['strand']
#print(all_predict_info)
# columns_len = []
# for i in all_predict_info.iloc[:,5:-1].columns:
#     columns_len.append(sum(c.isalpha() for c in i)**0.5)

# columns_len = np.array(columns_len) / np.max(columns_len,keepdims=True)
# all_predict_info.iloc[:,5:-1] *= columns_len


#归一化
# all_predict_info.iloc[:,5:-1] = all_predict_info.iloc[:,5:-1].div(all_predict_info.iloc[:,5:-1].sum(axis=1),axis=0)
all_predict_info.iloc[:,5:-1] = all_predict_info.iloc[:,5:-1].abs().div(all_predict_info.iloc[:,5:-1].abs().max(axis=1),axis=0)
all_predict_info['charge_motif'] = all_predict_info.iloc[:,5:-1].values.argmax(axis=1)


non_var_TSS = pd.read_csv(r"PI_W82_merge/motif_shift_analyze/nonvar_TSS.bed",sep='\t',header=None)

non_var_TSS['distance'] = np.abs(non_var_TSS.iloc[:,5] - (non_var_TSS.iloc[:,1]+500))
non_var_TSS = non_var_TSS.sort_values(by='distance')
#print(non_var_TSS)
non_var_TSS = non_var_TSS.drop_duplicates(subset=[0,1,2])
non_var_TSS = non_var_TSS.drop_duplicates(subset=3)
non_var_TSS = non_var_TSS.drop_duplicates(subset=[4,5,6])
non_var_TSS['strand'] = non_var_TSS.iloc[:,3].str.split('(').str.get(1).str[0]
#print(non_var_TSS)
non_var_TSS['pi46_TSS'] = non_var_TSS.iloc[:,3].str.split('::').str.get(0) + '-' + non_var_TSS['strand']
non_var_TSS['w82_TSS'] = non_var_TSS.iloc[:,4] + '-' + (non_var_TSS.iloc[:,5]).map(str) + '-' + non_var_TSS['strand']
non_var_TSS = non_var_TSS[['pi46_TSS','w82_TSS']]

pi_tss = list(set(non_var_TSS['pi46_TSS']) & set(all_predict_info.index))
non_var_TSS = non_var_TSS[non_var_TSS['pi46_TSS'].isin(pi_tss)]
# #print(non_var_TSS)
w82_tss = list(set(non_var_TSS['w82_TSS']) & set(all_predict_info.index))
non_var_TSS = non_var_TSS[non_var_TSS['w82_TSS'].isin(w82_tss)]
print(non_var_TSS)
non_var_TSS = list(non_var_TSS['pi46_TSS'].values) + list(non_var_TSS['w82_TSS'].values)
non_var_TSS_info = all_predict_info.loc[non_var_TSS,:]['charge_motif']
#print(non_var_TSS)
# ##print(len(w82_tss))


var_TSS = pd.read_csv(r"PI_W82_merge/motif_shift_analyze/var_TSS.bed",sep='\t',header=None)

var_TSS['distance'] = np.abs(var_TSS.iloc[:,5] - ((var_TSS.iloc[:,2] + var_TSS.iloc[:,1])//2))
var_TSS = var_TSS.sort_values(by='distance')
var_TSS = var_TSS.drop_duplicates(subset=[0,1,2])
var_TSS = var_TSS.drop_duplicates(subset=3)
var_TSS = var_TSS.drop_duplicates(subset=[4,5,6])

# orign_var_TSS =var_TSS.copy()
var_TSS['strand'] = var_TSS.iloc[:,3].str.split('(').str.get(1).str[0]

var_TSS['pi46_TSS'] = var_TSS.iloc[:,3].str.split('::').str.get(0) + '-' + var_TSS['strand']
var_TSS['w82_TSS'] = var_TSS.iloc[:,4] + '-' + (var_TSS.iloc[:,5]).map(str) + '-' + var_TSS['strand']
orign_var_TSS =var_TSS.copy(deep=True)

# var_TSS = var_TSS[['pi46_TSS','w82_TSS']]

pi_tss = list(set(var_TSS['pi46_TSS']) & set(all_predict_info.index))
var_TSS = var_TSS[var_TSS['pi46_TSS'].isin(pi_tss)]
w82_tss = list(set(var_TSS['w82_TSS']) & set(all_predict_info.index))
var_TSS = var_TSS[var_TSS['w82_TSS'].isin(w82_tss)]
print(var_TSS)
#print(len(set(all_predict_info.index) & set(var_TSS['w82_TSS'])))
#print(orign_var_TSS)
# var_TSS_idx = var_TSS.index
selected_var_tss = orign_var_TSS[orign_var_TSS['pi46_TSS'].isin(var_TSS['pi46_TSS'])].iloc[:,:7]
selected_var_tss.to_csv(r'PI_W82_merge/motif_shift_analyze/selected_var_TSS.bed',sep='\t',header=None,index=None)
print(selected_var_tss)
print('_______________')
var_TSS = list(var_TSS['pi46_TSS'].values) + list(var_TSS['w82_TSS'].values)
var_TSS_info = all_predict_info.loc[var_TSS,:]['charge_motif']

##print(len(var_TSS))
##print(len(w82_tss))

var_TSS_info = pd.concat([var_TSS_info[:var_TSS_info.shape[0]//2].reset_index(drop=True),var_TSS_info[var_TSS_info.shape[0]//2:].reset_index(drop=True)],axis=1)
non_var_TSS_info = pd.concat([non_var_TSS_info[:non_var_TSS_info.shape[0]//2].reset_index(drop=True),non_var_TSS_info[non_var_TSS_info.shape[0]//2:].reset_index(drop=True)],axis=1)
var_TSS_info.columns = ['pi_motif','w82_motif']
non_var_TSS_info.columns = ['pi_motif','w82_motif']

# print(var_TSS_info)
# print(non_var_TSS_info)
var_change_dict = {}
non_var_change_dict = {}
all_var_change_dict = {}
for i,_ in enumerate(all_predict_info.columns[5:-1]):
    for j,_ in enumerate(all_predict_info.columns[5:-1]):
        var_change_dict[(i,j)] = 0
        non_var_change_dict[(i,j)] = 0
        all_var_change_dict[(i,j)] = 0

for pi_motif,w82_motif in zip(var_TSS_info['pi_motif'],var_TSS_info['w82_motif']):
    var_change_dict[(pi_motif,w82_motif)] += 1 / len(var_TSS_info)

for pi_motif,w82_motif in zip(non_var_TSS_info['pi_motif'],non_var_TSS_info['w82_motif']):
    non_var_change_dict[(pi_motif,w82_motif)] += 1 / len(non_var_TSS_info)

all_TSS_info = pd.concat([non_var_TSS_info,var_TSS_info],axis=0)
for pi_motif,w82_motif in zip(all_TSS_info['pi_motif'],all_TSS_info['w82_motif']):
    all_var_change_dict[(pi_motif,w82_motif)] += 1 / len(all_TSS_info)

var_change = np.array(list(var_change_dict.values()))
non_var_change = np.array(list(non_var_change_dict.values()))
all_var_change = np.array(list(all_var_change_dict.values()))
var_change /= var_change.sum()
non_var_change /= non_var_change.sum()
all_var_change /= all_var_change.sum()
##print(all_var_change.shape)
idx = np.argsort(-all_var_change)
all_var_change = all_var_change[idx]
var_change = var_change[idx]
non_var_change = non_var_change[idx]
from scipy.stats import wilcoxon
from scipy.stats import ttest_rel,ttest_ind
import scipy.stats as stats
# 设置全局散点样式
plt.rcParams['lines.markersize'] = 3   # 调整点的大小
plt.rcParams['lines.markerfacecolor'] = 'black'  # 调整点的颜色
plt.rcParams['lines.markeredgewidth'] = 0.5    # 点边框宽度
plt.rcParams['lines.markeredgecolor'] = 'black'  # 点边框颜色

# 示例数据
# np.random.seed(42)
# all_var_change = np.random.normal(loc=0, scale=1, size=100)

# 绘制 QQ 图
stats.probplot(all_var_change, dist="norm", plot=plt)
plt.grid(False)
# plt.show()
plt.savefig(r'PI_W82_merge/result/qqnorm_判断motif shfit频率是否符合正态分布.svg',format='svg',transparent=True,bbox_inches='tight',dpi='figure')
# plt.show()
plt.close()

t_stat, p_value = wilcoxon(var_change, all_var_change)
# t_stat, p_value = ttest_ind(var_change, all_var_change)
##print(all_var_change)
##print(p_value)

plt.subplots(figsize=(15,3))
x = np.array([i for i in range(all_var_change.shape[0])])
width = 0.4
plt.bar(x=(x - width/2)[:30],height=all_var_change[:30],alpha=0.8,width=width,color='#044080',label='All')
plt.bar(x=(x + width/2)[:30],height=var_change[:30],alpha=0.8,width=width,color='#93b7e3',label='Var')
# 获取当前坐标轴对象
ax = plt.gca()
plt.text(1, 0.05, f'-log(p) of Wilcoxon Signed-Rank Test :{round(-np.log(p_value),2)}', transform=ax.transAxes, fontsize=8, ha='right', va='top')

# 隐藏上边框和右边框
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.legend()
plt.xticks([])
plt.xlabel('Distribution of Motif Shift')
plt.ylabel('Proportion')
# plt.show()
plt.savefig(r'PI_W82_merge/result/符号秩检验判断变异与整体分布间的变异比例是否有显著差异.svg',format='svg',transparent=True,bbox_inches='tight',dpi='figure')
plt.close()

map_dict = { k:v for k,v in enumerate(all_predict_info.columns[5:-1])}

##print(map_dict,var_TSS_info)
for charge_motif in set(var_TSS_info.values.flatten()):
    if sum(var_TSS_info.values.flatten() == charge_motif) <100:
        var_TSS_info = var_TSS_info.replace(charge_motif,np.nan)

##print(var_TSS_info.isna().sum())
blast_pi_info = var_TSS_info['pi_motif']
blast_w82_info = var_TSS_info['w82_motif']
mask = blast_w82_info.values != blast_pi_info.values

# ##print(blast_w82_info)

import plotly.graph_objects as go
import plotly.colors as pc

source = []
target = []
value = []
change_dict = {}
for i,j in zip(blast_pi_info,blast_w82_info):
    
    if pd.Series([i,j]).isna().sum() != 0:
        continue
    i,j = int(i),int(j)
    if (f'PI_{i}',f'W82_{j}') not in change_dict.keys():
        change_dict[(f'PI_{i}',f'W82_{j}')] = 1
    else:
        change_dict[(f'PI_{i}',f'W82_{j}')] +=1

# ##print(change_dict)
for k,v in change_dict.items():
    # ##print(k,v)
    source.append(k[0])
    target.append(k[1])
    value.append(v)
label = [ i[0] for i in change_dict.keys() ] + [ i[1] for i in change_dict.keys() ]
label = sorted(list(set(label)), key=lambda x: (x.split('_')[0], float(x.split('_')[1])))
node_map = { name:idx for idx,name in enumerate(label)}
# ##print(node_map)
source = [  node_map[i] for i in source]
target = [  node_map[i] for i in target]

flow_dict = { i:0 for i in node_map.values()}
for s,t,v in zip(source,target,value):
    flow_dict[s] += v
    flow_dict[t] += v
# ##print(flow_dict)

change_only_flow = {i:0 for i in node_map.values()}
for s,t,v in zip(source,target,value):
    if s + len(label)//2 != t:
        change_only_flow[t] += v
# ##print(change_only_flow)

label = [ f'{i}_({(change_only_flow[idx] / flow_dict[idx]):.2f})' if idx>=len(label)/2 else i for idx,i in enumerate(label) ]

# 根据流量大小对节点进行排序
sorted_nodes = sorted(flow_dict.items(), key=lambda x: x[1], reverse=True)

# 重新排序节点标签
sorted_labels = [label[i] for i, _ in sorted_nodes]
sorted_node_indices = [i for i, _ in sorted_nodes]
sorted_source = [ sorted_node_indices.index(i) for i in source]
sorted_target = [ sorted_node_indices.index(i) for i in target]

def get_color(label):
    id = float(label.split('_')[1])
    cmap = pc.diverging.RdBu
    print(id,min(node_map.values()),max(node_map.values()))
    normalized_values = (id - 0) / (18 - 0)
    color = cmap[int(normalized_values * (len(cmap) - 1))]
    return color
color_list = [ get_color(i) for i in sorted_labels]

fig = go.Figure(go.Sankey(
    node=dict(
        pad=15,
        thickness=20,
        label=sorted_labels,
        color=color_list

    ),
    link=dict(
        source=sorted_source,
        target=sorted_target,
        value=value
    )
))
for c,label in zip(color_list,sorted_labels):
    if 'PI' in label:
        motif = map_dict[int(label.split('_')[1])]
        fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode="markers",
        marker=dict(color=c, size=10),
        name=motif  # 图例名称
            ))
fig.update_layout(
    plot_bgcolor='rgba(0,0,0,0)',   # 绘图区域背景透明
    xaxis=dict(showgrid=False, showticklabels=False),
    yaxis=dict(showgrid=False, showticklabels=False),
    width=650,  # 设置宽度
    height=800  # 设置高度
)

new_labels = [ label.split('_')[2][1:-1] if 'W82' in label else None for label in sorted_labels]
fig.data[0].node.label = new_labels

fig.show()
fig.write_image("PI_W82_merge/result/motif shift桑基图.svg")
# fig.show()

