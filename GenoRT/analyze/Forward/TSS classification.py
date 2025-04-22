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
from matplotlib.cm import ScalarMappable

test_chrom = ["Gs10",'Gs08','GWHCAYC00000008','GWHCAYC00000010']
data_dir = r"PI_W82_merge/data"
motif_effect_path = [ os.path.join(data_dir,i) for i in os.listdir(data_dir) if '._' not in i and i.endswith('motif_effect.csv',0)]
print(motif_effect_path)

predict_info_path = [ os.path.join(data_dir,i) for i in os.listdir(data_dir) if '._' not in i and i.startswith('model_for_draw',0)]
print(predict_info_path)
for idx,path in enumerate(predict_info_path):
    file = pd.read_csv(path)
    value = file.iloc[:,6:].values[None,:,:]
    print(value.shape)
    info = file.iloc[:,:6]
    col = file.iloc[:,6:].columns
    if idx ==0:
        result = value
    else:
        result = np.concatenate([result,value],axis=0)
result =result.mean(axis=0)
result = pd.DataFrame(result,columns=col)
all_predict_info = pd.concat([info,result],axis=1)
print(all_predict_info)
all_predict_info = all_predict_info[all_predict_info['chrom'].isin(test_chrom)]
print(all_predict_info['cor'].mean())
type_value = ['PI46' if 'Gs' in i else 'W82' for i in all_predict_info['chrom']]
all_predict_info['Cultivar'] = type_value
for idx,path in enumerate(motif_effect_path):
    file = pd.read_csv(path)

    w82 = np.array([ast.literal_eval(i) for i in file['W82']])[None,:,:]
    pi46 = np.array([ast.literal_eval(i) for i in file['PI46']])[None,:,:]
    if idx ==0:
        motif_effect_w82 = w82
        motif_effect_pi46 = pi46
    else:
        motif_effect_w82 = np.concatenate([motif_effect_w82,w82],axis=0)
        motif_effect_pi46 = np.concatenate([motif_effect_pi46,pi46],axis=0)
motif_effect_w82 = motif_effect_w82.mean(axis=0)
motif_effect_pi46 = motif_effect_pi46.mean(axis=0)

#######绘制不同品种间模型预测误差的分布情况##########

cor_data = all_predict_info[all_predict_info['chrom'].isin(test_chrom)][['Cultivar','cor']]
print(cor_data)
cor_data = pd.melt(cor_data,id_vars='Cultivar', value_vars=['cor'])
all_cor= cor_data.copy()
all_cor['Cultivar'] = 'Total'
cor_data = pd.concat([all_cor,cor_data],axis=0)
print(cor_data)

custom_palette = {'PI46': '#FF7F0E', 'W82': '#1F77B4','Total':'#2CA02C'}
g = sns.kdeplot(cor_data,x='value',hue='Cultivar',fill=True,common_norm=False,palette=custom_palette)
g.set_xlabel('Pearson correlation coefficient',fontsize=14)
plt.savefig('PI_W82_merge/result/KDE for Cor.svg',format='svg',transparent=True,bbox_inches='tight',dpi='figure')
plt.close()

# ######与TSRmax的相关性##############
all_predict_info['TSR_max_level'] = np.log(all_predict_info['TSR_max_level'] + 1) / np.log(all_predict_info['TSR_max_level'] +1).max()
all_predict_info['cor'] = all_predict_info['cor'] / all_predict_info["cor"].max()
gx = sns.kdeplot(x=all_predict_info['TSR_max_level'],y=all_predict_info['cor'],fill=True, cmap="Blues", alpha=1,cbar=True)
gx.set_xlabel('TSR Max Level',fontsize=14)
gx.set_ylabel('Pearson correlation coefficient',fontsize=14)
gx.set_title('Merge Model Prediction')
plt.savefig('PI_W82_merge/result/KDE between Cor and TSR Max Level.svg',format='svg',transparent=True,bbox_inches='tight',dpi='figure')
# plt.show()
plt.close()


bins = [ i/10 for i in range(0,11,2)]
for idx,f_path in enumerate(predict_info_path):
    csv = pd.read_csv(f_path)
    csv = csv[ csv['chrom'].isin(test_chrom)]
    csv = csv[['TSR_max_level','TSR_sum_level','cor']]
    csv['TSR_sum_level'] = np.log(csv['TSR_sum_level'] + 1)
    csv['Cor_bins'] = pd.cut(csv['cor'], bins=bins)
    bin_list = set(csv['Cor_bins'])
    long_max_df = pd.melt(csv, id_vars='Cor_bins', value_vars=['TSR_sum_level'])
    if idx==0:
        df = long_max_df
    else:
        df = pd.concat([df,long_max_df],axis=0)
        print(df.shape)

base_color = "#1E90FF"
saturation_levels = [0.2,0.4,0.6,0.8,1]  # 定义不同的饱和度级别
palette = [sns.desaturate(base_color, s) for s in saturation_levels]

g = sns.kdeplot(df,x='value',hue='Cor_bins',common_norm=False,fill=True,palette=palette,bw_adjust=2.5)
g.set_xlabel('TSR Max Level',fontsize=14)
g.set_ylabel(g.get_ylabel(),fontsize=14)
plt.savefig('PI_W82_merge/result/new_KDE for Hue.svg',format='svg',transparent=True,bbox_inches='tight',dpi='figure')
plt.close()

#绘制motif效应曲线
motif_list = col[:-1]
print(motif_effect_pi46.shape)
motif_effect = np.stack([motif_effect_w82,motif_effect_pi46]).mean(axis=0)
print(motif_effect.shape)
sorted_indices = np.argsort(np.max(np.abs(motif_effect), axis=1))
motif_effect = motif_effect[sorted_indices]
motif_list = motif_list[sorted_indices]
fig,ax=plt.subplots(figsize=(20,5))

from scipy.signal import savgol_filter

eff_result = []
for i,(motif,effect) in enumerate(zip(motif_list,motif_effect)):
    smooth = savgol_filter(effect[-1000:], window_length=11, polyorder=2)
    eff_result.append(smooth)
eff_sort_index = np.argsort(np.mean(np.abs(motif_effect), axis=1))[::-1]
eff_result = np.array(eff_result)[eff_sort_index]
motif_list = np.array(motif_list)[eff_sort_index]
print(motif_list,eff_result.mean(axis=1))

know_motif = [
    '[C][G][T][G][G]',
    '[A][A][T][G][T][C]',
    '[A][T][G][G][C]',
    '[T][T][T][A][A][GT][A]',# HHO3-like
    '[A][A][AC].[C][T]',# AT4G36990 TBF1
    '[A].[T][A].[T].[AT][A]',# ATHB-23 
    '[G][G][G][C][C][C][A]',# TCP20
    '[C][A][C][G][T][G]',# HY5 MA0551.1
    '[T][A][T][A][A][A]',# TATA-box
    ]
for i,(motif,smooth) in enumerate(zip(motif_list,eff_result)):
    if smooth.mean() <0:
        print(motif)
        tab20 = get_cmap("Oranges")
    else:
        tab20 = get_cmap("Blues")
    print(motif)
    plt.plot(np.arange(-500,500),smooth,label=motif,color=tab20(i / len(motif_list)) if motif in know_motif else 'gray',linewidth=0.8)

plt.legend(bbox_to_anchor=(1.12, 1), fontsize=7)
plt.xlabel(rf'Relative to the position of TSS')
plt.ylabel(rf'Motif Effect')
# plt.show()
plt.savefig(r'PI_W82_merge/result/motif_效应曲线.svg',format='svg',dpi=800)
plt.close()

## 根据主效应motif对tss位点分类
# 获取基础 colormap 
base_cmap = plt.cm.Greens
n_colors = base_cmap.N  # 获取 colormap 的颜色数量
rgb_colors = base_cmap(np.linspace(0, 0.95, n_colors)**2)  # 获取 颜色梯度

# 添加透明度（alpha）渐变
alphas = np.linspace(0, 1, n_colors)   # 从 0（完全透明）到 1（完全不透明）
rgba_colors = np.zeros_like(rgb_colors)
rgba_colors[:, :3] = rgb_colors[:, :3]  # 保持原 RGB
rgba_colors[:, 3] = alphas**8 # 设置 alpha 通道

# 创建带透明度的 colormap
pred_path = r"PI_W82_merge/data"
pred_path = [ os.path.join(pred_path,i) for i in os.listdir(pred_path) if '._' not in i and i.startswith("model_for_predict_info_by_fft.fliter",0)]
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
all_predict_info.index = all_predict_info['chrom'] + '-' + all_predict_info['pos'].map(str) + '-' + all_predict_info['strand']
# print(all_predict_info)
custom_cmap = LinearSegmentedColormap.from_list("Blues_with_alpha", rgba_colors)
all_predict_info = all_predict_info.iloc[:,5:-1]
print(all_predict_info)
value_df = all_predict_info.values / all_predict_info.values.max(axis=1,keepdims=True)
value_df = pd.DataFrame(value_df,index=list(all_predict_info.index),columns=list(all_predict_info.columns))
column_counts = (value_df == 1).sum()
# 根据值为 1 的个数排序列
sorted_columns = column_counts.sort_values(ascending=False).index
sorted_data = value_df[sorted_columns]
print(sorted_data)
sorted_data['charge_motif'] =np.argmax(sorted_data,axis=1)
sorted_data = sorted_data.sort_values(by='charge_motif',ascending=True).iloc[:,:-1]
print(sorted_data)
plt.figure(figsize=(35, 15)) 
g = sns.heatmap(sorted_data.T,cmap=custom_cmap,xticklabels=False)
g.tick_params(axis='y', labelsize=15)
g.set_xlabel(None)
# 提取热力图的颜色映射和归一化对象
cmap = g.collections[0].cmap
norm = g.collections[0].norm

# 创建一个新的图像，仅包含颜色条
fig_cbar, ax_cbar = plt.subplots(figsize=(2, 5)) 

# 隐藏边框
for spine in ax_cbar.spines.values():
    spine.set_visible(False)

# 隐藏标题和刻度标签
ax_cbar.set_title('')
ax_cbar.set_xlabel('')
ax_cbar.set_ylabel('')
ax_cbar.set_xticks([])
ax_cbar.set_yticks([])

# 创建 ScalarMappable 对象
sm = ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # 必须设置为空数组

# 添加颜色条
cbar = fig_cbar.colorbar(sm, cax=ax_cbar)

# 设置颜色条的标题
cbar.set_label('Value', fontsize=12)

# 调整布局以确保标签显示完整
fig_cbar.tight_layout()

# 保存颜色条
plt.savefig('colorbar.svg', format='svg', dpi=800, bbox_inches='tight', transparent=False)
# plt.title('Motif Effect of TSS',fontsize=30)
# plt.savefig('PI_W82_merge/result/Motif Effect of TSS.png',format='png',transparent=False,bbox_inches='tight',dpi=600)
# plt.close()
plt.show()