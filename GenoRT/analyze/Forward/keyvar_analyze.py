import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import matplotlib as mpl

var_info = pd.read_csv(r'PI_W82_merge/data/var_info.csv')

var_predict_path = r"PI_W82_merge/data/predcit_var_final.csv"
var_predict = pd.read_csv(var_predict_path)
important_var = var_predict[var_predict['target_motif'] == var_predict['charge_motif']]
# 定义区间范围，例如 0-10, 10-20, ..., 90-100
bins = np.arange(-2000, 501, 50)  # 生成区间
# print(bins)
# 使用 pandas 的 cut 函数将数据分配到区间
labels = [f'[{x},{x+50})' for x in bins[:-1]]  # 区间标签，例如 '0-10', '10-20'
var_data_binned = pd.cut(important_var['var_pos'], bins=bins, labels=labels, right=False)  # right=False 表示区间右开
all_data_binned = pd.cut(var_predict['var_pos'], bins=bins, labels=labels, right=False)

all_data_dict = all_data_binned.value_counts()
var_data_dict = var_data_binned.value_counts()

print(all_data_dict)
print(var_data_dict)
var_data_dict = var_data_dict.sort_index()
all_data_dict = all_data_dict.sort_index()


var_data_dict = var_data_dict / all_data_dict
var_data_dict = (var_data_dict - var_data_dict.min()) / (var_data_dict.max() - var_data_dict.min())
# var_data_dict = var_data_dict / var_data_dict.max()
print(var_data_dict)
print(all_data_dict)
var_data_dict = var_data_dict.reset_index(drop=False)
var_data_dict.columns = ['Bin', 'Frequency']
var_data_dict['group'] = 'var_pos'
print(var_data_dict)
motif_pos = pd.read_csv(r"PI_W82_merge/data/motif_pos.csv")
motif_pos.columns = ['Bin', 'Frequency']
motif_pos['group'] = 'motif_pos'
print(motif_pos)

df = motif_pos.copy()
df['Frequency'] = motif_pos['Frequency'] * var_data_dict['Frequency']
df['group'] = 'motif x var'
print(df)
data = pd.concat([var_data_dict,motif_pos,df],axis=0)

cor = round(np.corrcoef(motif_pos['Frequency'],var_data_dict['Frequency'])[0,1],2)
print(cor)
print(data)
# print(motif_pos)
plt.figure(figsize=(25,8))

# 定义颜色映射函数
def get_color(group,):
    if group == 'motif x var':
        return mpl.cm.Reds(0.7)  # 红色系（0.7 控制颜色深浅）
    elif group =='var_pos':
        return mpl.cm.Blues(0.3)  # 蓝色系（0.7 控制颜色深浅）
    else:
        return mpl.cm.Greys(0.3)
# 创建颜色字典：将每个组映射到一个颜色
color_dict = {group: get_color(group) for group in data['group'].unique()}

# 使用 seaborn 绘制条形图
sns.barplot(x='Bin', y='Frequency', hue='group', data=data, palette=color_dict)
# 画出条形图
plt.xticks(rotation=45,fontsize=8)  
plt.ylabel('Position Score',fontsize=8)
plt.xlabel(None)
plt.savefig(r'PI_W82_merge/result/关键变异的位置与motif出现位置的分布情况.svg',format='svg',transparent=True,bbox_inches='tight',dpi='figure')
plt.savefig(r'PI_W82_merge/result/关键变异的位置与motif出现位置的分布情况.png',format='png',dpi=600)
plt.close()

plt.scatter(motif_pos['Frequency'],var_data_dict['Frequency'],s=0.7,c='black')
# 添加参考线
plt.plot([0, 1], [0, 1], color='black', linestyle='-', label='45-degree line',linewidth=0.7)
plt.text(0.03,0.95,f'correlation coefficient: {cor}',fontsize=8,transform=plt.gca().transAxes)
# 添加标题和标签
# plt.title('QQ Plot of Sample Data vs Normal Distribution')
plt.ylabel('Motif Pos Score')
plt.xlabel('Var Pos Score')
plt.legend()
# 显示图形
# plt.show()
plt.savefig(r'PI_W82_merge/result/关键变异的位置与motif出现位置的相关性分析.svg',format='svg',transparent=True,bbox_inches='tight',dpi='figure')
plt.savefig(r'PI_W82_merge/result/关键变异的位置与motif出现位置的相关性分析.png',format='png',dpi=600)