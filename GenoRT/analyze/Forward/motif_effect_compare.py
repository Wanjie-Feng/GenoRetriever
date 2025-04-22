import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.patches import Patch
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

dir_path = "predict_result"
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
all_predict_info['charge_motif'] = all_predict_info.iloc[:,5:-1].values.argmax(axis=1)
all_predict_info = all_predict_info.iloc[:,:-1]
p_result = all_predict_info[all_predict_info['chrom'].str.contains('Gs')]
w_result = all_predict_info[~all_predict_info['chrom'].str.contains('Gs')]

p_result['idx'] = p_result['chrom'] + p_result['pos'].map(str) + p_result['strand']
p_result = p_result.drop(['TSR_max_level','TSR_sum_level','chrom','pos','strand'],axis=1)
p_result = p_result.groupby(by='idx').mean()
p_result_value = p_result.values.argmax(axis=1)
unique_elements, counts = np.unique(p_result_value, return_counts=True)
pi_count = {}
for k,v in zip(unique_elements,counts):
    key = p_result.columns[k]
    # print(key,v)
    print('pi46:',key,v)
    if v < 100:
        continue
    pi_count[key] = v / p_result.shape[0]
dropped_elements = list(set([i for i in range(len(p_result.columns))]) - set(list(unique_elements)))

for k in dropped_elements:
    key = p_result.columns[k]
    pi_count[key] = 0
print(pi_count)

w_result['idx'] = w_result['chrom'] + w_result['pos'].map(str) + w_result['strand']
w_result = w_result.drop(['TSR_max_level','TSR_sum_level','chrom','pos','strand'],axis=1)
w_result = w_result.groupby(by='idx').mean()

w_result_value = w_result.values.argmax(axis=1)
unique_elements, counts = np.unique(w_result_value, return_counts=True)
dropped_elements = list(set([i for i in range(len(w_result.columns))]) - set(list(unique_elements)))
print(len(unique_elements))
w82_count = {}
print(w_result)
for k,v in zip(unique_elements,counts):
    key = w_result.columns[k]


    #过滤绝对数量过少的位点，以减少假阳性带来的影响
    if v < 100:
        continue
    w82_count[key] = v / w_result.shape[0]
    print('w82:',key,v)
for k in dropped_elements:
    key = w_result.columns[k]
    print(key)
    w82_count[key] = 0

w82_count = pd.DataFrame.from_dict(w82_count,orient='index')
pi_count = pd.DataFrame.from_dict(pi_count,orient='index')
w82_count = w82_count.sort_values(by=0,ascending=False)
print(pi_count.index)
re_idx = [i for i in w82_count.index]
print(re_idx)
print(w82_count.index)
pi_count = pi_count.loc[re_idx]

# print(pi_count)
# print(w82_count)
print(pi_count)
print(w82_count)
# 统计TSS位点主效motif的分布情况
plt.figure(figsize=(15,10))
df = (w82_count - pi_count) / pi_count
print(df)
df = df.dropna()
df = df.sort_values(by=0,ascending=False).T
palette = {}
for x in df:
    if df[x].values[0]>0:
        palette[x] = '#008C8A'
    else:
        palette[x] = '#D6AD9C'
# colors = ['#008C8A' if df[x].values[0] > 0 else '#D6AD9C' for x in df]
print(df)
sns.barplot(df,orient='h',palette=palette)
# sns.barplot(-pi_count.T,orient='h',color='#D6AD9C')
plt.xlabel('Relative Proportion Change of Motif')
# 手动创建图例的标签和方块（Patch）对象
legend_elements = [
    Patch(facecolor='#008C8A', edgecolor='#008C8A', label='Increase'),
    Patch(facecolor='#D6AD9C', edgecolor='#D6AD9C', label='Decrease')
]
# 创建图例
plt.legend(handles=legend_elements,bbox_to_anchor=(1, 1))
plt.savefig('PI_W82_merge/result/Percentage of TSS Dominated by Motif (Square Root).svg',format='svg',transparent=True,bbox_inches='tight',dpi='figure')

plt.close()

# print(w_result)
w_result.iloc[:,:] = w_result.values / w_result.values.max(axis=1,keepdims=True)
p_result.iloc[:,:] = p_result.values / p_result.values.max(axis=1,keepdims=True)
w_result = w_result.mean(axis=0)
p_result = p_result.mean(axis=0)
print(w_result)
w_result = w_result.sort_values(ascending=False)
p_result = p_result.sort_values(ascending=False)

w_result = w_result/w_result.sum()
p_result = p_result/p_result.sum()

plt.figure(figsize=(15,10))

plt.figure(figsize=(15,10))  
bar1 = sns.barplot(w_result ,orient='h',color='#008C8A')
bar2 = sns.barplot(-p_result.T,orient='h',color='#D6AD9C')
def show_values(ax, space=0.01):
    for p in ax.patches[:5]:
        width = p.get_width()  # 获取条形的宽度
        ax.annotate(f'{width:.2f}',  # 格式化数值
                    (width - 0.001, p.get_y() + p.get_height() / 2),  # 设置文本的位置
                    ha='right', va='center',  # 水平左对齐，垂直居中对齐
                    xytext=(space, 0),  # 设置文本偏移量
                    textcoords='offset points',
                    color='white')  # 使用偏移量作为文本坐标
    for p in ax.patches[18:23]:
        width = p.get_width()  # 获取条形的宽度
        ax.annotate(f'{width:.2f}'[1:],  # 格式化数值
                    (width + 0.001, p.get_y() + p.get_height() / 2),  # 设置文本的位置
                    ha='left', va='center',  # 水平左对齐，垂直居中对齐
                    xytext=(space, 0),  # 设置文本偏移量
                    textcoords='offset points')  # 使用偏移量作为文本坐标
show_values(bar2)

plt.xlabel('Percentage of Motif Effect (Exp)')
# 手动创建图例的标签和方块（Patch）对象
legend_elements = [
    Patch(facecolor='#008C8A', edgecolor='#008C8A', label='W82'),
    Patch(facecolor='#D6AD9C', edgecolor='#D6AD9C', label='PI468916')
]

# 创建图例
plt.legend(handles=legend_elements,bbox_to_anchor=(1, 1))
plt.savefig('PI_W82_merge/result/Percentage of Motif Effect (Exp).svg',format='svg',transparent=True,bbox_inches='tight',dpi='figure')

# plt.show()
plt.close()
