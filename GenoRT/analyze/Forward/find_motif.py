import pandas as pd
import selene_sdk 
import os
import re
import numpy as np
from reset import reset
tss_file = pd.read_csv(r"PI_W82_merge/data/Unidirection_result.txt",sep='\t')
tss_file['tss_pos'] = tss_file['seqnames'] + '_' + tss_file['thick.start'].map(str) + '_' + tss_file['strand']
print(tss_file)
genome = selene_sdk.sequences.Genome(
    input_path=r"W82_t2t/Wm82-NJAU.fasta",
)
motifs = np.load(rf"PI_W82_merge/data/sorted_cl_motif.npy")

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

map_dict = str.maketrans({
    'A':'T',
    'C':'G',
    'G':'C',
    'T':'A'
} )


n_tss = tss_file.shape[0]
pos_list = []
for motif in motif_list:
    for tss_id in range(n_tss):
        chr,start,strand = tss_file.loc[tss_id,'tss_pos'].split('_')
        # print(chr,start,strand)
        if strand == '-':
            seq = genome.get_sequence_from_coords(
                chr,
                int(start) - 500,
                int(start) + 2000,
                strand,
            )
            seq = seq.translate(map_dict)

            matches = list(re.finditer(motif, seq))
            positions = [ 500 - match.start() for match in matches]
            # print(positions)
        else:
            seq = genome.get_sequence_from_coords(
                chr,
                int(start) - 2000,
                int(start) + 500,
                strand,
            )

            matches = list(re.finditer(motif, seq))
            positions = [ match.start() - 2000 for match in matches]
        if len(positions) != 0:
            positions = positions[np.abs(positions).argmin()]

            pos_list.append(positions)
pos_list = np.array(pos_list)
pos_list = pd.DataFrame(pos_list)
# print(pos_list)
bins = np.arange(-2000, 501, 50)  # 生成区间
labels = [f'[{x},{x+50})' for x in bins[:-1]]
pos_list = pd.cut(pos_list[0], bins=bins, labels=labels, right=False)  # right=False 表示区间右开
pos_list = pos_list.value_counts().sort_index().reset_index(drop=False)
pos_list.columns = ['Bins','Frequency']
print(pos_list)
pos_list['Frequency'] = (pos_list['Frequency'] - pos_list['Frequency'].min()) / (pos_list['Frequency'].max() - pos_list['Frequency'].min())
# pos_list['Frequency'] = pos_list['Frequency'] / pos_list['Frequency'].sum()

print(pos_list)
pos_list.to_csv(r"PI_W82_merge/data/motif_pos.csv",index=None)
# pos_list
# pos_list = pos_list.groupby(by='bin').mean()
# pos_list = (pos_list - pos_list.min()) / (pos_list.max() - pos_list.min())
# print(pos_list)
# import matplotlib.pyplot as plt
# import seaborn as sns
# # # 使用seaborn的kdeplot函数绘制密度估计图
# ax = sns.kdeplot(pos_list, bw_adjust=1)

# # # 获取线对象
# lines = ax.get_lines()

# # 假设密度图只有一条线，获取第一条线的x和y坐标
# x = lines[0].get_xdata()
# y = lines[0].get_ydata()

# # 找到y坐标中的最大值及其对应的x坐标
# max_density_index = np.argmax(y)
# peak_x_coordinate = x[max_density_index]

# # 打印最高峰的x坐标
# print(f"The x-coordinate of the peak is: {peak_x_coordinate}")
# plt.show()
# n, bins, patches = plt.hist(pos_list, bins=1000, alpha=0.7, color='blue', edgecolor='black',density=True)

# # 找到频率最高的区间
# max_index = n.argmax()  # 获取最大频率的索引
# max_freq = n[max_index]  # 最大频率的值
# max_bin_left = bins[max_index]  # 最高频率区间的左边界
# max_bin_right = bins[max_index + 1]  # 最高频率区间的右边界

# # 打印结果
# print(f"频率最多的区间是：{max_bin_left} 到 {max_bin_right}")
# print(f"该区间的频率是：{max_freq}")
# plt.show()