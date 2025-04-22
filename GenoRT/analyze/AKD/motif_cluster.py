import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import logomaker
import networkx as nx
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D



tissue_list = ['Flower','Leaf','Nodule','Pod','Root','Seed','Shoot','Stemtip','merge']
# tissue_list = ['Flower','Leaf','Nodule','Pod']
stage = '1'
def cross_corr(x, y):
    cors = []
    i=0
    if stage=='1':
        for j in range(y.shape[1]-5):
            cors.append(np.fmax(np.corrcoef(x.flatten(), 
                                    np.concatenate((y[:,j:],y[:,:j]), axis=1).flatten())[0,1],
                            np.corrcoef(x.flatten(), 
                                    np.concatenate((y[:,j:],y[:,:j]), axis=1)[::-1,::-1].flatten())[0,1]))
    else:
        for j in range(y.shape[1]):
            cors.append(np.fmax(np.corrcoef(x.flatten(), 
                                    np.concatenate((y[:,j:],y[:,:j]), axis=1).flatten())[0,1],
                            np.corrcoef(x.flatten(), 
                                    np.concatenate((y[:,j:],y[:,:j]), axis=1)[::-1,::-1].flatten())[0,1]))
    return np.array(cors).max()

def corr_compute(array1,array2,t1,t2,if_bidirectional=True):
    cor_dict = {}
    for idx1,d1 in enumerate(array1):
        for idx2,d2 in enumerate(array2):
            cors = cross_corr(d1,d2)
            cor_dict[(t1+str(idx1),t2+str(idx2))] = cors
    return cor_dict

motif_dict = {}
decov_dict = {}
for tissue in tissue_list:
    path = rf"analyze/w82npy/stage{stage}/{tissue}/t_conv_weight.npy"
    de_path = rf"analyze/w82npy/stage{stage}/{tissue}/t_deconv_weight.npy"
    motif = np.load(path)
    print(motif.shape)
    motif_dict[tissue] = motif
    de_conv = np.load(de_path)
    decov_dict[tissue] = de_conv
motif_cor = {}
for tissue_idx in range(len(motif_dict.keys())):
    for t2_idx in range(tissue_idx+1,len(motif_dict.keys())):
        result = corr_compute(motif_dict[list(motif_dict.keys())[tissue_idx]],motif_dict[list(motif_dict.keys())[t2_idx]],t1=list(motif_dict.keys())[tissue_idx],t2=list(motif_dict.keys())[t2_idx])
        motif_cor.update(result)
print(motif_cor)

G = nx.Graph()
for (node1,node2),edge in motif_cor.items():
    G.add_node(node1)
    G.add_node(node2)
    if edge > 0.8:
        G.add_edge(node1,node2,weight=edge)

components = list(nx.connected_components(G))
pos = nx.spring_layout(G)  # 使用spring布局
edges = G.edges(data=True)
weights = [edge[2]['weight'] for edge in edges]

shapes_dict = {
    'Flower':'o', 
    'Pod':'s', 
    'Seed':'^', 
    'Leaf':'D', 
    'Root':'v', 
    'Shoot':'p', 
    'Nodule':'*', 
    'Stemtip':'h',
    'merge':'>'
    }
color_dict = [
    '#003366',  # 深蓝色 (Dark Blue)
    '#006400',  # 深绿色 (Dark Green)
    '#8B4513',  # 赭色 (Saddle Brown)
    '#B22222',  # 深红色 (Firebrick)
    '#FF4500',  # 橙红色 (Orange Red)
    '#A0522D',  # 土褐色 (Sienna)
    '#FFD700',  # 金色 (Gold)
    '#4B0082',  # 靛青色 (Indigo)
    '#FF6347',  # 番茄色 (Tomato)
    '#FF1493',  # 深粉色 (Deep Pink)
    '#FF8C00',  # 暗橙色 (Dark Orange)
    '#FF00FF',  # 品红色 (Magenta)
    '#00BFFF',  # 深天蓝色 (Deep Sky Blue)
    '#32CD32',  # 石青色 (Lime Green)
    '#FF69B4',  # 热粉色 (Hot Pink)
    '#8A2BE2',  # 蓝紫色 (Blue Violet)
    '#7FFF00',  # 查特酒绿 (Chartreuse)
    '#DC143C',  # 猩红色 (Crimson)
    '#FF7F50',  # 珊瑚色 (Coral)
    '#ADFF2F',  # 黄绿 (Green Yellow)
    '#FF00FF',  # 品红色 (Magenta)
    '#00FA9A',  # 中海绿色 (Medium Spring Green)
    '#FFB6C1',  # 浅粉色 (Light Pink)
    '#4682B4',  # 钢蓝色 (Steel Blue)
    '#2E8B57',  # 海绿色 (Sea Green)
    '#8B008B',  # 深紫红色 (Dark Magenta)
    '#7CFC00',  # 草绿色 (Lawn Green)
    '#F0E68C',  # 草绿色 (Khaki)
    '#B0E57C',  # 黄绿色 (Inchworm)
    '#F5DEB3',  # 小麦色 (Wheat)
    '#D2691E',  # 巧克力色 (Chocolate)
    '#8B0000',  # 深红色 (Dark Red)
    '#A0522D',  # 土褐色 (Sienna)
    '#FF1493',  # 深粉色 (Deep Pink)
    '#FF8C00',  # 暗橙色 (Dark Orange)
    '#9932CC',  # 深紫色 (Dark Orchid)
    '#FF4500',  # 橙红色 (Orange Red)
    '#EE82EE',  # 紫罗兰 (Violet)
    '#98FB98',  # 苍绿 (Pale Green)
    '#00FF7F',  # 春绿 (Spring Green)
    '#6A5ACD',  # 板岩蓝 (Slate Blue)
    '#483D8B',  # 深板岩蓝 (Dark Slate Blue)
    '#2F4F4F',  # 深灰青色 (Dark Slate Gray)
    '#00CED1',  # 暗宝石青 (Dark Turquoise)
    '#9400D3',  # 暗紫色 (Dark Violet)
    '#FFDEAD',  # 纳瓦白 (Navajo White)
    '#FFA07A',  # 浅鲑鱼色 (Light Salmon)
    '#7B68EE',  # 中度板岩蓝 (Medium Slate Blue)
    '#00FF00',  # 绿色 (Lime)
    '#DA70D6',  # 兰花色 (Orchid)
    '#ADFF2F',  # 黄绿色 (Green Yellow)
    '#DDA0DD',  # 梅红 (Plum)
    '#808080',  # 灰色 (Gray)
    '#FA8072',  # 鲑鱼色 (Salmon)
    '#8FBC8F',  # 深海绿 (Dark Sea Green)
    '#00FF00',  # 亮绿色 (Lime)
    '#5F9EA0',  # 青色 (Cadet Blue)
    '#F08080',  # 浅珊瑚色 (Light Coral)
    '#FFE4E1',  # 雪 (Snow)
    '#40E0D0',  # 青绿色 (Turquoise)
    '#FFDAB9',  # 桃色 (Peach Puff)
    '#E9967A',  # 深鲑鱼色 (Dark Salmon)
    '#DC143C',  # 深猩红 (Crimson)
    '#FF7F50',  # 珊瑚色 (Coral)
    '#FFE4B5',  # 秘鲁 (Peru)
    '#8A2BE2',  # 蓝紫色 (Blue Violet)
    '#BC8F8F',  # 玫瑰褐色 (Rosy Brown)
    '#D2691E',  # 巧克力色 (Chocolate)
    '#FF4500',  # 橙红色 (Orange Red)
    '#FF6347',  # 番茄色 (Tomato)
    '#FF8C00',  # 暗橙色 (Dark Orange)
    '#FFD700',  # 金色 (Gold)
    '#FFFF00',  # 黄色 (Yellow)
    '#808000',  # 橄榄色 (Olive)
    '#6B8E23',  # 橄榄褐色 (Olive Drab)
    '#008000',  # 绿色 (Green)
    '#2E8B57',  # 海绿色 (Sea Green)
    '#556B2F',  # 深橄榄绿 (Dark Olive Green)
    '#006400',  # 深绿色 (Dark Green)
    '#8B0000',  # 深红色 (Dark Red)

]
node_shape = {}
for n in G.nodes():
    for key in shapes_dict.keys():
        if key in n:
            node_shape[n] = shapes_dict[key]
            break
import re
def reset(motifpwm):
    offset = (motifpwm.shape[0]-1)/2 - np.round((np.arange(motifpwm.shape[0]) * np.square((motifpwm - motifpwm.mean(axis=1,keepdims=True))).sum(axis=1)).sum()/  np.square((motifpwm - motifpwm.mean(axis=1,keepdims=True))).sum(axis=1).sum())
    # print(motifpwm)
    try:
        if  offset > 0:
            motifpwm[int(offset):, :] = motifpwm[:-int(offset),:]
            motifpwm[:int(offset), :] = 0
        else:
            motifpwm[:-int(np.abs(offset)), :] = motifpwm[int(np.abs(offset)):,:]
            motifpwm[-int(np.abs(offset)):, :] = 0
    except ValueError:
        pass
        
    motifpwm = motifpwm - np.mean(motifpwm,axis=1,keepdims=True)
    motifpwm = motifpwm - np.abs(motifpwm).mean(1,keepdims=True)*0.4
    motifpwm = np.where(np.abs(motifpwm)<motifpwm[motifpwm>0].mean() + motifpwm[motifpwm>0].std(),0,motifpwm)
    return motifpwm
map_dict = {
    0:'A',
    1:'C',
    2:'G',
    3:'T'
}

node_color ={}
motif_label_list = []
csv = {
    'motif':[],
    'tissue':[],
    'type':[]
}

motif_list = []
de_conv_list = []
de_rcconv_list = []
for idx, component in enumerate(components):
    subgraph = G.subgraph(component)  # 提取该连通分量的子图
    max_node = max(subgraph.nodes, key=lambda node: subgraph.degree(node))  # 找到度最大的节点
    t,idx = re.sub(r'\d+','',max_node),int(re.findall(r"\d+",max_node)[0])
    print(t,idx)
    # t,idx = re.sub(r'\d+','',list(component)[0]),int(re.findall(r"\d+",list(component)[0])[0])
    npy = motif_dict[t][idx]
    de_npy = decov_dict[t][idx]
    print(motif_dict[t].shape,decov_dict[t].shape)
    de_rcnpy = decov_dict[t][motif_dict[t].shape[0] + idx]

    motif_list.append(motif_dict[t][idx])
    de_conv_list.append(de_npy)
    de_rcconv_list.append(de_rcnpy)
    print(npy.shape)
    npy = reset(npy.T).astype(object)
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

    motif_label_list.append(motif_str)
    t_list = []
    for node in component:
        node_color[node] = color_dict[idx]
        t_list.append(re.sub(r'\d+','',node))
    t_list = ','.join(np.array(list(set(t_list))))
    csv['motif'].append(motif_str)
    csv['tissue'].append(t_list)

    if len(list(component)) ==1:
        print('tissue-only:',t,motif_str)
        csv['type'].append('Tissue specificity')
    else:
        csv['type'].append('Tissue non-specificity')
motif_list = np.array(motif_list)
csv = pd.DataFrame.from_dict(csv)
deconv_ = np.concatenate([np.stack(de_conv_list),np.stack(de_rcconv_list)],axis=0)
shape = G.nodes
if stage =='1':
    csv.to_csv(r"/Data5/pfGao/xtwang/TSS/tss/analyze/new_model_result/motif_info.csv")
    np.save('analyze/result/motif.npy',motif_list)
    np.save('analyze/result/motif_deconv.npy',deconv_)

else:
    csv.to_csv(r"/Data5/pfGao/xtwang/TSS/tss/analyze/new_model_result/other_info.csv")
    np.save('analyze/result/other.npy',motif_list)
    np.save('analyze/result/other_deconv.npy',deconv_)

# 绘制图形
# 设置布局
# pos = nx.spring_layout(G, seed=42,k=10)  # 固定seed以保证布局一致性
pos = nx.circular_layout(G)

# 美化节点
node_size = 8

# 美化边
edge_width = [G[u][v]['weight'] for u, v in G.edges()]  # 根据权重调整边的宽度
edge_color = 'gray'

# 绘制图形

fig,ax = plt.subplots(figsize=(40, 40))
for shape in set(shapes_dict.values()):
    nx.draw_networkx_nodes(G, pos,ax=ax,
                        nodelist=[node for node in G.nodes() if node_shape[node] == shape],
                        node_color=[node_color.get(node, 'grey') for node in G.nodes() if node_shape[node] == shape],
                        node_shape=shape,
                        node_size=200, alpha=0.8)
nx.draw_networkx_edges(G, pos, ax=ax,width=edge_width, alpha=0.5, edge_color='gray')

# 添加边标签
color_handles = [mpatches.Patch(color=color_dict[idx], label=motif_label_list[idx]) for idx, cc in enumerate(components)]

shape_handles = [Line2D([0], [0], marker=shapes_dict[tt], color='w', label=tt,
                        markerfacecolor='gray', markersize=10)  for idx, tt in enumerate(tissue_list)]

legend1 = ax.legend(handles=color_handles, loc='upper right', title="Motif",bbox_to_anchor=(1.08,1),fontsize=12)

# 添加形状图例
legend2 = ax.legend(handles=shape_handles, loc='lower center', bbox_to_anchor=(0.5, -0.02), ncol=len(shape_handles),title='Tissues',fontsize=12)

# 显示图形
plt.gca().add_artist(legend1)
plt.title("Network Graph of Motif in Tissues", fontsize=50)

plt.savefig(f'/Data5/pfGao/xtwang/TSS/tss/analyze/new_model_result/W82_Network Graph of Motif in Tissues {stage}.svg',dpi=800,format='svg')


