import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress
sample_list = ['930-3','YY1-12']
motif = 'HY5-1'
WT = 'X930.5.1'
KO = 'HY5.1.1'
exp_result = pd.read_csv(r"/Volumes/My_Disk_1/code/Geno_RT/analyze/KO_experienment/merge3/result/Unidirection_result.xls",sep='\t')
compute_list = [('YY1','YY1.12.1','X930.3.1'),('HY5','HY5.1.1','X930.5.1'),('TCP20','TCP.1.1','X930.5.1'),('DREB1E','DREB.1.1','X930.5.1'),('ABF1','ABF.2.1','X930.5.1')]
result_dict = {}
for motif,KO,WT in compute_list:
    # print(motif,KO,WT)
    result_dict[motif] = {
        'corr':[],
        'rank':[],
        'k':0
    }
    predict_result  = pd.read_csv(rf"/Volumes/My_Disk_1/code/Geno_RT/analyze/KO_experienment/net_predict_KO_{motif}_W82_info_root.csv")
    motif_exp_result = exp_result.loc[:,[KO,WT]]
    predict_result['change'] = (predict_result['KO_max'] - predict_result['Pred_max']) / predict_result['Pred_max']
    motif_exp_result['change'] = (motif_exp_result[KO] - motif_exp_result[WT]) / (motif_exp_result[WT] + 0.000001)
    predict_result.index = predict_result['chrom'] + predict_result['pos'].map(str) + predict_result['strand']
    motif_exp_result.index = exp_result['seqnames'] + exp_result['thick.start'].map(str) + exp_result['strand']

    both_tss = list(set(predict_result.index) & set(motif_exp_result.index))
    predict_result = predict_result.loc[both_tss]
    motif_exp_result = motif_exp_result.loc[both_tss]
    predict_result['change.abs'] = predict_result['change'].abs()
    motif_exp_result['change.abs'] = motif_exp_result['change'].abs()
    lb = motif_exp_result['change'] * predict_result['change']
    acc = lb[lb>0]
    predict_result = predict_result.loc[acc.index]
    motif_exp_result = motif_exp_result.loc[acc.index]
    predict_result = predict_result.sort_values(by='change.abs',ascending=True)
    motif_exp_result = motif_exp_result.loc[predict_result.index]

    predict_result['row_id'] = np.arange(predict_result.shape[0]) / predict_result.shape[0]
    print(predict_result)
    predict_result.to_csv(f'HC_TSS_{motif}_predict.csv')
    for rank in range(1,100,9):
        # rank_pred_result = predict_result[(predict_result['row_id']>(rank-1)/100)&(predict_result['row_id']<=rank/100) ]
        rank_pred_result = predict_result[predict_result['row_id']<=rank/100]

        rank_exp_result = motif_exp_result.loc[rank_pred_result.index]
        c = np.corrcoef(rank_exp_result['change'],rank_pred_result['change'])[0,1]
        result_dict[motif]['corr'].append(c)
        result_dict[motif]['rank'].append(rank/100)

        print(motif,c)
    corr_k = linregress(result_dict[motif]['rank'],result_dict[motif]['corr'])
    result_dict[motif]['k'] = corr_k.slope
    print(corr_k.slope)
    print(len(result_dict[motif]['corr']))
    # break

result_dict = pd.DataFrame(result_dict).T
result_dict = result_dict.sort_values(by='k',ascending=True)
fig,ax = plt.subplots(figsize=(5,5))
for idx in result_dict.index:
    plt.plot(result_dict.loc[idx,'rank'],result_dict.loc[idx,'corr'][::-1],label=idx,lw=0.8,marker = "o",mfc = "white", ms = 4)
# ax.spines["left"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# plt.ylim(0,1.1)
plt.xlim(-0.1,1.01)
plt.legend()
# plt.show()
plt.ylabel('Correlation')
plt.xlabel('Rank')
plt.savefig('corr_rank.svg',format='svg',dpi=800)
plt.savefig('corr_rank.png',format='png',dpi=800)
plt.show()
print(result_dict)





