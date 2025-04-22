import glob
import torch.nn as nn
import torch
import logomaker
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
def reset(motifpwm):
    offset = (motifpwm.shape[0]-1)/2 - np.round((np.arange(motifpwm.shape[0]) * np.square((motifpwm - motifpwm.mean(axis=1,keepdims=True))).sum(axis=1)).sum()/  np.square((motifpwm - motifpwm.mean(axis=1,keepdims=True))).sum(axis=1).sum())
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
#模型权重文件路径
dir_path = r"/home/share/yfrm3rms/home/pfgao/GenoRT/models"

prop_cycle = plt.rcParams['axes.prop_cycle']
itercolor =  prop_cycle()
#需要计算的组织列表
tissue_list = ['Flower','Leaf','Nodule','Pod','Root','Seed','Shoot','Stemtip','merge']
#模型阶段
stage = '1'



def plotfun(motifpwm, title=None, ax=None):
    motifpwm = pd.DataFrame(motifpwm,columns=['A','C','G','T'])
    # mean = motifpwm.mean(axis=)
    crp_logo = logomaker.Logo(motifpwm,
                              shade_below=.2,
                              fade_below=.2,
                            #   font_name='Arial Rounded MT Bold',
                             ax=ax)

    # style using Logo methods
    crp_logo.style_spines(visible=False)
    crp_logo.style_spines(spines=['left', 'bottom'], visible=True)
    crp_logo.style_xticks(rotation=90, fmt='%d', anchor=0)
    if title is not None:
        crp_logo.ax.set_title(title)
    # style using Axes methods
    crp_logo.ax.set_ylabel("", labelpad=-1)
    crp_logo.ax.xaxis.set_ticks_position('none')
    crp_logo.ax.xaxis.set_tick_params(pad=-1)
    return crp_logo
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
            nn.Conv1d(4, 64, kernel_size=51, padding=25),
            nn.BatchNorm1d(64),
            )
        
        self.encoder = nn.Sequential(
            nn.Conv1d(4,64,kernel_size=15,padding=7),
            nn.BatchNorm1d(64)
            )


        self.atten = attention(feature_num=64)
        
        self.activation = nn.Sigmoid()

        self.deconv = nn.Sequential(
            nn.Conv1d(128, 10, kernel_size=601, padding=300),
            nn.BatchNorm1d(10),
            )
        
        self.decoder = nn.Sequential(
            nn.Conv1d(128,10,kernel_size=15,padding=7),
            nn.BatchNorm1d(10)
            )
        

        self.softplus = nn.Softplus()
        
        self.apply(init_weights)
        
    def forward(self, x):
        
        y = torch.cat([self.atten(self.activation(self.conv(x))), 
                       self.atten(self.activation(self.conv(x.flip([1, 2])).flip([2])))], dim=1)
        
        y_inr = torch.cat([self.encoder((x)), 
                       self.encoder(x.flip([1, 2])).flip([2])], dim=1)

        
        y_act = self.activation(y)
        y_inr_act = self.activation(y_inr)
        y_pred = self.softplus(self.deconv(y_act) + self.decoder(y_inr_act))
    
        return y_pred

class GenoRT_s2(nn.Module):
    def __init__(self,n_motifs):
        super(GenoRT_s2, self).__init__()
        
        self.conv = nn.Conv1d(4, n_motifs, kernel_size=51, padding=25)
        self.conv_inr = nn.Conv1d(4, 40, kernel_size=15, padding=7)
        self.deconv = nn.Sequential(
            nn.Conv1d(in_channels=n_motifs * 2,out_channels=2, kernel_size=601, padding=300),
            nn.BatchNorm1d(2))
        self.deconv_inr = nn.Sequential(
            nn.Conv1d(80, 2, kernel_size=15, padding=7),
            nn.BatchNorm1d(2))

        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()
        self.scaler = nn.Parameter(torch.ones(1))
        self.scaler2 = nn.Parameter(torch.ones(1))

    def forward(self, x,withscaler=True):
        y = torch.cat([self.conv(x), self.conv(x.flip([1, 2])).flip([2])], 1)
        y_inr = torch.cat(
            [self.conv_inr(x), self.conv_inr(x.flip([1, 2])).flip([2])], 1
        )
        if withscaler:
            yact = self.softplus(y * self.scaler**2)

            y_inr_act = self.softplus(y_inr * self.scaler2**2)
        else:
            yact = self.softplus(y)
            y_inr_act = self.softplus(y_inr)
        y_pred = self.softplus(
            self.deconv(yact) + self.deconv_inr(y_inr_act)
        )
        
        return y_pred

class GenoRT_s3(nn.Module):
    def __init__(self,n_motifs,n_inrs):
        super(GenoRT_s3, self).__init__()
        
        self.conv = nn.Conv1d(4, n_motifs, kernel_size=51, padding=25)
        self.conv_inr = nn.Conv1d(4, n_inrs, kernel_size=15, padding=7)
        self.conv_tri = nn.Conv1d(4,40,kernel_size=3,padding=1)
        self.deconv = nn.Sequential(
            nn.Conv1d(in_channels=n_motifs * 2,out_channels=2, kernel_size=601, padding=300),
            nn.BatchNorm1d(2))
        self.deconv_inr = nn.Sequential(
            nn.Conv1d(n_inrs *2 , 2, kernel_size=15, padding=7),
            nn.BatchNorm1d(2))
        self.deconv_tri = nn.Sequential(
            nn.Conv1d(in_channels=80,out_channels=2,kernel_size=15,padding=7),
            nn.BatchNorm1d(2)
        )
        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()
        self.scaler = nn.Parameter(torch.ones(1))
        self.scaler2 = nn.Parameter(torch.ones(1))
        self.scaler3 = nn.Parameter(torch.ones(1))

    def forward(self, x,withscaler=True):
        y = torch.cat([self.conv(x), self.conv(x.flip([1, 2])).flip([2])], 1)
        y_inr = torch.cat(
            [self.conv_inr(x), self.conv_inr(x.flip([1, 2])).flip([2])], 1
        )
        y_tri = torch.cat(
            [self.conv_tri(x), self.conv_tri(x.flip([1, 2])).flip([2])], 1
        )
        if withscaler:
            yact = self.softplus(y * self.scaler**2)
            y_inr_act = self.softplus(y_inr * self.scaler2**2)
            y_tri_act = self.softplus(y_tri * self.scaler3**2)
        else:
            yact = self.softplus(y)
            y_inr_act = self.softplus(y_inr)
            y_tri_act = self.softplus(y_tri)

        y_pred = self.softplus(
            self.deconv(yact) + self.deconv_inr(y_inr_act) + self.deconv_tri(y_tri_act)
        )
        
        return y_pred
    
import os
for tissue in tissue_list:
    nets_ = os.listdir(dir_path)
    nets_ = [os.path.join(dir_path,i) for i in nets_ if i.startswith(f'{tissue}_stage_{stage}',0)]
    print(nets_)
    print(len(nets_))
    nets = []
    # filenames = []
    for i in nets_:
        net =GenoRT()
        net = torch.load(i)
        net.cpu()
        nets.append(net)
    if stage =='1':
        mats = [nets[i].conv[0].weight.detach().numpy() for i in range(len(nets))]
        mats_norm = [mat - mat.mean(axis=1, keepdims=True) for mat in mats]
        demats = [nets[i].deconv[0].weight.detach().numpy() for i in range(len(nets))]
    else:
        mats = [nets[i].conv_inr.weight.detach().numpy() for i in range(len(nets))]
        mats_norm = [mat - mat.mean(axis=1, keepdims=True) for mat in mats]
        demats = [nets[i].deconv_inr[0].weight.detach().numpy() for i in range(len(nets))]
    from scipy.stats import pearsonr
    from scipy.signal import correlate2d
    from matplotlib import pyplot as plt

    import random
    import numpy as np
    # @njit( )
    # from reset import reset
    def cross_corr(x, y):
        cors = []
        i=0
        if stage=='1':
            for j in range(y.shape[1]-5):
                minlen = np.fmin(x.shape[1]-i, y.shape[1]-j)
                cors.append(np.fmax(np.corrcoef(x.flatten(), 
                                        np.concatenate((y[:,j:],y[:,:j]), axis=1).flatten())[0,1],
                                np.corrcoef(x.flatten(), 
                                        np.concatenate((y[:,j:],y[:,:j]), axis=1)[::-1,::-1].flatten())[0,1]))
        else:
            for j in range(y.shape[1]):
                minlen = np.fmin(x.shape[1]-i, y.shape[1]-j)
                cors.append(np.fmax(np.corrcoef(x.flatten(), 
                                        np.concatenate((y[:,j:],y[:,:j]), axis=1).flatten())[0,1],
                                np.corrcoef(x.flatten(), 
                                        np.concatenate((y[:,j:],y[:,:j]), axis=1)[::-1,::-1].flatten())[0,1]))
        return np.array(cors)

    def comparemats(mats):
        crossmats = {}
        validmats = {}
        for ii in range(len(mats)):
            for jj in range(ii+1, len(mats)):
                cors = []
                print(mats_norm[ii].shape)
                for i in range(mats_norm[ii].shape[0]):
                    cors_row = []
                    for j in range(mats_norm[ii].shape[0]):
                        cors_row.append(cross_corr(mats_norm[ii][i], mats_norm[jj][j]).max())
                        if cross_corr(mats_norm[ii][i], mats_norm[jj][j]).max() > 0.90:
                            print(cross_corr(mats_norm[ii][i], mats_norm[jj][j]).max())
                    
                    cors.append(cors_row)
                print(ii,jj)
                cors = np.array(cors)
                crossmats[(ii,jj)]=cors
                validmats[(ii,jj)]= (np.abs(mats_norm[ii]).max(axis=2).max(axis=1)[:,None]>0.1) & (np.abs(mats_norm[jj]).max(axis=2).max(axis=1)[None,:]>0.1)

        return crossmats, validmats
    crossmats, validmats = comparemats(mats)
    matchlist = []
    matchscores = []
    for i in range(len(nets)):
        for j in range(i+1,len(nets)):
            mat = crossmats[(i,j)].copy()
            mat[mat < mat.max(axis=1,keepdims=True)] = 0
            mat[mat < mat.max(axis=0,keepdims=True)] = 0
            mat[~validmats[(i,j)]] = 0
            matchlist.append(np.argwhere(mat>0.90).astype(str)+np.array(["_"+str(i), "_"+str(j)],dtype=object)[None,:])
            matchscores.append(mat[mat>0.90])

    import seaborn as sns
    sns.set(rc={"figure.dpi":300, 'savefig.dpi':300})
    sns.set_style("white")
    import networkx as nx
    from collections import defaultdict
    g = nx.Graph()
    g.add_weighted_edges_from(np.hstack([np.concatenate(matchlist, axis=0),np.concatenate(matchscores)[:,None]]))
    from matplotlib.backends.backend_pdf import PdfPages
    
    bestmats = []
    bestscores = []
    selectmats = []
    selectdemats = []
    selectdemats_rc = []
    
    for idx,cc in enumerate(list(nx.connected_components(g))):
        bestscore = 0
        matinds = []
        motifinds = []
        scores = []
        attens = []
        _, axes = plt.subplots( figsize=(10,len(cc)*1), nrows=len(cc),ncols=2, dpi=300)
        save_path = rf"/home/share/yfrm3rms/home/pfgao/GenoRT/analyze/npy/stage{stage}/{tissue}/edge_{idx}"
        os.makedirs(save_path,exist_ok=True)
        for i in cc:
            motifind, matind = list(map(int, i.split('_')))
            score = np.sum([crossmats[i,matind][:,motifind].max() if i < matind else crossmats[matind,i][motifind,:].max() for i in np.setdiff1d(range(len(nets)),[matind])  ])
            matinds.append(matind)
            motifinds.append(motifind)
            scores.append(score)

        for ii, i in enumerate(np.argsort(-np.array(scores))):
            # plotfun(mats[matinds[i]][motifinds[i]].T, ax=axes[ii][0])
            np.save(rf'{save_path}/conv_mat_net_{matinds[i]}_motif_{motifinds[i]}_axe{ii}_score{scores[i]}.npy',mats[matinds[i]][motifinds[i]])
            np.save(rf'{save_path}/demat_net_{matinds[i]}_motif_{motifinds[i]}_axe{ii}_plus_score{scores[i]}.npy',demats[matinds[i]][:,motifinds[i]])
            np.save(rf'{save_path}/demat_net_{matinds[i]}_motif_{motifinds[i]}_axe{ii}_minus_score{scores[i]}.npy',demats[matinds[i]][:,mats[matinds[i]][motifinds[i]].T.shape[1]+motifinds[i]])

        
        if len(cc)>len(nets)*0.3:
            for i in cc:
                motifind,matind = list(map(int,i.split('_')))
                score = np.sum([crossmats[i,matind][:,motifind].max() if i < matind else crossmats[matind,i][motifind,:].max() for i in np.setdiff1d(range(len(nets)),[matind])  ])
                if score > bestscore:
                    bestmat = mats[matind][motifind]
                    bestdemat = demats[matind][:,motifind]
                    bestdematrc = demats[matind][:,bestmat.T.shape[1]+motifind]
                    bestscore = score

            if stage == '1':
                inp = bestmat.T
                motif_reset = reset(inp)
                motif_idx = np.argwhere(motif_reset.max(axis=1)!=0)
                motif_len = motif_idx[-1] - motif_idx[0]
                print(motif_reset.shape)
                idx_list = np.argmax(motif_reset[motif_idx,:][:,0,:],axis=1)
                # if motif_len[0] > 3 and len(list(set(idx_list)))!=1 :
                if motif_len[0] > 0 and motif_len[0] < 40 and len(list(set(idx_list)))!=1 :

                    print(motif_idx[-1],motif_idx[0])
                    print('motif len:',motif_len)
                    selectmats.append(bestmat)
                    selectdemats.append(bestdemat)
                    selectdemats_rc.append(bestdematrc)
            else:
                inp = bestmat.T
                motif_reset = reset(inp)
                motif_idx = np.argwhere(motif_reset.max(axis=1)!=0)
                if len(motif_idx) != 0:
                    selectmats.append(bestmat)
                    selectdemats.append(bestdemat)
                    selectdemats_rc.append(bestdematrc)
    motif_weight = np.stack(arrays=selectmats)
    deconv_weight = np.concatenate([np.stack(selectdemats),np.stack(selectdemats_rc)],axis=0)
    np.save(arr=motif_weight,file=f'/home/share/yfrm3rms/home/pfgao/GenoRT/analyze/npy/stage{stage}/{tissue}/t_conv_weight.npy')
    np.save(arr=deconv_weight,file=f'/home/share/yfrm3rms/home/pfgao/GenoRT/analyze/npy/stage{stage}/{tissue}/t_deconv_weight.npy')

            
            
        

