#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
sys.path.insert(0, 'D:/pythonbao/')
from tcplot import tcplot,sen,spe
import pandas as pd
import numpy as np
import os
from pandas import DataFrame
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import h5py
import gc
import seaborn as sns
import warnings
from scipy import signal
from sklearn.preprocessing import label_binarize
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split,GridSearchCV,LeaveOneOut,cross_val_score,StratifiedKFold
from sklearn import svm,datasets
from sklearn.svm import SVC
from itertools import cycle
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import f1_score,classification_report,confusion_matrix, precision_score,recall_score,roc_auc_score, roc_curve, auc, precision_recall_curve
import scipy
from scipy.stats import pearsonr,spearmanr
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier
from skopt import BayesSearchCV
import joblib
from sklearn.preprocessing import LabelEncoder
warnings.filterwarnings("ignore")
get_ipython().run_line_magic('matplotlib', 'inline')

def readfilez(file):  

    f = h5py.File(file,'r')
    f.keys() 
    eegz=f['z'][:]   
#     print(np.shape(eegz))
    eeg1=np.array(eegz[:,:30]).astype(np.float16)
    eeg2=np.array(eegz[:,30:60]).astype(np.float16)
    emg=np.array(eegz[:,60:90]).astype(np.float16)
    scores=np.array(eegz[:,90:91]).astype(np.float16)
    f.close()
    gc.collect()
    return emg,eeg1,eeg2,scores
plt.rcParams['font.sans-serif'] = ['SimHei']




def fill_plot(interp_tpr,name,aucs):
    colors = ['#7E4D99', '#53A362','#DA3F34']   
    target_names = ['Wakefulness', 'NREM', 'REM']
    interp_tpr=np.array(interp_tpr)
    fig, ax = plt.subplots(figsize=(8, 8))
    mean_tpr = np.mean(interp_tpr[:,0,:], axis=0)
    mean_tpr[-1] = 1.0
    std_tpr = np.std(interp_tpr[:,0,:], axis=0)
    mean_auc = np.mean(interp_tpr[:,0,:])
    std_auc = np.std(interp_tpr[:,0,:])   
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")
    ax.plot(
        mean_fpr,
        mean_tpr,
        color=colors[0],
        label=f"Mean ROC of '{target_names[0]}'(Mean AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
        lw=2,
        alpha=0.8,
    )
    conf_intveral = scipy.stats.norm.interval(0.95,mean_tpr,std_tpr)
    ax.fill_between(mean_fpr, conf_intveral[0], conf_intveral[1], color=colors[0], alpha=0.2,label=r"95% Confidence interval")  

    
    mean_tpr = np.mean(interp_tpr[:,1,:], axis=0)
    std_tpr = np.std(interp_tpr[:,1,:], axis=0)

    mean_auc = np.mean(interp_tpr[:,1,:])
    std_auc = np.std(interp_tpr[:,1,:])
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)    
    ax.plot(
        mean_fpr,
        mean_tpr,
        color=colors[1],
        label=f"Mean ROC of '{target_names[1]}'(Mean AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
        lw=2,
        alpha=0.8,
    )

    conf_intveral = scipy.stats.norm.interval(0.95,mean_tpr,std_tpr)
    ax.fill_between(mean_fpr, conf_intveral[0], conf_intveral[1], color=colors[1], alpha=0.2,label=r"95% Confidence interval")  
     
    mean_tpr = np.mean(interp_tpr[:,2,:], axis=0)
    std_tpr = np.std(interp_tpr[:,2,:], axis=0)

    mean_auc = np.mean(interp_tpr[:,2,:])
    std_auc = np.std(interp_tpr[:,2,:])   
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)    
    ax.plot(
        mean_fpr,
        mean_tpr,
        color=colors[2],
        label=f"Mean ROC for '{target_names[2]}'(Mean AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
        lw=2,
        alpha=0.8,
    )
     
    conf_intveral = scipy.stats.norm.interval(0.95,mean_tpr,std_tpr)
    ax.fill_between(mean_fpr, conf_intveral[0], conf_intveral[1], color=colors[2], alpha=0.1,label=r"95% Confidence interval")   

    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontdict={'family': 'Times New Roman', 'size': 26,'weight':'bold'})
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontdict={'family': 'Times New Roman', 'size': 26,'weight':'bold'})
    plt.xticks(fontname='Times New Roman', fontsize=16,weight='bold')
    plt.yticks(fontname='Times New Roman', fontsize=16,weight='bold')
    ax.legend(loc="lower right")
    ax.legend(prop={'family': 'Times New Roman', 'size': 14,'weight':'bold'}) 
    plt.savefig('E:/fenqi/{}'.format(name),dpi=300,bbox_inches = 'tight')
    plt.close()
#     plt.show()
    

def feature_impo(num,lgbcan,pbar):
    eegx=list(i for i in np.arange(1,31))
    eegx = [str(x)+ ' Hz'for x in eegx]
    f_head2=[]
    f_head2.extend(list('Cerebellar EEG '+j for j in eegx))
    f_head2.extend(list('Cerebral EEG '+j for j in eegx))
    f_head2.extend(list('EMG '+j for j in eegx))  
    impo,fh=[],[]
    for j in range(30):
        info=os.listdir(f_h5z)
        infoxl = np.random.choice(info,num, replace=False)
        infoyc =np.setdiff1d(info, infoxl)
        tab=[]
        eeg1f,eeg2f,emgf=[],[],[]
        for i in range(np.shape(infoxl)[0]):

            domain = os.path.abspath(f_h5z)
            infor = os.path.join(domain,infoxl[i])
    #             print(infor)
            emg,eeg1,eeg2,scores=readfilez(infor)
            eeg1f.extend(eeg1)
            eeg2f.extend(eeg2)
            emgf.extend(emg)
            tab.extend(scores)

        tabyc=[]
        eeg1fyc,eeg2fyc,emgfyc=[],[],[]

        for i in range(np.shape(infoyc)[0]):

            domain = os.path.abspath(f_h5z)
            infor = os.path.join(domain,infoyc[i])
            emg,eeg1,eeg2,scores=readfilez(infor)
            eeg1fyc.extend(eeg1)
            eeg2fyc.extend(eeg2)
            emgfyc.extend(emg)
            tabyc.extend(scores)
        ez= np.concatenate((eeg1f, eeg2f,emgf), axis=1)
        ezyc= np.concatenate((eeg1fyc, eeg2fyc,emgfyc), axis=1)
        can=dict(lgbcan)

        le = LabelEncoder()
        tab = le.fit_transform(tab)
        tabyc = le.fit_transform(tabyc)
        x_train,y_train=np.array(ez),np.array(tab)
        x_test,y_test=np.array(ezyc),np.array(tabyc)
        model = LGBMClassifier(learning_rate=can['learning_rate'],num_leaves=can['num_leaves'],
                               max_depth=can['max_depth'],min_child_samples=can['min_child_samples'],
                               max_bin=can['max_bin'],subsample=can['subsample'],
                               colsample_bytree=can['colsample_bytree'],min_child_weight=can['min_child_weight'],
                               reg_lambda=can['reg_lambda'],reg_alpha=can['reg_alpha'],
                               scale_pos_weight=can['scale_pos_weight'],
                               n_estimators=can['n_estimators'],verbose=-1,
                               importance_type='split')

        model.fit(x_train, y_train)
        pred=model.predict(x_test)
        preds=model.predict_proba(x_test)
        score = model.score(x_test, y_test)
        target_names = ['Wakefulness', 'NREM', 'REM']
        impo.extend(model.feature_importances_)
        
        fh.extend(f_head2)
   
        pbar.update()

    importance_df = pd.DataFrame({'Feature':fh, 'Importance': impo})

    return importance_df

bcanshu=[('colsample_bytree', 0.5334545598649226),
             ('learning_rate', 0.026594231510148683),
             ('max_bin', 1000),
             ('max_depth', 50),
             ('min_child_samples', 48),
             ('min_child_weight', 9),
             ('n_estimators', 1500),
             ('num_leaves', 50),
             ('reg_alpha', 1e-09),
             ('reg_lambda', 1e-09),
             ('scale_pos_weight', 8.30563367736703e-06),
             ('subsample', 0.01)]

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
f_h5z="E:/fenqi/h5zz"
f_sv="E:/fenqi/compare_cerebellar_emg"
df = pd.read_csv('E:/fenqi/bestfea/feat_impo.csv')

def get_top5_features(df, keyword):
    df_filtered = df[df['Feature'].str.contains(keyword)]
    mean_df = df_filtered.groupby('Feature')['Importance'].mean()
    top_5_features = mean_df.nlargest(5).index
    return df_filtered[df_filtered['Feature'].isin(top_5_features)]
def get_top1_features(df, keyword):
    df_filtered = df[df['Feature'].str.contains(keyword)]
    mean_df = df_filtered.groupby('Feature')['Importance'].mean()
    top_5_features = mean_df.nlargest(5).index
    fifth_feature = top_5_features[0]
    return df_filtered[df_filtered['Feature'] == fifth_feature]

df_emg = get_top5_features(df, 'EMG')
df_cerebral = get_top5_features(df, 'Cerebral')
df_cerebellar = get_top5_features(df, 'Cerebellar')

df_emg.to_csv(f_sv+'top5_emg_output.csv', index=False)
df_cerebral.to_csv(f_sv+'top5_cerebral_output.csv', index=False)
df_cerebellar.to_csv(f_sv+'top5_cerebellar_output.csv', index=False)
dfx=pd.read_csv('E:/fenqi/bestfea/top5_cerebellar_output.csv')
dfd=pd.read_csv('E:/fenqi/bestfea/top1_cerebral_output.csv')
dfj=pd.read_csv('E:/fenqi/bestfea/top5_emg_output.csv')

featuresx =dfx['Feature'].unique().tolist()
featuresd =dfd['Feature'].unique().tolist()
featuresj =dfj['Feature'].unique().tolist()
featuresx.extend(featuresd)
featuresj.extend(featuresd)

def xunliancc(num):

    interp_tprcc=[]
    interp_tprce=[]

    scorecc,scorece=[],[]
    dfcc=pd.DataFrame()
    dfce=pd.DataFrame()
    dfacc=pd.DataFrame()
    dface=pd.DataFrame()   
    dfa=pd.DataFrame()  
    for j in range(30):
        info=os.listdir(f_h5z)
        print(info)
        infoxl = np.random.choice(info,num, replace=False)
        infoyc =np.setdiff1d(info, infoxl)
        tab=[]
        eeg1f,eeg2f,emgf=[],[],[]
        for i in range(np.shape(infoxl)[0]):

            domain = os.path.abspath(f_h5z)
            infor = os.path.join(domain,infoxl[i])

            emg,eeg1,eeg2,scores=readfilez(infor)
            eeg1f.extend(eeg1)
            eeg2f.extend(eeg2)
            emgf.extend(emg)
            tab.extend(scores)

        print(infoyc)
        tabyc=[]
        eeg1fyc,eeg2fyc,emgfyc=[],[],[]
        
        for i in range(np.shape(infoyc)[0]):

            domain = os.path.abspath(f_h5z)
            infor = os.path.join(domain,info[i])
            emg,eeg1,eeg2,scores=readfilez(infor)
            eeg1fyc.extend(eeg1)
            eeg2fyc.extend(eeg2)
            emgfyc.extend(emg)
            tabyc.extend(scores)

        ezcc=np.concatenate((eeg1f, eeg2f), axis=1)     
        ezccyc=np.concatenate((eeg1fyc, eeg2fyc), axis=1) 
        ezce=np.concatenate((emgf, eeg2f), axis=1)     
        ezceyc=np.concatenate((emgfyc, eeg2fyc), axis=1)         

        ez=ezcc[:,indicescc]
        ezyc=ezccyc[:,indicescc]
        print(np.shape(ez))
        interp_tprl,score,df,dfauc=tcplot(ez,ezyc,tab,tabyc,bcanshu,'Cerebellar EEG&Cerebral EEG',num)
        interp_tprcc.append(interp_tprl)
        scorecc.append(score)
        dfcc=dfcc.append(df)
        dfacc= dfacc.append(dfauc)
        
        ez=ezce[:,indicesce]
        ezyc=ezceyc[:,indicesce]
        print(np.shape(ez))
        interp_tprl,score,df,dfauc=tcplot(ez,ezyc,tab,tabyc,bcanshu,'EMG&Cerebral EEG',num)
        interp_tprce.append(interp_tprl)
        scorece.append(score)
        dfce=dfce.append(df)        
        dface= dface.append(dfauc)
        
    print("ni",dfcc)
    print("ni",dfce)
    dfacc['Area']="Cerebellar EEG&Cerebral EEG"
    dface['Area']="EMG&Cerebral EEG"
    dfa=dfa.append(dfacc) 
    dfa=dfa.append(dface)
    dfa.to_csv(f_sv+"{}AUC.csv".format(num))  

    fill_plot(interp_tprce,"{}cetop".format(num),scorece)
    fill_plot(interp_tprcc,"{}cctop".format(num),scorecc)
    dfcc.to_csv(f_sv+"{}cerebellar_cerebral_report.csv".format(num))
    dfce.to_csv(f_sv+"{}cerebral_emg_report.csv".format(num))
    
mean_fpr = np.linspace(0, 1, 100)
eegx=list(i for i in np.arange(1,31))
eegx = [str(x)+ ' Hz'for x in eegx]
f_head1=[]
f_head1.extend(list('Cerebellar EEG '+j for j in eegx))
f_head1.extend(list('Cerebral EEG '+j for j in eegx))

intersection = np.intersect1d(f_head1, featuresx)

indicescc = [f_head1.index(i) for i in intersection]

f_head2=[]
f_head2.extend(list('Cerebral EEG '+j for j in eegx))
f_head2.extend(list('EMG '+j for j in eegx))

intersection = np.intersect1d(f_head2, featuresj)

indicesce = [f_head2.index(i) for i in intersection]

for i in [5]:
    xunliancc(i)


# In[ ]:


import pandas as pd
import numpy as np
import os
from pandas import DataFrame
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import h5py
import gc
import seaborn as sns
import warnings
from scipy import signal
from sklearn.preprocessing import label_binarize
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split,GridSearchCV,LeaveOneOut,cross_val_score,StratifiedKFold
from sklearn import svm,datasets
from sklearn.svm import SVC
from itertools import cycle
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import f1_score,classification_report,confusion_matrix, precision_score,recall_score,roc_auc_score, roc_curve, auc, precision_recall_curve
import scipy
from scipy.stats import pearsonr,spearmanr
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier
from skopt import BayesSearchCV
import joblib
from sklearn.preprocessing import LabelEncoder
warnings.filterwarnings("ignore")
get_ipython().run_line_magic('matplotlib', 'inline')

fcsvcc="E:/fenqi/bestfea/5cerebellar_cerebral_report.csv"
fcsvce="E:/fenqi/bestfea/5cerebral_emg_report.csv"

fsv="E:/fenqi/compare/"
df=pd.DataFrame()
dfcc=pd.read_csv(fcsvcc, usecols=['Data', 'Sleep_Stage', 'Cata', 'Sample_number', 'Area'])
dfce=pd.read_csv(fcsvce, usecols=['Data', 'Sleep_Stage', 'Cata', 'Sample_number', 'Area'])
df=df.append(dfcc)
df=df.append(dfce)
sns.set_theme(style="whitegrid",font='Times New Roman',font_scale=2,rc={'font.weight': 'bold'})
zihao=36
dfzzz=df
colors=['#C49A98', '#7E4D99', '#3B84C4','#E6873E','#53A362','#DA3F34']
#presicion
g = sns.catplot(
    data=dfzzz[dfzzz['Cata']=="Precision"], x='Sample_number', y='Data', col="Sleep_Stage",hue="Area",
    palette=colors, errorbar="se",scale='width',
    kind="violin"
)

for ax in g.axes.flat:

    ax.set_ylabel('Precision rate', fontdict={'family': 'Times New Roman', 'size': zihao,'weight':'bold'})
    ax.set_yticks(np.arange(0.2,1.2,0.2))
    ax.set_ylim(0.1,1.19)

g._legend.set_title("")

g.set_titles(col_template='')
g.set(xlabel='')
g.set(xticklabels=[])
g.despine(left=True)

plt.savefig(fsv+'Precision.jpg',dpi=600)

#Recall
g = sns.catplot(
    data=dfzzz[dfzzz['Cata']=="Recall"], x='Sample_number', y='Data',  col="Sleep_Stage",hue="Area",
    palette=colors, errorbar="se",scale='width',
    kind="violin"
)

for ax in g.axes.flat:

    ax.set_ylabel('Recall/sensitivity rate', fontdict={'family': 'Times New Roman', 'size': zihao,'weight':'bold'})
    ax.set_yticks(np.arange(0.2,1.2,0.2))
    ax.set_ylim(0.1,1.19)

g._legend.set_title("")
g.set_titles(col_template='')
g.set(xlabel='')
g.set(xticklabels=[])
g.despine(left=True)

plt.savefig(fsv+'Recall_sensitivity.jpg',dpi=600)

#Specificity
g = sns.catplot(
    data=dfzzz[dfzzz['Cata']=="Specificity"], x='Sample_number', y='Data',  col="Sleep_Stage",hue="Area",
    palette=colors, errorbar="se",scale='width',
    kind="violin"
)


for ax in g.axes.flat:

    ax.set_ylabel('Specificity rate', fontdict={'family': 'Times New Roman', 'size': zihao,'weight':'bold'})
    ax.set_yticks(np.arange(0.2,1.2,0.2))
    ax.set_ylim(0.1,1.19)

g._legend.set_title("")
g.set_titles(col_template='')
g.set(xlabel='')
g.set(xticklabels=[])
g.despine(left=True)
plt.savefig(fsv+'Specificity.jpg',dpi=600)
from scipy.stats import ttest_ind

for stage in dfzzz['Sleep_Stage'].unique():
    for cata in dfzzz['Cata'].unique():
        data_stage_cata = dfzzz[(dfzzz['Sleep_Stage'] == stage) & (dfzzz['Cata'] == cata)]

        area1, area2 = dfzzz['Area'].unique()
        data_area1 = data_stage_cata[data_stage_cata['Area'] == area1]
        data_area2 = data_stage_cata[data_stage_cata['Area'] == area2]
        t_stat, p_val = ttest_ind(data_area1['Data'], data_area2['Data'])
        print(f"对于Sleep_Stage = {stage}, Cata = {cata}, Area1 = {area1} 和 Area2 = {area2}，T统计量: {t_stat}, P值: {p_val}")

dfz=pd.DataFrame()
for i in [5]:
    fcsv="E:/fenqi/bestfea/{}AUC.csv".format(i)
    target_names = ['Wakefulness', 'NREM', 'REM']
    Area=['Cerebellar EEG','Cerebral EEG','EMG']
    df=pd.read_csv(fcsv)
    df['Sample_number']=i
    dfz=dfz.append(df)
dfz= dfz.drop(df.columns[0], axis=1)    
sns.set_theme(style="whitegrid",font='Times New Roman',font_scale=1.0,rc={'font.weight': 'bold'})

colors=['#C49A98', '#7E4D99', '#3B84C4','#E6873E','#53A362','#DA3F34']
g = sns.catplot(
    data=dfz, x='Sleep Stage', y='AUC', hue='Area',
    palette=colors, errorbar="se",scale='width',
    kind="violin"
)

for ax in g.axes.flat:
    ax.set_xticklabels([])
    ax.set_xlabel('', fontdict={'family': 'Times New Roman', 'size': 18,'weight':'bold'})
    ax.set_ylabel('Mean AUC (%)', fontdict={'family': 'Times New Roman', 'size': 18,'weight':'bold'})
    ax.set_yticks(np.arange(0.6,1.1,0.2))
    ax.set_ylim(0.6,1.09)
    ax.tick_params(labelsize=14)

g._legend.set_title("")
g.set_titles(col_template='')
g.despine(left=True)
plt.savefig(fsv+'AUC.jpg',dpi=300,bbox_inches = 'tight')

from scipy.stats import ttest_ind


for stage in dfz['Sleep Stage'].unique():   
    data_stage_cata = dfz[(dfz['Sleep Stage'] == stage)]

    area1, area2 = dfz['Area'].unique()
    data_area1 = data_stage_cata[data_stage_cata['Area'] == area1]
    data_area2 = data_stage_cata[data_stage_cata['Area'] == area2]
    t_stat, p_val = ttest_ind(data_area1['AUC'], data_area2['AUC'])
    print(f"Sleep_Stage = {stage}, Area1 = {area1} 和 Area2 = {area2}，T: {t_stat}, P: {p_val}")


# In[ ]:




