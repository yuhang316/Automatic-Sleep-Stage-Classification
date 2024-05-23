#!/usr/bin/env python
# coding: utf-8

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
    eeg1=np.array(eegz[:,:30]).astype(np.float16)
    eeg2=np.array(eegz[:,30:60]).astype(np.float16)
    emg=np.array(eegz[:,60:90]).astype(np.float16)
    scores=np.array(eegz[:,90:91]).astype(np.float16)
    f.close()
    gc.collect()
    return emg,eeg1,eeg2,scores

plt.rcParams['font.sans-serif'] = ['SimHei']


def sen(Y_test,Y_pred,n):
    
    sen = []
    con_mat = confusion_matrix(Y_test,Y_pred)
    for i in range(n):
        tp = con_mat[i][i]
        fn = np.sum(con_mat[i,:]) - tp
        sen1 = tp / (tp + fn)
        sen.append(sen1)
        
    return sen
def spe(Y_test,Y_pred,n):
    
    spe = []
    con_mat = confusion_matrix(Y_test,Y_pred)
    for i in range(n):
        number = np.sum(con_mat[:,:])
        tp = con_mat[i][i]
        fn = np.sum(con_mat[i,:]) - tp
        fp = np.sum(con_mat[:,i]) - tp
        tn = number - tp - fn - fp
        spe1 = tn / (tn + fp)
        spe.append(spe1)
    
    return spe

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
    plt.savefig(f_sv+'{}'.format(name),dpi=300,bbox_inches = 'tight')
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
    fpr={}
    tpr={}
    roc_auc={}
    rocz=[]
    interp_tprf=[]
    pres=[]
    recal=[]
    sens=[]
    spec=[]
    for j in range(5):
        info=os.listdir(f_h5z)

        infoxl = np.random.choice(info,num, replace=False)
        infoyc =np.setdiff1d(info, infoxl)
        tab=[]
        eeg1f,eeg2f,emgf=[],[],[]
        for i in range(np.shape(infoxl)[0]):

            domain = os.path.abspath(f_h5z)

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
    return importance_df,interp_tprf,score,dfz,dfauc

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
def tcplot_feature(eeg2f,eeg2fyc,tab,tabyc,can,num):
    can=dict(can)

    ez=eeg2f
    ezyc=eeg2fyc
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
                           scale_pos_weight=can['scale_pos_weight'],n_estimators=can['n_estimators'],verbose=-1)

    model.fit(x_train, y_train)
    pred=model.predict(x_test)
    preds=model.predict_proba(x_test)
    score = model.score(x_test, y_test)
    target_names = ['Wakefulness', 'NREM', 'REM']

    y_test1 = label_binarize(y_test, classes=[0, 1, 2])
    eegx=list(i for i in np.arange(1,31))
    eegx = [str(x)+ ' Hz'for x in eegx]
    f_head2=[]
    f_head2.extend(list('Cerebellar EEG '+j for j in eegx))
    f_head2.extend(list('Cerebral EEG '+j for j in eegx))
    f_head2.extend(list('EMG '+j for j in eegx))  
    
    fpr={}
    tpr={}
    roc_auc={}
    rocz=[]
    interp_tprf=[]

    pres=[]
    recal=[]
    sens=[]
    spec=[]

    pres.append(precision_score(y_test, pred,average=None))
    recal.append(recall_score(y_test, pred,average=None))
    sens.append(sen(y_test, pred,3))
    spec.append(spe(y_test, pred,3))


    for i in range(3):

        fpr[i], tpr[i], _ = roc_curve(y_test1[:, i], preds[:, i])
        rocz.append(auc(fpr[i], tpr[i]))
        roc_auc[i] = auc(fpr[i], tpr[i])

        interp_tprff=np.interp(mean_fpr, fpr[i], tpr[i])
        interp_tprf.append(interp_tprff)
    df_rocz = pd.DataFrame(rocz, columns=['AUC'])
    df_sleep_stages = pd.DataFrame(target_names, columns=['Sleep Stage'])
    dfauc = pd.concat([df_sleep_stages, df_rocz], axis=1)    

    pres=np.array(pres)
    recal=np.array(recal)
    sens=np.array(sens)
    spec=np.array(spec)
    df=pd.DataFrame()
    df1=pd.DataFrame()
    df2=pd.DataFrame()
    df['Data']=pres[:,0]
    df['Sleep_Stage']=['Wakefulness']
    df1['Data']=pres[:,1]
    df1['Sleep_Stage']=['Nrem']
    df2['Data']=pres[:,2]
    df2['Sleep_Stage']=['Rem']
    dfj=pd.DataFrame()
    dfj=dfj._append(df)
    dfj=dfj._append(df1)
    dfj=dfj._append(df2)
    dfj['Cata']='Precision'
    dfj['Sample_number']=num

    
    df=pd.DataFrame()
    df1=pd.DataFrame()
    df2=pd.DataFrame()
    df['Data']=recal[:,0]
    df['Sleep_Stage']=['Wakefulness']
    df1['Data']=recal[:,1]
    df1['Sleep_Stage']=['Nrem']
    df2['Data']=recal[:,2]
    df2['Sleep_Stage']=['Rem']
    dfr=pd.DataFrame()
    dfr=dfr._append(df)
    dfr=dfr._append(df1)
    dfr=dfr._append(df2)
    dfr['Cata']='Recall'
    dfr['Sample_number']=num

    
    df=pd.DataFrame()
    df1=pd.DataFrame()
    df2=pd.DataFrame()
    df['Data']=sens[:,0]
    df['Sleep_Stage']=['Wakefulness']
    df1['Data']=sens[:,1]
    df1['Sleep_Stage']=['Nrem']
    df2['Data']=sens[:,2]
    df2['Sleep_Stage']=['Rem']
    dfm=pd.DataFrame()
    dfm=dfm._append(df)
    dfm=dfm._append(df1)
    dfm=dfm._append(df2)
    dfm['Cata']='Sensitivity'
    dfm['Sample_number']=num


    df=pd.DataFrame()
    df1=pd.DataFrame()
    df2=pd.DataFrame()
    df['Data']=spec[:,0]
    df['Sleep_Stage']=['Wakefulness']
    df1['Data']=spec[:,1]
    df1['Sleep_Stage']=['Nrem']
    df2['Data']=spec[:,2]
    df2['Sleep_Stage']=['Rem']
    dft=pd.DataFrame()
    dft=dft._append(df)
    dft=dft._append(df1)
    dft=dft._append(df2)
    dft['Cata']='Specificity'
    dft['Sample_number']=num

    
    dfz=pd.DataFrame()
    dfz=dfz._append(dfj)
    dfz=dfz._append(dfr)
    dfz=dfz._append(dfm)
    dfz=dfz._append(dft)

    return model.feature_importances_,f_head2,interp_tprf,score,dfz,dfauc

num=15

df=pd.DataFrame()
f_h5z="E:/fenqi/h5zz"
f_sv="E:/fenqi/bestfea/"
mean_fpr = np.linspace(0, 1, 100)

interp_tprall=[]
scoreall=[]
dfall=pd.DataFrame()
dfaall=pd.DataFrame()   
impo=[]
fh=[]
for j in range(30):
    info=os.listdir(f_h5z)

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
    impo_,fh_,interp_tprf,score,dfz,dfauc=tcplot_feature(ez,ezyc,tab,tabyc,bcanshu,num)
    impo.extend(impo_)
    fh.extend(fh_)
    interp_tprall.append(interp_tprf)
    scoreall.append(score)
    dfall=dfall._append(dfz)
    dfaall= dfaall._append(dfauc)

print("ni",dfall)
print("ni",dfaall)
dfaall['Area']="Cerebellar EEG&Cerebral EEG&EMG"
dfaall.to_csv(f_sv+"{}AUC.csv".format(num))  

fill_plot(interp_tprall,"{}all".format(num),scoreall)
dfall.to_csv(f_sv+"{}Cerebellar EEG&Cerebral EEG&EMG_report.csv".format(num))
importance_df = pd.DataFrame({'Feature':fh, 'Importance': impo})    
importance_df.to_csv(f_sv+"feat_impo.csv")
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

df=pd.read_csv('E:/fenqi/bestfea/feat_impo.csv')
importance_df=df

mean_df = importance_df.groupby('Feature')['Importance'].mean()

importance_df['mean_importance'] = importance_df['Feature'].map(mean_df)

top_20_features = mean_df.nlargest(20).index

filtered_df = importance_df[importance_df['Feature'].isin(top_20_features)]

sorted_df = filtered_df.sort_values(by='mean_importance', ascending=False)
sorted_df = sorted_df.drop(columns=['mean_importance'])
sorted_df.to_csv(f_sv+'top20output.csv', index=False)

mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['font.size'] = 38
mpl.rcParams['font.weight'] = 'bold'

plt.figure(figsize=(40, 16))
sns.violinplot(x='Feature', y='Importance', data=sorted_df, inner=None)
sns.swarmplot(x='Feature', y='Importance', data=sorted_df, color='k', size=10)
plt.ylabel('Feature Importance', fontdict={'family': 'Times New Roman', 'size': 42, 'weight': 'bold'})
plt.xlabel('Feature name', fontdict={'family': 'Times New Roman', 'size': 42, 'weight': 'bold'})
plt.xticks(rotation=70)
plt.savefig(f_sv+'fimpo.jpg', dpi=600, bbox_inches = 'tight')
plt.show()

