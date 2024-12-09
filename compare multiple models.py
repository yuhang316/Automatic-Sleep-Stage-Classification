#!/usr/bin/env python
# coding: utf-8

# In[29]:


import sys
sys.path.insert(0, 'D:/pythonbao/')
import pandas as pd
import numpy as np
import os
from pandas import DataFrame
import matplotlib.pyplot as plt
import h5py
import gc
import seaborn as sns
import warnings
from scipy import signal
from sklearn.preprocessing import label_binarize
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split,GridSearchCV,LeaveOneOut,cross_val_score,StratifiedKFold
from itertools import cycle
from sklearn.metrics import f1_score,classification_report,confusion_matrix, precision_score,recall_score,roc_auc_score, roc_curve, auc, precision_recall_curve
import scipy
from scipy.stats import pearsonr,spearmanr
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import joblib
import lightgbm as lgb
from catboost import CatBoostClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout,Conv1D, MaxPooling1D, Flatten
from keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sleep_fenqi_bao import readfilez,tiaocanlgb,canshulgb,tcplotlgb,xunlianlgb,tiaocanxgb,canshuxgb,tcplotxgb,xunlianxgb,tiaocancat,canshucat,tcplotcat,sen,spe,fill_plot,xl_all
import time
from imblearn.under_sampling import RandomUnderSampler
warnings.filterwarnings("ignore")
get_ipython().run_line_magic('matplotlib', 'inline')


def f_head(name,model):
    importa,f_head2=[],[]
    eegx=list(i for i in np.arange(0,30))
    eegx = [str(x) for x in eegx]
    EEG_head=[name]
    for i in EEG_head:
#         f_head2.extend(list(i+'Cerebellar EEG'+j for j in eegx))
        f_head2.extend(list(i+'Cerebral EEG'+j for j in eegx))
        f_head2.extend(list(i+'EMG'+j for j in eegx))
    impo=pd.Series(model.feature_importances_, index=f_head2)
    importa.append(impo)
    df1=pd.DataFrame(importa)    
    return df1
    
def tcplotcnn(eeg2f, eeg2fyc, tab, tabyc, cnncan, name):
    training_time = []
    time1 = time.time()
    
    # 获取CNN参数
    filters, kernel_size, pool_size, epochs, batch_size, learning_rate, dropout_rate = cnncan
    
    # 数据预处理
    le = LabelEncoder()
    tab = le.fit_transform(tab)
    tabyc = le.transform(tabyc)  # 注意这里使用transform而不是fit_transform
    ohe = OneHotEncoder(sparse_output=False)
    y_train = ohe.fit_transform(tab.reshape(-1, 1))
    y_test = ohe.transform(tabyc.reshape(-1, 1)) 
    print(np.shape(eeg2f))
    x_train = np.array(eeg2f).reshape(eeg2f.shape[0], 30, 2)
    x_test = np.array(eeg2fyc).reshape(eeg2fyc.shape[0], 30, 2)

    # 创建CNN模型
    model = Sequential([
        Conv1D(filters=filters, kernel_size=kernel_size, activation='relu', input_shape=(30, 2)),
        MaxPooling1D(pool_size=pool_size),
        Flatten(),
        Dropout(dropout_rate),
        Dense(3, activation='softmax')  # 三分类问题
    ])
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    # 训练模型
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

    # 预测
    preds = model.predict(x_test)
    pred = np.argmax(preds, axis=1)  # 获取最大概率对应的类别索引
    score = model.evaluate(x_test, y_test, verbose=0)[1]  # 获取准确率

    time2 = time.time()
    training_time.append(time2 - time1)

    df1 = []  # 如果有额外的头部信息需要添加，请在这里定义
    interp_tprf, df = plt_featureltsm(y_test, pred, preds, name)  # 使用原有的函数绘制ROC曲线
    df2 = pd.DataFrame(training_time, columns=['Training Time (s)'])
    return interp_tprf, score, df, df1, df2
    
def tcplotltsm(eeg2f, eeg2fyc, tab, tabyc, ltsmcan, name):
    training_time = []
    time1 = time.time()
    
    # 获取LSTM参数
    units, epochs, batch_size,learning_rate,dropout_rate= ltsmcan
    
    # 数据预处理
    le = LabelEncoder()
    tab = le.fit_transform(tab)
    tabyc = le.transform(tabyc)  # 注意这里使用transform而不是fit_transform
    # x_train,y_train=np.array(ez),np.array(tab)
    # x_test,y_test=np.array(ezyc),np.array(tabyc)    
    # 将标签转换为one-hot编码

    ohe = OneHotEncoder(sparse_output=False)
    y_train = ohe.fit_transform(tab.reshape(-1, 1))
    y_test = ohe.transform(tabyc.reshape(-1, 1)) 
    print(np.shape(eeg2f))
    x_train = np.array(eeg2f.reshape(eeg2f.shape[0], 30, 2))
    x_test = np.array(eeg2fyc.reshape(eeg2fyc.shape[0], 30, 2))

    # 确保x_train和x_test是3D数组
    if len(x_train.shape) == 2:
        x_train = np.expand_dims(x_train, axis=-1)
        x_test = np.expand_dims(x_test, axis=-1)

    # 创建LSTM模型
    model = Sequential([
        LSTM(units, input_shape=(x_train.shape[1], x_train.shape[2]), return_sequences=False),
        Dropout(dropout_rate),
        
        Dense(3, activation='softmax')  # 三分类问题
    ])
    optimizer = Adam(learning_rate=learning_rate)
    # 编译模型
    model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    # 训练模型
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

    # 预测
    preds = model.predict(x_test)
    pred = np.argmax(preds, axis=1)  # 获取最大概率对应的类别索引
    score = model.evaluate(x_test, y_test, verbose=0)[1]  # 获取准确率

    time2 = time.time()
    training_time.append(time2 - time1)

    # df1 = f_head(name, model)
    df1 = []
    interp_tprf, df = plt_featureltsm(y_test, pred, preds, name)
    df2 = pd.DataFrame(training_time)
    return interp_tprf, score, df, df1, df2

    
def tcplotcat(eeg2f,eeg2fyc,tab,tabyc,catcan,name):
    training_time=[]
    time1 = time.time()
    it,lr,dh=catcan
    #大脑
    ez=eeg2f
    ezyc=eeg2fyc
    le = LabelEncoder()
    tab = le.fit_transform(tab)
    tabyc = le.fit_transform(tabyc)
    x_train,y_train=np.array(ez),np.array(tab)
    x_test,y_test=np.array(ezyc),np.array(tabyc)

    # model = svm.SVC(probability=True,C=0.4,gamma=0.2,  kernel='rbf')#默认参数
    model = CatBoostClassifier(iterations=it,learning_rate=lr,depth=dh,task_type='GPU',logging_level='Silent')

    model.fit(x_train, y_train)
    # x_test, y_test = ros.fit_resample(x_test, y_test)
    pred=model.predict(x_test)
    preds=model.predict_proba(x_test)
    score = model.score(x_test, y_test)
    
    time2 = time.time()
    training_time.append(time2 - time1  )  
#     print(y_test)
#     print(pred)
    # print(preds)
    df1=f_head(name,model)
    
#     print(y_test)
#     print(pred)
    # print(preds)
    interp_tprf,df=plt_feature(y_test, pred,preds,name)
    df2=pd.DataFrame(training_time)
    return interp_tprf,score,df,df1,df2


def tcplotxgb(eeg2f,eeg2fyc,tab,tabyc,xgbcan,name):
    training_time=[]
    time1 = time.time()
    ra,rl,md,lr,ga,mw,ns,mp,ce,se=xgbcan
    #大脑
    ez=eeg2f
    ezyc=eeg2fyc
    le = LabelEncoder()
    tab = le.fit_transform(tab)
    tabyc = le.fit_transform(tabyc)
    x_train,y_train=np.array(ez),np.array(tab)
    x_test,y_test=np.array(ezyc),np.array(tabyc)


    # model = svm.SVC(probability=True,C=0.4,gamma=0.2,  kernel='rbf')#默认参数
    model = xgb.XGBClassifier(objective='multi:softmax',num_class=3,learning_rate= lr, n_estimators=ns, max_depth= md, min_child_weight= mw,
                            subsample= se, colsample_bytree= ce, gamma= ga,max_delta_step=md,reg_alpha=ra,reg_lambda=rl,tree_method='gpu_hist')

    model.fit(x_train, y_train)
    # x_test, y_test = ros.fit_resample(x_test, y_test)
    pred=model.predict(x_test)
    preds=model.predict_proba(x_test)
    score = model.score(x_test, y_test)
#     print(y_test)
#     print(pred)
    # print(preds)
    time2 = time.time()
    training_time.append(time2 - time1)
    
    df1=f_head(name,model)
    
    interp_tprf,df=plt_feature(y_test, pred,preds,name)
    df2=pd.DataFrame(training_time)
    return interp_tprf,score,df,df1,df2

def tcplotlgb(eeg2f,eeg2fyc,tab,tabyc,lgbcan,name):
    training_time=[]
    time1 = time.time()

    nl,lr,ns,md,ss,cb,ra,rl=lgbcan
    #大脑
    ez=eeg2f
    ezyc=eeg2fyc
    le = LabelEncoder()
    tab = le.fit_transform(tab)
    tabyc = le.fit_transform(tabyc)
    x_train,y_train=np.array(ez),np.array(tab)
    x_test,y_test=np.array(ezyc),np.array(tabyc)
    model = lgb.LGBMClassifier(num_leaves=nl,learning_rate= lr, n_estimators=ns, max_depth= md,
                            subsample= ss, colsample_bytree= cb,reg_alpha=ra,reg_lambda=rl,verbosity= -1)

    model.fit(x_train, y_train)
    # x_test, y_test = ros.fit_resample(x_test, y_test)
    pred=model.predict(x_test)
    preds=model.predict_proba(x_test)
    score = model.score(x_test, y_test)
    
    time2 = time.time()
    training_time.append(time2 - time1) 
    
    df1=f_head(name,model)
    
#     print(y_test)
#     print(pred)
    # print(preds)
    interp_tprf,df=plt_feature(y_test, pred,preds,name)
    df2=pd.DataFrame(training_time)
    return interp_tprf,score,df,df1,df2

def tcplotsvcm(eeg2f,eeg2fyc,tab,tabyc,name):
    from sklearn import svm
    training_time=[]
    time1 = time.time()
    #大脑
    ez=eeg2f
    ezyc=eeg2fyc
    le = LabelEncoder()
    tab = le.fit_transform(tab)
    tabyc = le.fit_transform(tabyc)
    x_train,y_train=np.array(ez),np.array(tab)
    x_test,y_test=np.array(ezyc),np.array(tabyc)
    model = svm.SVC(probability=True,decision_function_shape='ovr',kernel='rbf')
    model.fit(x_train, y_train)
    # x_test, y_test = ros.fit_resample(x_test, y_test)
    pred=model.predict(x_test)
    preds=model.predict_proba(x_test)
    score = model.score(x_test, y_test)   
    time2 = time.time()
    training_time.append(time2 - time1) 
    df1=pd.DataFrame()
#     print(y_test)
#     print(pred)
    # print(preds)
    interp_tprf,df=plt_nofeature(y_test, pred,preds,name)
    df2=pd.DataFrame(training_time)
    return interp_tprf,score,df,df1,df2

def tcplotlr(eeg2f,eeg2fyc,tab,tabyc,name):
    from sklearn.linear_model import LogisticRegression
    training_time=[]
    time1 = time.time()
    #大脑
    ez=eeg2f
    ezyc=eeg2fyc
    le = LabelEncoder()
    tab = le.fit_transform(tab)
    tabyc = le.fit_transform(tabyc)
    x_train,y_train=np.array(ez),np.array(tab)
    x_test,y_test=np.array(ezyc),np.array(tabyc)
    model = LogisticRegression(multi_class='ovr')
    model.fit(x_train, y_train)
    # x_test, y_test = ros.fit_resample(x_test, y_test)
    pred=model.predict(x_test)
    preds=model.predict_proba(x_test)
    score = model.score(x_test, y_test)   
    time2 = time.time()
    training_time.append(time2 - time1) 
    df1=pd.DataFrame()
#     print(y_test)
#     print(pred)
    # print(preds)
    interp_tprf,df=plt_nofeature(y_test, pred,preds,name)
    df2=pd.DataFrame(training_time)
    return interp_tprf,score,df,df1,df2

def tcplotrf(eeg2f,eeg2fyc,tab,tabyc,name):
    from sklearn.ensemble import RandomForestClassifier
    training_time=[]
    time1 = time.time()
    #大脑
    ez=eeg2f
    ezyc=eeg2fyc
    le = LabelEncoder()
    tab = le.fit_transform(tab)
    tabyc = le.fit_transform(tabyc)
    x_train,y_train=np.array(ez),np.array(tab)
    x_test,y_test=np.array(ezyc),np.array(tabyc)
    model = RandomForestClassifier(n_estimators=300)
    model.fit(x_train, y_train)
    # x_test, y_test = ros.fit_resample(x_test, y_test)
    pred=model.predict(x_test)
    preds=model.predict_proba(x_test)
    score = model.score(x_test, y_test)   
    time2 = time.time()
    training_time.append(time2 - time1) 
    df1=pd.DataFrame()
#     print(y_test)
#     print(pred)
    # print(preds)
    interp_tprf,df=plt_nofeature(y_test, pred,preds,name)
    df2=pd.DataFrame(training_time)
    return interp_tprf,score,df,df1,df2

def plt_featureltsm(y_test, pred,preds,name):
    target_names = ['Wake', 'NREM', 'REM']
    y_test_indices = np.argmax(y_test, axis=1)
    y_test1 = label_binarize(y_test_indices, classes=[0, 1, 2])

    fpr={}
    tpr={}
    roc_auc={}
    rocz=[]
    f_head2=[]
    interp_tprf=[]

    pres=[]
    recal=[]
    sens=[]
    spec=[]
    
    pres.append(precision_score(y_test_indices, pred,average=None))
    recal.append(recall_score(y_test_indices, pred,average=None))
    sens.append(sen(y_test_indices, pred,3))
    spec.append(spe(y_test_indices, pred,3))


    for i in range(3):
    #         print(i)
        fpr[i], tpr[i], _ = roc_curve(y_test1[:, i], preds[:, i])
        rocz.append(auc(fpr[i], tpr[i]))
        roc_auc[i] = auc(fpr[i], tpr[i])
#         print("roc_auc:",roc_auc)
        interp_tprff=np.interp(mean_fpr, fpr[i], tpr[i])
        interp_tprf.append(interp_tprff)
    # Plot all ROC curves
    plt.figure()
    lw = 2
    colors = cycle(['darkorange','cornflowerblue','DeepPink'])
    for i, color in zip(range(3), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of {0} (area = {1:0.2f})'
                 ''.format(target_names[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.show()
    pres=np.array(pres)
    recal=np.array(recal)
    sens=np.array(sens)
    spec=np.array(spec)
    df=pd.DataFrame({'精确率Wakefulness':pres[:,0],'精确率Nrem':pres[:,1],'精确率Rem':pres[:,2],
                     '召回率Wakefulness':recal[:,0],'召回率Nrem':recal[:,1],'召回率Rem':recal[:,2],
                     '敏感性Wakefulness':sens[:,0],'敏感性Nrem':sens[:,1],'敏感性Rem':sens[:,2],
                     '特异性Wakefulness':spec[:,0],'特异性Nrem':spec[:,1],'特异性Rem':spec[:,2]})
#     df.to_excel(f_sv+"{}report.xls".format(name))
    return interp_tprf,df

def plt_feature(y_test, pred,preds,name):
    target_names = ['Wake', 'NREM', 'REM']

    y_test1 = label_binarize(y_test, classes=[0, 1, 2])

    fpr={}
    tpr={}
    roc_auc={}
    rocz=[]
    f_head2=[]
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
    #         print(i)
        fpr[i], tpr[i], _ = roc_curve(y_test1[:, i], preds[:, i])
        rocz.append(auc(fpr[i], tpr[i]))
        roc_auc[i] = auc(fpr[i], tpr[i])
#         print("roc_auc:",roc_auc)
        interp_tprff=np.interp(mean_fpr, fpr[i], tpr[i])
        interp_tprf.append(interp_tprff)
    # Plot all ROC curves
    plt.figure()
    lw = 2
    colors = cycle(['darkorange','cornflowerblue','DeepPink'])
    for i, color in zip(range(3), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of {0} (area = {1:0.2f})'
                 ''.format(target_names[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.show()
    pres=np.array(pres)
    recal=np.array(recal)
    sens=np.array(sens)
    spec=np.array(spec)
    df=pd.DataFrame({'精确率Wakefulness':pres[:,0],'精确率Nrem':pres[:,1],'精确率Rem':pres[:,2],
                     '召回率Wakefulness':recal[:,0],'召回率Nrem':recal[:,1],'召回率Rem':recal[:,2],
                     '敏感性Wakefulness':sens[:,0],'敏感性Nrem':sens[:,1],'敏感性Rem':sens[:,2],
                     '特异性Wakefulness':spec[:,0],'特异性Nrem':spec[:,1],'特异性Rem':spec[:,2]})
#     df.to_excel(f_sv+"{}report.xls".format(name))
    return interp_tprf,df

def plt_nofeature(y_test, pred,preds,name):
    target_names = ['Wake', 'NREM', 'REM']

    y_test1 = label_binarize(y_test, classes=[0, 1, 2])

    fpr={}
    tpr={}
    roc_auc={}
    rocz=[]

    interp_tprf=[]
    pres=[]
    recal=[]
    sens=[]
    spec=[]
    importa=[]
    pres.append(precision_score(y_test, pred,average=None))
    recal.append(recall_score(y_test, pred,average=None))
    sens.append(sen(y_test, pred,3))
    spec.append(spe(y_test, pred,3))
#     impo=pd.Series(model.feature_importances_, index=f_head2)
#     importa.append(impo)    
    
    for i in range(3):
    #         print(i)
        fpr[i], tpr[i], _ = roc_curve(y_test1[:, i], preds[:, i])
        rocz.append(auc(fpr[i], tpr[i]))
        roc_auc[i] = auc(fpr[i], tpr[i])
#         print("roc_auc:",roc_auc)
        interp_tprff=np.interp(mean_fpr, fpr[i], tpr[i])
        interp_tprf.append(interp_tprff)
    # Plot all ROC curves
    plt.figure()
    lw = 2
    colors = cycle(['darkorange','cornflowerblue','DeepPink'])
    for i, color in zip(range(3), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of {0} (area = {1:0.2f})'
                 ''.format(target_names[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.show()
    pres=np.array(pres)
    recal=np.array(recal)
    sens=np.array(sens)
    spec=np.array(spec)
    df=pd.DataFrame({'精确率Wakefulness':pres[:,0],'精确率Nrem':pres[:,1],'精确率Rem':pres[:,2],
                     '召回率Wakefulness':recal[:,0],'召回率Nrem':recal[:,1],'召回率Rem':recal[:,2],
                     '敏感性Wakefulness':sens[:,0],'敏感性Nrem':sens[:,1],'敏感性Rem':sens[:,2],
                     '特异性Wakefulness':spec[:,0],'特异性Nrem':spec[:,1],'特异性Rem':spec[:,2]})
#     df.to_excel(f_sv+"{}report.xls".format(name))
#     df1=pd.DataFrame(importa)
#     df1.to_csv(f_sv+"{}importa.csv".format(name))
    return interp_tprf,df


#求解敏感性、特异性
plt.rcParams['font.sans-serif'] = ['SimHei']
f_h5z="E:/fenqi/h5zz"
f_sv="E:/fenqi/daxiu"
mean_fpr = np.linspace(0, 1, 100)
info=os.listdir(f_h5z)
print(info)


# In[30]:


lgb_nl,lgb_lr,lgb_ns,lgb_md,lgb_ss,lgb_cb,lgb_ra,lgb_rl=28,0.29000000000000004,900,3,0.7,0.9000000000000001,0.30000000000000004,0.1
xgb_ra,xgb_rl,xgb_md,xgb_lr,xgb_ga,xgb_mw,xgb_ns,xgb_mp,xgb_ce,xgb_se=0.1,0.2,7,0.29000000000000004,0.30000000000000004,2,240,1,0.6000000000000001,0.8999999999999999

lgbcan=28,0.29000000000000004,500,3,0.7,0.9000000000000001,0.30000000000000004,0.1
xgbcan=0.1,0.2,7,0.29000000000000004,0.30000000000000004,2,500,1,0.6000000000000001,0.8999999999999999
catcan=500,0.21,7
ltsmcan=64,50,32,0.001,0.0
cnncan=2024,4,4,50,32,0.001,0.0
num=6#训练文件数量
xgb_ra,xgb_rl,xgb_md,xgb_lr,xgb_ga,xgb_mw,xgb_ns,xgb_mp,xgb_ce,xgb_se=xgbcan
lgb_nl,lgb_lr,lgb_ns,lgb_md,lgb_ss,lgb_cb,lgb_ra,lgb_rl=lgbcan
cat_it,cat_lr,cat_dh=catcan

interp_tprd_cnn=[]
dfd_ltsm,dfd1_cnn=[],[]
scored_cnn=[]
dfd_cnn=pd.DataFrame()
dfd1_cnn=pd.DataFrame()  
dfd2_cnn=pd.DataFrame()  

interp_tprd_ltsm=[]
dfd_ltsm,dfd1_ltsm=[],[]
scored_ltsm=[]
dfd_ltsm=pd.DataFrame()
dfd1_ltsm=pd.DataFrame()  
dfd2_ltsm=pd.DataFrame()  

interp_tprd_lgb=[]
dfd_lgb,dfd1_lgb=[],[]
scored_lgb=[]
dfd_lgb=pd.DataFrame()
dfd1_lgb=pd.DataFrame()  
dfd2_lgb=pd.DataFrame()  

interp_tprd_xgb=[]
dfd_xgb,dfd1_xgb=[],[]
scored_xgb=[]
dfd_xgb=pd.DataFrame()
dfd1_xgb=pd.DataFrame()
dfd2_xgb=pd.DataFrame()

interp_tprd_cat=[]
dfd_cat,dfd1_cat=[],[]
scored_cat=[]
dfd_cat=pd.DataFrame()
dfd1_cat=pd.DataFrame()
dfd2_cat=pd.DataFrame()

interp_tprd_svc=[]
dfd_svc,dfd1_svc=[],[]
scored_svc=[]
dfd_svc=pd.DataFrame()
dfd1_svc=pd.DataFrame()
dfd2_svc=pd.DataFrame()

interp_tprd_lr=[]
dfd_lr,dfd1_lr=[],[]
scored_lr=[]
dfd_lr=pd.DataFrame()
dfd1_lr=pd.DataFrame()
dfd2_lr=pd.DataFrame()

interp_tprd_rf=[]
dfd_rf,dfd1_rf=[],[]
scored_rf=[]
dfd_rf=pd.DataFrame()
dfd1_rf=pd.DataFrame()
dfd2_rf=pd.DataFrame()


ez,ezyc=[],[]
tab,tabyc=[],[]

for j in range(30):
    info=os.listdir(f_h5z)
#     print(info)
    infoxl = np.random.choice(info,num, replace=False)
    infoyc =np.setdiff1d(info, infoxl)
    tab=[]
    eeg1f,eeg2f,emgf=[],[],[]
    for i in range(np.shape(infoxl)[0]):
        print("xxxxxxxxxx")
        print("woshi:",infoxl[i])
        print("xxxxxxxxxx")
        domain = os.path.abspath(f_h5z)
        infor = os.path.join(domain,infoxl[i])
#             print(infor)
        emg,eeg1,eeg2,scores=readfilez(infor)
#         eeg1f.extend(eeg1)
        eeg2f.extend(eeg2)
        emgf.extend(emg)
        tab.extend(scores)
#     ez= np.concatenate((eeg1f, eeg2f, emgf), axis=1)
    ez= np.concatenate((eeg2f, emgf), axis=1)
#     ez=np.array(eeg2f)

    tab=np.array(tab) 
    # 创建RandomUnderSampler实例
    rus = RandomUnderSampler(sampling_strategy={1: 500,2: 250,3: 75,},random_state=6)
    # 使用RandomUnderSampler进行降采样
    ez, tab = rus.fit_resample(ez, tab)
    
    #预测数据
#     print(infoyc)
    tabyc=[]
    eeg1fyc,eeg2fyc,emgfyc=[],[],[]

    for i in range(np.shape(infoyc)[0]):
#             print(info[i])
#         print("xxxxxxxxxx")
        domain = os.path.abspath(f_h5z)
        infor = os.path.join(domain,info[i])
        emg,eeg1,eeg2,scores=readfilez(infor)
#         eeg1fyc.extend(eeg1)
        eeg2fyc.extend(eeg2)
        emgfyc.extend(emg)
        tabyc.extend(scores)
#     ezyc= np.concatenate((eeg1fyc, eeg2fyc, emgfyc), axis=1)   
    ezyc= np.concatenate((eeg2fyc, emgfyc), axis=1) 
#     ezyc=np.array(eeg2fyc)
    tabyc=np.array(tabyc)  
#     print("herewego")
#     print(np.shape(eeg1f))
    tabyc=np.array(tabyc) 
    # 创建RandomUnderSampler实例
    rus = RandomUnderSampler(sampling_strategy={1: 2000,2: 1000,3: 200,},random_state=6)
    # 使用RandomUnderSampler进行降采样
    ezyc, tabyc = rus.fit_resample(ezyc, tabyc)
    
    #大脑
    interp_tprl,score,df,df1,df2=tcplotcnn(ez,ezyc,tab,tabyc,cnncan,"CNN")
    interp_tprd_cnn.append(interp_tprl)
    scored_cnn.append(score)
    dfd_cnn= pd.concat([dfd_cnn, df])
    # dfd1_cnn= pd.concat([dfd1_cnn, df1])
    dfd2_cnn= pd.concat([dfd2_cnn, df2])
    
    interp_tprl,score,df,df1,df2=tcplotltsm(ez,ezyc,tab,tabyc,ltsmcan,"LTSM")
    interp_tprd_ltsm.append(interp_tprl)
    scored_ltsm.append(score)
    dfd_ltsm= pd.concat([dfd_ltsm, df])
    # dfd1_ltsm= pd.concat([dfd1_ltsm, df1])
    dfd2_ltsm= pd.concat([dfd2_ltsm, df2])

    
    interp_tprl,score,df,df1,df2=tcplotxgb(ez,ezyc,tab,tabyc,xgbcan,"XGBoost")        
    interp_tprd_xgb.append(interp_tprl)
    scored_xgb.append(score)
    dfd_xgb= pd.concat([dfd_xgb, df])
    dfd1_xgb= pd.concat([dfd1_xgb, df1])
    dfd2_xgb= pd.concat([dfd2_xgb, df2])
#     print("wo",dfd_xgb)

    interp_tprl,score,df,df1,df2=tcplotlgb(ez,ezyc,tab,tabyc,lgbcan,"LGBoost")
    interp_tprd_lgb.append(interp_tprl)
    scored_lgb.append(score)
    dfd_lgb= pd.concat([dfd_lgb, df])
    dfd1_lgb= pd.concat([dfd1_lgb, df1])
    dfd2_lgb= pd.concat([dfd2_lgb, df2])
#     print("wo",dfd_lgb)

    interp_tprl,score,df,df1,df2=tcplotcat(ez,ezyc,tab,tabyc,catcan,"CATBoost")
    interp_tprd_cat.append(interp_tprl)
    scored_cat.append(score)
    dfd_cat= pd.concat([dfd_cat, df])
    dfd1_cat= pd.concat([dfd1_cat, df1])
    dfd2_cat= pd.concat([dfd2_cat, df2])
    
    
    interp_tprl,score,df,df1,df2=tcplotsvcm(ez,ezyc,tab,tabyc,"SVM")
    interp_tprd_svc.append(interp_tprl)
    scored_svc.append(score)
    dfd_svc= pd.concat([dfd_svc, df])
    dfd1_svc= pd.concat([dfd1_svc, df1])
    dfd2_svc= pd.concat([dfd2_svc, df2])
    
    interp_tprl,score,df,df1,df2=tcplotlr(ez,ezyc,tab,tabyc,"LogisticRegression")
    interp_tprd_lr.append(interp_tprl)
    scored_lr.append(score)
    dfd_lr= pd.concat([dfd_lr, df])
    dfd1_lr= pd.concat([dfd1_lr, df1])
    dfd2_lr= pd.concat([dfd2_lr, df2])   
    
    interp_tprl,score,df,df1,df2=tcplotrf(ez,ezyc,tab,tabyc,"RandomForest")
    interp_tprd_rf.append(interp_tprl)
    scored_rf.append(score)
    dfd_rf= pd.concat([dfd_rf, df])
    dfd1_rf= pd.concat([dfd1_rf, df1])
    dfd2_rf= pd.concat([dfd2_rf, df2])     
    
    
#     print("wo",dfd_lgb)        
#         df=pd.DataFrame(df)
#         df1=pd.DataFrame(df1)
dfd_cnn.to_csv(f_sv+"{}CNN_report.csv".format(num),encoding='utf_8_sig')
# dfd1_cnn.to_csv(f_sv+"{}CNN_importa.csv".format(num),encoding='utf_8_sig')
dfd2_cnn.to_csv(f_sv+"{}CNN_time.csv".format(num),encoding='utf_8_sig')

dfd_ltsm.to_csv(f_sv+"{}LTSM_report.csv".format(num),encoding='utf_8_sig')
# dfd1_ltsmltsm.to_csv(f_sv+"{}LTSM_importa.csv".format(num),encoding='utf_8_sig')
dfd2_ltsm.to_csv(f_sv+"{}LTSM_time.csv".format(num),encoding='utf_8_sig')

dfd_lgb.to_csv(f_sv+"{}LGBoost_report.csv".format(num),encoding='utf_8_sig')
dfd1_lgb.to_csv(f_sv+"{}LGBoost_importa.csv".format(num),encoding='utf_8_sig')
dfd2_lgb.to_csv(f_sv+"{}LGBoost_time.csv".format(num),encoding='utf_8_sig')

dfd_xgb.to_csv(f_sv+"{}XGBoost_report.csv".format(num),encoding='utf_8_sig')
dfd1_xgb.to_csv(f_sv+"{}XGBoost_importa.csv".format(num),encoding='utf_8_sig')
dfd2_xgb.to_csv(f_sv+"{}XGBoost_time.csv".format(num),encoding='utf_8_sig')

dfd_cat.to_csv(f_sv+"{}CATBoost_report.csv".format(num),encoding='utf_8_sig')
dfd1_cat.to_csv(f_sv+"{}CATBoost_importa.csv".format(num),encoding='utf_8_sig') 
dfd2_cat.to_csv(f_sv+"{}CATBoost_time.csv".format(num),encoding='utf_8_sig') 

dfd_svc.to_csv(f_sv+"{}SVM_report.csv".format(num),encoding='utf_8_sig')
dfd1_svc.to_csv(f_sv+"{}SVM_importa.csv".format(num),encoding='utf_8_sig') 
dfd2_svc.to_csv(f_sv+"{}SVM_time.csv".format(num),encoding='utf_8_sig') 

dfd_lr.to_csv(f_sv+"{}LogisticRegression_report.csv".format(num),encoding='utf_8_sig')
dfd1_lr.to_csv(f_sv+"{}LogisticRegression_importa.csv".format(num),encoding='utf_8_sig') 
dfd2_lr.to_csv(f_sv+"{}LogisticRegression_time.csv".format(num),encoding='utf_8_sig') 

dfd_rf.to_csv(f_sv+"{}RandomForest_report.csv".format(num),encoding='utf_8_sig')
dfd1_rf.to_csv(f_sv+"{}RandomForest_importa.csv".format(num),encoding='utf_8_sig') 
dfd2_rf.to_csv(f_sv+"{}RandomForest_time.csv".format(num),encoding='utf_8_sig') 

#     fill_plot_all(interp_tprd_lgb,"{}LGBoost".format(num),scored_lgb,interp_tprd_xgb,"{}XGBoost".format(num),scored_xgb,f_sv)


# In[31]:


def fill_plotm(interp_tpr,aucs,path):  


    mean_fpr = np.linspace(0, 1, 100)
    colors = ['#C49A98', '#7E4D99', '#3B84C4','#E6873E','#53A362','#DA3F34','#FF99CC','#8A2BE2']
    target_names = ['LGBoost', 'XGBoost','CatBoost','SVM','LR','RF','LSTM','CNN']
    print(np.shape(interp_tpr))
#     figsize=(8, 6)
#     fig, ax = plt.subplots(1, 3,figsize=(8, 6))
    
#     fig, ax = plt.subplots(1, 3,figsize=(21, 7))  # 修改为两行两列
    for j in range(3):
        fig, ax = plt.subplots(figsize=(8, 6))
        print(np.shape(interp_tpr)[0])
        ax.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)") 
        for i in range(np.shape(interp_tpr)[0]):
            interp_tpr_tmp=np.array(interp_tpr[i])
            mean_tpr = np.mean(interp_tpr_tmp[:,j,:], axis=0)
            mean_tpr[-1] = 1.0
            std_tpr = np.std(interp_tpr_tmp[:,j,:], axis=0)
            mean_auc = auc(mean_fpr, mean_tpr)
            std_auc = np.std(aucs)
            tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
            tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
            ax.plot(
                mean_fpr,
                mean_tpr,
                color=colors[i],
                label=f"Mean ROC for {target_names[i]}(AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
                lw=2,
                alpha=0.8,
            )

        ax.set(
            xlim=[-0.01, 1.00],
            ylim=[-0.00, 1.01],
        )

        ax.set_xlabel("False Positive Rate (1 - Specificity)", fontdict={'family': 'Times New Roman', 'size': 26,'weight':'bold'})
        ax.set_ylabel("True Positive Rate (Sensitivity)", fontdict={'family': 'Times New Roman', 'size': 26,'weight':'bold'})
    #     ax.set_title("ROC curve of {}".format(title), fontdict={'family': 'Times New Roman', 'size': 20})
    #     ax.axis("square")
        plt.xticks(fontname='Times New Roman', fontsize=16,weight='bold')
        plt.yticks(fontname='Times New Roman', fontsize=16,weight='bold')
        ax.legend(loc="lower right")
        ax.legend(prop={'family': 'Times New Roman', 'size': 14,'weight':'bold'}) 

        # 修改 label 的字体和字号
        plt.savefig(path+'/{}'.format(j),dpi=1200,bbox_inches = 'tight')

        plt.show()
    

fill_plotm([interp_tprd_lgb,interp_tprd_xgb,interp_tprd_cat,interp_tprd_svc,interp_tprd_lr,interp_tprd_rf,interp_tprd_ltsm,interp_tprd_cnn],[scored_lgb,scored_xgb,scored_cat,scored_svc,scored_lr,scored_rf,scored_ltsm,scored_cnn],f_sv)


# In[32]:


def compare_m(lv):
    # 转置每个DataFrame
    dfd_rf_t = dfd_rf[[lv+'Wakefulness',lv+'Nrem',lv+'Rem']].transpose()
    dfd_lr_t = dfd_lr[[lv+'Wakefulness',lv+'Nrem',lv+'Rem']].transpose()
    dfd_svc_t = dfd_svc[[lv+'Wakefulness',lv+'Nrem',lv+'Rem']].transpose()
    dfd_cat_t = dfd_cat[[lv+'Wakefulness',lv+'Nrem',lv+'Rem']].transpose()
    dfd_xgb_t = dfd_xgb[[lv+'Wakefulness',lv+'Nrem',lv+'Rem']].transpose()
    dfd_lgb_t = dfd_lgb[[lv+'Wakefulness',lv+'Nrem',lv+'Rem']].transpose()
    dfd_ltsm_t = dfd_ltsm[[lv+'Wakefulness',lv+'Nrem',lv+'Rem']].transpose()
    dfd_cnn_t = dfd_cnn[[lv+'Wakefulness',lv+'Nrem',lv+'Rem']].transpose()
    # 将这些DataFrame合并到一个DataFrame中
    df_combined = pd.concat([dfd_lgb_t, dfd_xgb_t, dfd_cat_t, dfd_svc_t, dfd_lr_t, dfd_rf_t, dfd_ltsm_t, dfd_cnn_t], axis=1)
    # 设置列名
    column_names = ['LGBoost', 'XGBoost','CatBoost','SVM','LR','RF','LTSM','CNN']
    column_names_repeated = [name for name in column_names for _ in range(30)]
    df_combined.columns=column_names_repeated
    # 将DataFrame写入Excel
    df_combined.to_csv(f_sv+"8模型{}.csv".format(lv),encoding='utf_8_sig') 

compare_m('精确率')
compare_m('召回率')
compare_m('敏感性')
compare_m('特异性')


# In[33]:


dfd_rf_t =dfd2_rf.transpose()
dfd_lr_t = dfd2_lr.transpose()
dfd_svc_t = dfd2_svc.transpose()
dfd_cat_t = dfd2_cat.transpose()
dfd_xgb_t = dfd2_xgb.transpose()
dfd_lgb_t = dfd2_lgb.transpose()
dfd_ltsm_t = dfd2_ltsm.transpose()
dfd_cnn_t = dfd2_cnn.transpose()
df_combined = pd.concat([dfd_lgb_t, dfd_xgb_t, dfd_cat_t, dfd_svc_t, dfd_lr_t, dfd_rf_t, dfd_ltsm_t, dfd_cnn_t], axis=1)
# 设置列名
column_names = ['LGBoost', 'XGBoost','CatBoost','SVM','LR','RF','LTSM','CNN']
column_names_repeated = [name for name in column_names for _ in range(30)]
df_combined.columns=column_names_repeated
# 将DataFrame写入Excel
df_combined.to_csv(f_sv+"8模型time.csv",encoding='utf_8_sig') 


# In[34]:


#单独输出图
def fill_plot(interp_tpr,title,name,aucs,path):

    import matplotlib.font_manager as fm
    
    colors = ['aqua', 'green', 'red']   
    target_names = ['Wake', 'NREM', 'REM']
    interp_tpr=np.array(interp_tpr)    
#     figsize=(8, 6)
    fig, ax = plt.subplots(figsize=(8, 6))
    mean_tpr = np.mean(interp_tpr[:,0,:], axis=0)
    mean_tpr[-1] = 1.0
    std_tpr = np.std(interp_tpr[:,0,:], axis=0)
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
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
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
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
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)    
    ax.plot(
        mean_fpr,
        mean_tpr,
        color=colors[2],
        label=f"Mean ROC of '{target_names[2]}'(Mean AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
        lw=2,
        alpha=0.8,
    )  
    

    conf_intveral = scipy.stats.norm.interval(0.95,mean_tpr,std_tpr)
    ax.fill_between(mean_fpr, conf_intveral[0], conf_intveral[1], color=colors[2], alpha=0.1,label=r"95% Confidence interval")   
    ax.set(
        xlim=[-0.05, 1.00],
        ylim=[-0.05, 1.1],
#         xlabel="False Positive Rate",
#         ylabel="True Positive Rate",
#         title="ROC curve of {}".format(title)
    )

    ax.set_xlabel("False Positive Rate", fontdict={'family': 'Times New Roman', 'size': 16})
    ax.set_ylabel("True Positive Rate", fontdict={'family': 'Times New Roman', 'size': 16})
    ax.set_title("ROC curve of {}".format(title), fontdict={'family': 'Times New Roman', 'size': 20})
#     ax.axis("square")
    ax.legend(loc="lower right")
    ax.legend(prop={'family': 'Times New Roman', 'size': 10})  # 修改 label 的字体和字号
    plt.savefig(path+'/{}'.format(name),dpi=600,bbox_inches = 'tight')
    plt.show()
    
    
fill_plot(interp_tprd_lgb,"LGBoost","{}Lightgbm".format(num),scored_lgb,f_sv)
fill_plot(interp_tprd_xgb,"XGBoost","{}XGboost".format(num),scored_xgb,f_sv)
fill_plot(interp_tprd_cat,"CATBoost","{}CATboost".format(num),scored_cat,f_sv)
fill_plot(interp_tprd_svc,"SVM","{}SVC".format(num),scored_svc,f_sv)
fill_plot(interp_tprd_lr,"Logistic Regression","{}Logistic Regression".format(num),scored_lr,f_sv)
fill_plot(interp_tprd_rf,"Random Forest","{}Random Forest".format(num),scored_rf,f_sv)
fill_plot(interp_tprd_ltsm,"LTSM","{}LTSM".format(num),scored_ltsm,f_sv)
fill_plot(interp_tprd_cnn,"CNN","{}CNN".format(num),scored_cnn,f_sv)


# In[ ]:




