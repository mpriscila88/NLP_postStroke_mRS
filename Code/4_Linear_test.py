###########################
# Linear model
###########################

import sys
import os
import pandas as pd
import numpy as np
import random
random.seed(42)
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV

# path = path here
sys.path.insert(0, path) # insert path

x_train = pd.read_csv(os.path.join(path,'x_train_mrs_features.csv'))
x_test = pd.read_csv(os.path.join(path,'x_test_mrs_features.csv'))

# Each Admission represents [PatientID, HospitalAdmitDTS, 
# HospitalDischargeDTS, Date_mrs_post_discharge]

X_train = pd.read_csv(os.path.join(path,'X_train_mrs.csv'))
X_test = pd.read_csv(os.path.join(path,'X_test_mrs.csv'))

e = pd.read_csv(os.path.join(path,'extractions.csv'))

X_test = pd.merge(X_test.reset_index(drop=True),e,how='left', on=['Admission'])

y_train = X_train.mrs
y_test = X_test.mrs


#------------------------------------------------------------------------
#  Modeling - Lasso
#------------------------------------------------------------------------

lasso = Lasso(random_state=0, max_iter=100)
alphas = [0.0001, 0.001, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.5, 1, 1.5, 2, 5]

tuned_parameters = [{"alpha": alphas}]
n_folds = 5

clf = GridSearchCV(lasso, tuned_parameters, cv=n_folds, refit=False)
clf.fit(x_train, y_train)
scores = clf.cv_results_["mean_test_score"]
scores_std = clf.cv_results_["std_test_score"]

# Alpha parameter
scores_ = pd.DataFrame(scores)
maxpos = scores_.index[scores_[0]==max(scores_[0])] 

best_alpha = alphas[maxpos[0]] # 0.01


clf = Lasso(alpha=best_alpha)

clf.fit(x_train, y_train)

# clf.intercept_

y_pred = clf.predict(x_test)

y_pred[y_pred<0] = 0
y_pred[y_pred>6] = 6

# without stages
mean_squared_error(y_test, y_pred, squared=False)

#with stages

c = (x_test.Expired == 1)
y_pred[c] = 6

mean_squared_error(y_test, y_pred, squared=False) 
                    
# for follow-up visits

c = X_test.mrs_extractions.astype(str)!='nan'

y_pred[c] = X_test['mrs_extractions'][c]

mean_squared_error(y_test, y_pred, squared=False)




sns.set_style("whitegrid", {"grid.color": ".6", "grid.linestyle": ":"})


plt.figure(figsize=(6,6))
plt.rcParams["axes.edgecolor"] = "0.15"
plt.rcParams["axes.linewidth"]  = 0.5

y = pd.DataFrame(y_pred)
y=y.rename(columns={0:'y_pred'})
n = pd.concat([pd.DataFrame(y_test),y], axis=1)
sns.boxplot(data=n, x="mrs", y="y_pred", linewidth=.75, color = 'steelblue', boxprops=dict(alpha=.9))

plt.rcParams["font.family"] = "Cambria"
plt.xlabel('Target mRS', fontsize=25)
plt.ylabel('Predicted mRS', fontsize=25)
plt.rcParams["font.family"] = "Cambria"
plt.xticks(ticks = [0,1,2,3,4,5,6],labels=['0','1','2','3','4','5','6'])
plt.yticks(ticks = [0,1,2,3,4,5,6],labels=['0','1','2','3','4','5','6'])
plt.rcParams.update({'font.size': 25})
plt.show()








                                             
# configure bootstrap

from numpy.random import seed, randint

def get_CI_boot_outcome(y_true,y_pred,boot):
    # bootstrap confidence intervals
    # seed the random number generator
    seed(1)
    i = 0
    # generate dataset
    dataset = y_pred
    real = y_true
    # bootstrap
    scores = list()
    while i < boot:
        # bootstrap sample
        indices = randint(0, len(y_pred) - 1, len(y_pred))
        sample = dataset[indices]
        real = y_true[indices]
        if len(np.unique(y_true[indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            # to be defined: reject the sample
            continue
    	# calculate and store statistic 
        else:
            statistic = mean_squared_error(real,sample, squared=False)
            scores.append(statistic)
            i += 1
    # calculate 95% confidence intervals (100 - alpha)
    alpha = 0.95
    p = ((1.0-alpha)/2.0) * 100
    lower =  np.percentile(scores, p)
    p = (alpha+((1.0-alpha)/2.0)) * 100
    upper =  np.percentile(scores, p)
    return lower,upper
    
lower,upper = get_CI_boot_outcome(y_test,y_pred,boot=1000) 

print(mean_squared_error(y_test, y_pred, squared=False), lower, upper)

# 0.896379591478753 0.8621067401744963 0.9299760911202595

#------------------------------------------------------------------------
# Top features
#------------------------------------------------------------------------

coef = pd.DataFrame(clf.coef_).T

coef.columns = x_train.columns

ind = (coef == 0).all()
ind = pd.DataFrame(ind,columns=['Bool'])
top_features = pd.DataFrame(ind.loc[(ind.Bool==False)].index, columns=['Features'])
 
# Select N from top features 

N = 20

coef = pd.DataFrame(clf.coef_, columns=['Coef'])

a = pd.DataFrame(x_train.columns, columns=['Features'])

coef = pd.concat([a, coef], axis=1)

coef = coef[coef.Features.isin(top_features.Features)]

coef.Coef = np.abs(coef.Coef)

sorted_idx = coef.Coef.sort_values(ascending=False)

sorted_idx = sorted_idx[0:N]

coef=coef[coef.index.isin(sorted_idx.index)]

top_features = coef[['Features']]
  
#------------------------------------------------------------------------
# Number of uni, bi and trig in the set of features
#------------------------------------------------------------------------

from nltk.tokenize import word_tokenize 

top_features['n'] = top_features.Features.astype(str).apply(lambda x: len(word_tokenize(x)))

len(top_features[top_features.n == 1]) 
len(top_features[top_features.n == 2]) 
len(top_features[top_features.n == 3]) 

a = pd.DataFrame(x_train.columns, columns=['Features'])

a['n'] = a.Features.astype(str).apply(lambda x: len(word_tokenize(x)))

len(a[a.n == 1]) 

#------------------------------------------------------------------------
# Feature importance estimates - plot
#------------------------------------------------------------------------

import matplotlib.pyplot as plt

import matplotlib
matplotlib.rc('pdf', fonttype=42)
matplotlib.rc('font', **{'size': 22})
from operator import itemgetter

import seaborn as sns
sns.set_style("whitegrid", {"grid.color": ".6", "grid.linestyle": ":"})

def plt_importance_all2(coef0, features, top_features, size):
    
    indices, L_sorted = zip(*sorted(enumerate(coef0.iloc[0,:]), key=itemgetter(1)))
    
    var = pd.DataFrame(features[pd.DataFrame(indices)[0]],columns=['Features'])
      
    # Select top features only
    var = var[var.Features.isin(top_features.Features)]    
    L_sorted = pd.DataFrame(L_sorted,columns=['coef'])
    L_sorted = L_sorted[L_sorted.index.isin(var.index)]                          
    
   
    data_x = var.Features
    data_hight = sorted(L_sorted.coef)
    
    data_hight_normalized = pd.DataFrame( [x / max(np.abs(data_hight)) for x in data_hight]) * 100
    
    
    
    # my_cmap = plt.cm.get_cmap('coolwarm')
    # colors = my_cmap(data_hight_normalized[0])
    
    fig = plt.figure(figsize=(20, size)) 
    ax = fig.add_subplot(1, 1, 1) 
    ax.barh(data_x, data_hight_normalized[0], color ='tab:blue') 

    plt.rcParams["font.family"] = "Cambria"
    plt.grid(color='grey', linestyle=':', linewidth=1)
    # plt.xlabel('Relative features importance (%)')

    get_indexes_neg = data_hight_normalized.apply(lambda x: x[x<0].index)
    b = ax.barh(data_x, data_hight_normalized[0], color='crimson',alpha=0.8)
    for ind in range(len(get_indexes_neg)):
        b[get_indexes_neg[0][ind]].set_color('#346cb0')
     
    ax.set_xticks([-100,-75,-50,-25,0,25,50,75,100])
    
    ax.set_yticklabels(data_x, fontsize=35)
    ax.set_xticklabels([-100,-75,-50,-25,0,25,50,75,100],fontsize=35)
    ax.set_xlabel('Relative Feature Importance (%)', fontsize=35)

    import matplotlib.patches as mpatches
    red_patch = mpatches.Patch(color='#346cb0', label='Negative coefficient')
    blue_patch = mpatches.Patch(color='crimson', alpha=0.8, label='Positive coefficient')
    plt.rcParams["font.family"] = "Cambria"
    plt.legend(handles=[red_patch, blue_patch],loc='lower right',prop={'size': 35})
    plt.rcParams["font.family"] = "Cambria"

    
coef = pd.DataFrame(clf.coef_).T

coef.columns = x_train.columns

plt_importance_all2(coef, x_train.columns, top_features, size=20)



# Spearman correlation between target vs predicted (and confidence interval) 

from scipy import stats

x = stats.spearmanr(y_test, y_pred)[0] # 

import pingouin as pg

stat = stats.spearmanr(y_test, y_pred)[0]

ci = pg.compute_bootci(y_test, y_pred, func='spearman', n_boot=1000, confidence=0.95,
                       paired=True, seed=42, decimals=4)


print(round(stat, 4), ci) 



# ###################################
# # Analysis for notes availability
# ###################################

# ae = pd.concat([X_test,pd.Series(y_pred)], axis=1)
 
# ae = ae.rename(columns={0:'y_pred'})

# # Discharge
# ndis = pd.read_csv(os.path.join(path,'dis1.csv'))

# cols = ['HospitalAdmitDTS', 'HospitalDischargeDTS']

# for i in cols:
#     ndis[i] = ndis[i].astype('datetime64[ns]')

# # discharge notes

# ae_dis = ae[ae.follow_up==0]

# ae_dis2 = pd.merge(ae_dis[['PatientID', 'HospitalAdmitDTS', 'HospitalDischargeDTS', 'mrs','y_pred']].drop_duplicates(), 
#                ndis[['PatientID', 'SexDSC', 'Age', 'HospitalAdmitDTS', 'HospitalDischargeDTS',
#                'OT_days','PT_days', 'DS_days', 'DD_days']].drop_duplicates(), 
#                 on=['PatientID', 'HospitalAdmitDTS', 'HospitalDischargeDTS'])

# # Follow-up
# df_fup = pd.read_csv(os.path.join(path,'fup1.csv'))

# ae_dis = ae[ae.follow_up==1]

# cols = ['HospitalAdmitDTS', 'HospitalDischargeDTS','Date_mrs_post_discharge']

# for i in cols:
#     df_fup[i] = df_fup[i].astype('datetime64[ns]')
#     ae[i] = ae[i].astype('datetime64[ns]')
    
  
# aefup = pd.merge(ae[['PatientID', 'HospitalAdmitDTS', 'HospitalDischargeDTS', 'Date_mrs_post_discharge','mrs','y_pred']].drop_duplicates(), 
#                df_fup[['PatientID', 'HospitalAdmitDTS', 'HospitalDischargeDTS', 'Date_mrs_post_discharge',
#                      'OT_days', 'PT_days', 'DS_days', 'DD_days']].drop_duplicates(), 
#                 on=['PatientID', 'HospitalAdmitDTS', 'HospitalDischargeDTS','Date_mrs_post_discharge'])

# len(ae_dis2)+len(aefup)

# cols = ['PatientID', 'HospitalAdmitDTS', 'HospitalDischargeDTS', 'mrs', 'y_pred',
#        'OT_days', 'PT_days', 'DS_days', 'DD_days']
    
# ae = pd.concat([ae_dis2[cols], aefup[cols]], axis = 0)    
    
# ae['nmin'] = np.nanmin(ae[['OT_days', 'PT_days', 'DS_days', 'DD_days']], axis=1)

# ae = ae.reset_index(drop=True)

# col = 'nmin'
 
# rms = []
# scs = []
# Ns = []

# rms_l = []
# rms_u = []
# scs_l = []
# scs_u = []

# def results(aen, rms, scs, Ns, rms_l, rms_u, scs_l, scs_u):
    
#     y_pred = aen.y_pred
#     y_test = aen.mrs
    
#     rm = mean_squared_error(y_test, y_pred, squared=False) # rmse

#     y = pd.DataFrame(y_test).reset_index(drop=True)

#     lower,upper = get_CI_boot_outcome(y.mrs,y_pred,boot=1000)

#     sc = stats.spearmanr(y_test, y_pred)[0]

#     ci = pg.compute_bootci(y_test, y_pred, func='spearman', n_boot=1000, confidence=0.95,
#                     paired=True, seed=42, decimals=4)

#     Ns.append(len(aen))
#     rms.append(rm)
#     scs.append(sc)
#     Ns.append(Ns)
#     rms_l.append(lower)
#     rms_u.append(upper)
#     scs_l.append(ci[0])
#     scs_u.append(ci[1])
    
#     return rms, scs, Ns, rms_l, rms_u, scs_l, scs_u


# # 0 weeks
# aen = ae[(ae[col].astype(int) ==  0)].reset_index(drop=True)
# rms, scs, Ns, rms_l, rms_u, scs_l, scs_u = results(aen, rms, scs, Ns, rms_l, rms_u, scs_l, scs_u)

# # ]0, 1] weeks
# aen = ae[(ae[col].astype(int) >  0) & (ae[col].astype(int) <= 7*1)].reset_index(drop=True)
# rms, scs, Ns, rms_l, rms_u, scs_l, scs_u = results(aen, rms, scs, Ns, rms_l, rms_u, scs_l, scs_u)

# # ]1, 5] weeks
# aen = ae[(ae[col].astype(int) >  7*1) & (ae[col].astype(int) <= 7*5)].reset_index(drop=True)
# rms, scs, Ns, rms_l, rms_u, scs_l, scs_u = results(aen, rms, scs, Ns, rms_l, rms_u, scs_l, scs_u)

# # >= 5 weeks
# aen = ae[(ae[col].astype(int) > 7*5)].reset_index(drop=True) 
# rms, scs, Ns, rms_l, rms_u, scs_l, scs_u = results(aen, rms, scs, Ns, rms_l, rms_u, scs_l, scs_u)
  
# # > 1 weeks
# aen = ae[(ae[col].astype(int) > 7*1)].reset_index(drop=True) 
# rms, scs, Ns, rms_l, rms_u, scs_l, scs_u = results(aen, rms, scs, Ns, rms_l, rms_u, scs_l, scs_u)
  

# rms = pd.DataFrame(rms)
# scs = pd.DataFrame(scs)

# rms_l = pd.DataFrame(rms_l)
# rms_u = pd.DataFrame(rms_u)

# scs_l = pd.DataFrame(scs_l)
# scs_u = pd.DataFrame(scs_u)  

# a = pd.concat([rms,rms_l,rms_u,scs,scs_l,scs_u], axis = 1)