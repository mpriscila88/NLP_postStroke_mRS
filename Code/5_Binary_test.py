###########################
# Binary model
###########################

import sys
import os
import pandas as pd
import numpy as np
import random
random.seed(42)
import matplotlib.pyplot as plt

# path = path here
sys.path.insert(0, path) # insert path

x_train = pd.read_csv(os.path.join(path,'x_train_mrs_features.csv'))
x_test = pd.read_csv(os.path.join(path,'x_test_mrs_features.csv'))

# Each Admission represents [PatientID, HospitalAdmitDTS, 
# HospitalDischargeDTS, Date_mrs_post_discharge]

X_train = pd.read_csv(os.path.join(path,'X_train_mrs.csv'))
X_test = pd.read_csv(os.path.join(path,'X_test_mrs.csv'))

e = pd.read_csv(os.path.join(path,'extractions.csv'))

#------------------------------------------------------------------------
# Binary
#------------------------------------------------------------------------

label = '36' # MRS 0-2, 3-6
    
class_labels = ['No symptoms to slight disability','Moderate disability to death']
cm_labels = ['mRS 0-2','mRS 3-6']

def turn(y):
    y[y.astype(str).str.contains('1|2')] = 0
    y[y.astype(str).str.contains('3|4|5|6')] = 1
    return y.astype(int)


e.mrs_extractions = turn(e.mrs_extractions)

X_test = pd.merge(X_test.reset_index(drop=True),e,how='left', on=['Admission'])

y_train = turn(X_train.mrs)
y_test = turn(X_test.mrs)
X_test['mrs'] = y_test 


### Binary

from sklearn.metrics import precision_recall_curve

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import FunctionTransformer
from sklearn.linear_model import LogisticRegression

#Pipelines ###################################################################

get_numeric_data = FunctionTransformer(lambda x: x[x_train.columns], validate=False)

clf = LogisticRegression(random_state = 42, max_iter=1000, penalty='l1', solver='liblinear', class_weight='balanced')

num_pipe = Pipeline([
  ('select_num', get_numeric_data),
  ])

full_pipeline = Pipeline([
    ('feat_union', FeatureUnion(transformer_list=[
            ('num_pipeline', num_pipe),
          ])),
    ('clf', clf)
    ])


# LR with text
lst_params =  {
                                                                  'clf__C':[0.0001, 0.001, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.5, 1, 1.5, 2, 5],
                                                                  'clf__warm_start':[True,False]}

random_search = RandomizedSearchCV(full_pipeline, param_distributions=lst_params, n_iter=100, cv=5, refit = True, n_jobs=-1, verbose=1, random_state = 42)


import sys
sys.setrecursionlimit(10000)

random_search.fit(x_train, y_train) 

clf = random_search.best_estimator_

# random_search.best_params_
    
#------------------------------------------------------------------------
# Performance
#------------------------------------------------------------------------

y_train_pred = clf.predict_proba(x_train*1)[:,1]
y_test_pred = clf.predict_proba(x_test*1)[:,1]
    
#%% Threshold functions #######################################################
 
# Optimal Threshold for Precision-Recall Curve (Imbalanced Classification)
    
def optimal_threshold_auc(target, predicted):
    precision, recall, threshold = precision_recall_curve(target, predicted)
    fscore = (2 * precision * recall) / (precision + recall)
    ix = np.argmax(fscore)
    return threshold[ix]
  
# Threshold in train
threshold = optimal_threshold_auc(y_train, y_train_pred)

y_pred = (clf.predict_proba(x_test*1)[:,1] >= threshold).astype(int)

from performance_binary2 import perf

probs = clf.predict_proba(x_test)

# performance without stages
boot_all_micro, boot_all_macro, boot_label = perf(y_test, y_pred, probs, labels=cm_labels)


#with stages

c = (x_test.Expired == 1)

y_pred[c] = 1 

c = X_test.mrs_extractions.astype(str)!='nan'

y_pred[c] = X_test['mrs_extractions'][c]


probs = clf.predict_proba(x_test)
   
probs = pd.DataFrame(probs, columns=['zero','one'])
                 
cond = (y_pred == 0) & (X_test.mrs_extractions.astype(str)!='nan')

probs[cond] = probs[cond]*0
probs.zero[cond] = 1
  
cond = (y_pred == 1) & (X_test.mrs_extractions.astype(str)!='nan')

probs[cond] = probs[cond]*0
probs.one[cond] = 1

from performance_binary2 import perf

boot_all_micro, boot_all_macro, boot_label = perf(y_test, y_pred, np.array(probs), labels=cm_labels)




coef = pd.DataFrame(clf.steps[1][1].coef_, columns = x_train.columns)
   
ind = (coef == 0).all()
ind = pd.DataFrame(ind,columns=['Bool'])
top_features = pd.DataFrame(ind.loc[(ind.Bool==False)].index, columns=['Features'])



from nltk.tokenize import word_tokenize 

top_features['n'] = top_features.Features.astype(str).apply(lambda x: len(word_tokenize(x)))

len(top_features[top_features.n == 1]) 
len(top_features[top_features.n == 2]) 
len(top_features[top_features.n == 3]) 


#------------------------------------------------------------------------
# Feature importance estimates - plot
#------------------------------------------------------------------------

import matplotlib
matplotlib.rc('pdf', fonttype=42)
matplotlib.rc('font', **{'size': 22})
from operator import itemgetter

import seaborn as sns
# sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
sns.set_style("whitegrid",{"grid.color": ".6", "grid.linestyle": ":"})

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

N=20    

coef0 = clf.steps[1][1].coef_
coef = abs(coef0)

c = pd.DataFrame(clf.steps[1][1].coef_, columns = x_train.columns)
ind = (c == 0).all()
ind = pd.DataFrame(ind,columns=['Bool'])
a = ind.loc[(ind.Bool==False)].index   

coef0 = coef0[:, (coef0 != 0).any(axis=0)]
coef = abs(coef0)

plt.rcParams["font.family"] = "Cambria"

class_labels = [0]
for i, class_label in enumerate(class_labels):
    feature_importance = coef[class_label]
    feature_signal = coef0[class_label]
    sorted_idx = np.argsort(np.transpose(feature_importance))
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    topN = sorted_idx[-N:]
    cols_names = []
    
    x1 = pd.DataFrame(feature_signal[topN])

for j in topN:
    cols_names.append(a[j])
     
coef = np.transpose(x1)

coef.columns=np.array(cols_names)
        

plt_importance_all2(coef,coef.columns ,pd.DataFrame(coef.columns, columns=['Features']), size=20)



# ###################################
# # Analysis for notes availability
# ###################################

# ae = pd.concat([X_test,pd.Series(y_pred)], axis=1)
 
# ae = ae.rename(columns={0:'y_pred'})

# ae = pd.concat([ae,probs], axis=1)

# # Discharge

# ndis = pd.read_csv(os.path.join(path, 'dis1.csv'))

# cols = ['HospitalAdmitDTS', 'HospitalDischargeDTS']

# for i in cols:
#     ndis[i] = ndis[i].astype('datetime64[ns]')

# # discharge notes

# ae_dis = ae[ae.follow_up==0]

# ae_dis2 = pd.merge(ae_dis[['PatientID', 'HospitalAdmitDTS', 'HospitalDischargeDTS', 'mrs','y_pred','zero','one']].drop_duplicates(), 
#                ndis[['PatientID', 'SexDSC', 'Age', 'HospitalAdmitDTS', 'HospitalDischargeDTS',
#                'OT_days','PT_days', 'DS_days', 'DD_days']].drop_duplicates(), 
#                 on=['PatientID', 'HospitalAdmitDTS', 'HospitalDischargeDTS'])


# len(ae_dis2)

# # Follow-up
# df_fup = pd.read_csv(os.path.join(path, 'fup1.csv'))

# ae_dis = ae[ae.follow_up==1]

# cols = ['HospitalAdmitDTS', 'HospitalDischargeDTS','Date_mrs_post_discharge']

# for i in cols:
#     df_fup[i] = df_fup[i].astype('datetime64[ns]')
#     ae_dis[i] = ae_dis[i].astype('datetime64[ns]')
    
  
# aefup = pd.merge(ae_dis[['PatientID', 'HospitalAdmitDTS', 'HospitalDischargeDTS', 'Date_mrs_post_discharge','mrs','y_pred','zero','one']].drop_duplicates(), 
#                df_fup[['PatientID', 'HospitalAdmitDTS', 'HospitalDischargeDTS', 'Date_mrs_post_discharge',
#                      'OT_days', 'PT_days', 'DS_days', 'DD_days']].drop_duplicates(), 
#                 on=['PatientID', 'HospitalAdmitDTS', 'HospitalDischargeDTS','Date_mrs_post_discharge'])

# len(ae_dis2)+len(aefup)

# cols = ['PatientID', 'HospitalAdmitDTS', 'HospitalDischargeDTS', 'mrs', 'y_pred','zero','one',
#        'OT_days', 'PT_days', 'DS_days', 'DD_days']
    
# ae = pd.concat([ae_dis2[cols], aefup[cols]], axis = 0)    
    
    
# ae['nmin'] = np.nanmin(ae[['OT_days', 'PT_days', 'DS_days', 'DD_days']], axis=1)

# ae = ae.reset_index(drop=True)

# col = 'nmin'
 
# macros = []
# Ns = []

# def results(aen, cols, cm_labels, macros, Ns):
    
#     probs = np.array(aen[cols])
#     y_pred = np.array(aen['y_pred']).astype(int)
#     y_test = aen.mrs
 
#     boot_all_micro,boot_all_macro, boot_label = perf(y_test, y_pred, probs, cm_labels) # for binary change for macro in functions_multiclassification 
 
#     macros.append(boot_all_macro)
#     Ns.append(len(aen))
    
#     return macros, Ns

# cols = ['zero','one']

# # 0 weeks
# aen = ae[(ae[col].astype(int) ==  0)].reset_index(drop=True)
# macros, Ns = results(aen, cols, cm_labels, macros, Ns)

# # ]0, 1] weeks
# aen = ae[(ae[col].astype(int) >  0) & (ae[col].astype(int) <= 7*1)].reset_index(drop=True)
# macros, Ns = results(aen, cols, cm_labels, macros, Ns)

# # ]1, 5] weeks
# aen = ae[(ae[col].astype(int) >  7*1) & (ae[col].astype(int) <= 7*5)].reset_index(drop=True)
# macros, Ns = results(aen, cols, cm_labels, macros, Ns)

# # >= 5 weeks
# aen = ae[(ae[col].astype(int) > 7*5)].reset_index(drop=True) 
# macros, Ns = results(aen, cols, cm_labels, macros, Ns)
  
# # > 1 week
# aen = ae[(ae[col].astype(int) > 7*1)].reset_index(drop=True) 
# macros, Ns = results(aen, cols, cm_labels, macros, Ns)
  