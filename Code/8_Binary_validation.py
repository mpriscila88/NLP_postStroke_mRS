###########################
# Binary model
###########################

import sys
import os
import pandas as pd
import numpy as np
import random
random.seed(42)

# path = path here
sys.path.insert(0, path) # insert path

x_train = pd.read_csv(os.path.join(path,'x_train_mrs_features.csv'))
x_test = pd.read_csv(os.path.join(path,'x_val_mrs_features.csv'))

# Each Admission represents [PatientID, HospitalAdmitDTS, 
# HospitalDischargeDTS, Date_mrs_post_discharge]

X_train = pd.read_csv(os.path.join(path,'X_train_mrs.csv'))
X_test = pd.read_csv(os.path.join(path,'X_val_mrs.csv'))

# y validation

X_test.mrs[X_test.mrs.astype(str).str.contains(str('0 -'))] = 0
X_test.mrs[X_test.mrs.astype(str).str.contains(str('1 -'))] = 1
X_test.mrs[X_test.mrs.astype(str).str.contains(str('2 -'))] = 2
X_test.mrs[X_test.mrs.astype(str).str.contains(str('3 -'))] = 3
X_test.mrs[X_test.mrs.astype(str).str.contains(str('4 -'))] = 4
X_test.mrs[X_test.mrs.astype(str).str.contains(str('5 -'))] = 5
X_test.mrs[X_test.mrs.astype(str).str.contains(str('6 -'))] = 6
X_test.mrs[X_test.mrs.astype(str).str.contains('Deceased')] = 6

X_test['mrs'] = X_test['mrs'].astype(float)

y_train = X_train.mrs
y_test = X_test.mrs

# in case there are any extracted scores, 
# here loads the extractions dataset for validation

# e = pd.read_csv(os.path.join(path,'extractions_val.csv'))

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

# in case there are any extracted scores

# e.mrs_extractions = turn(e.mrs_extractions)
# X_test = pd.merge(X_test.reset_index(drop=True),e,how='left', on=['Admission'])

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
# boot_all_micro, boot_all_macro, boot_label = perf(y_test, y_pred, probs, labels=cm_labels)


#with stages

c = (x_test.Expired == 1)

y_pred[c] = 1 

probs = pd.DataFrame(probs, columns=['zero','one'])
probs['zero'][c] = 0
probs['one'][c] = 1

boot_all_micro, boot_all_macro, boot_label = perf(y_test, y_pred, np.array(probs), labels=cm_labels) 



# ###################################
# # Analysis for notes availability
# ###################################

# ae = pd.concat([X_test,pd.DataFrame(y_pred, columns=['y_pred'])], axis=1)

# ae = pd.concat([ae,probs], axis=1)

# # Discharge

# ndis = pd.read_csv(os.path.join(path, 'dis1.csv'))

# # discharge notes

# ae_dis = ae[ae.follow_up==0]

# ae_dis2 = pd.merge(ae_dis, 
#                ndis[['PatientID', 'gender', 'Age', 'admission', 'discharge',
#                'DS_days', 'DD_days']].drop_duplicates(), 
#                 on=['PatientID', 'admission', 'discharge'])


# len(ae_dis2)

# # Follow-up
# df_fup = pd.read_csv(os.path.join(path, 'fup1.csv'))

# ae_dis = ae[ae.follow_up==1]
    
  
# aefup = pd.merge(ae_dis, 
#                df_fup[['PatientID', 'admission', 'discharge',
#                      'DS_days', 'DD_days']].drop_duplicates(), 
#                 on=['PatientID', 'admission', 'discharge'])

# len(ae_dis2)+len(aefup)

# cols = ['PatientID', 'admission', 'discharge', 'mrs', 'y_pred','zero','one',
#        'DS_days', 'DD_days']
    
# ae = pd.concat([ae_dis2[cols], aefup[cols]], axis = 0)    
    
    
# ae['nmin'] = np.nanmin(ae[['DS_days', 'DD_days']], axis=1)

# ae = ae[~(ae.nmin.astype(str)=='nan')]  

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

# # > 0 weeks
# aen = ae[(ae[col].astype(int) >  0)].reset_index(drop=True)
# macros, Ns = results(aen, cols, cm_labels, macros, Ns)
