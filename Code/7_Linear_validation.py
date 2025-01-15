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

path = path here
sys.path.insert(0, path) # insert path

x_train = pd.read_csv(os.path.join(path,'x_train_mrs_features.csv'))
x_test = pd.read_csv(os.path.join(path,'x_val_mrs_features.csv'))

# Each Admission represents [PatientID, HospitalAdmitDTS, 
# HospitalDischargeDTS, Date_mrs_post_discharge]

X_train = pd.read_csv(os.path.join(path,'X_train_mrs.csv'))
X_test = pd.read_csv(os.path.join(path,'X_val_mrs.csv'))

# in case there are any extracted scores, 
# here loads the extractions dataset for validation

# e = pd.read_csv(os.path.join(path,'extractions_val.csv')) 

# X_test = pd.merge(X_test.reset_index(drop=True),e,how='left', on=['Admission'])


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


#------------------------------------------------------------------------
#  Modeling - Lasso
#------------------------------------------------------------------------

clf = Lasso(alpha=0.01)

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
                    
## for follow-up visits -- in case there are extractions for validation set

# c = X_test.mrs_extractions.astype(str)!='nan'

# y_pred[c] = X_test['mrs_extractions'][c]

# mean_squared_error(y_test, y_pred, squared=False)




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

# ae = pd.concat([X_test.reset_index(drop=True),pd.DataFrame(y_pred, columns=['y_pred'])], axis=1)


# # Discharge
# ndis = pd.read_csv(os.path.join(path,'dis1.csv'))


# # discharge notes

# ae_dis = ae[ae.follow_up==0]

# ae_dis2 = pd.merge(ae_dis, 
#                ndis[['PatientID','admission', 'discharge',
#                'DS_days', 'DD_days']].drop_duplicates().reset_index(drop=True), 
#                 on=['PatientID', 'admission', 'discharge'])


# # Follow-up
# df_fup = pd.read_csv(os.path.join(path,'fup1.csv'))

# ae_dis = ae[ae.follow_up==1]

 
# aefup = pd.merge(ae_dis.reset_index(drop=True),
#                df_fup[['PatientID', 'admission', 'discharge', 'DS_days', 'DD_days']].drop_duplicates().reset_index(drop=True), 
#                 on=['PatientID', 'admission', 'discharge'])

# len(ae_dis2)+len(aefup) # 394-387 = 7 deceased patients without notes

# cols = ['PatientID', 'admission', 'discharge', 'mrs', 
#         'y_pred', 'DS_days', 'DD_days']
    
# ae = pd.concat([ae_dis2[cols], aefup[cols]], axis = 0).reset_index(drop=True)    

# ae['nmin'] = np.nanmin(ae[['DS_days', 'DD_days']], axis=1)

# ae = ae[~(ae.nmin.astype(str)=='nan')]  

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


# # > 0 weeks
# aen = ae[(ae[col].astype(int) > 0)].reset_index(drop=True) 
# rms, scs, Ns, rms_l, rms_u, scs_l, scs_u = results(aen, rms, scs, Ns, rms_l, rms_u, scs_l, scs_u)
  



# rms = pd.DataFrame(rms)
# scs = pd.DataFrame(scs)

# rms_l = pd.DataFrame(rms_l)
# rms_u = pd.DataFrame(rms_u)

# scs_l = pd.DataFrame(scs_l)
# scs_u = pd.DataFrame(scs_u)  

# a = pd.concat([rms,rms_l,rms_u,scs,scs_l,scs_u], axis = 1)
