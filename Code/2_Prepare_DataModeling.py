#############################
# PREPARE DATA FOR MODELING
###########################

import sys
import os
import pandas as pd
import numpy as np
import re
import random
random.seed(42)

# path = path here
sys.path.insert(0, path) # insert path


from traintestencode import train_test_encode
from filterfeatures import filter_features
from ngram_vec import ngram

#------------------------------------------------------------------------
# Load data
#------------------------------------------------------------------------
 
n = pd.read_csv(os.path.join(path,'df_with_lemma_mrs.csv'))
    
def arrange(n, df):
    
    n.Notes = ' ' + n.Notes.astype(str) + ' '
    
    n = n.groupby(['PatientID', 'SexDSC', 'Age', 'HospitalAdmitDTS',
                   'HospitalDischargeDTS', 'Date_mrs_post_discharge','mrs', 
                   'DischargeDispositionDSC','follow_up']).Notes.sum().reset_index() 
    
    n.Notes = n.Notes.apply(lambda x: " ".join(x.split())) # removes duplicated spaces  
     
    # Count tokens   
    n['Ntokens'] = n.Notes.astype('str').apply(lambda x: len(re.findall(r'\w+',x)))
    
    n['remove'] = 0
    n['remove'][n.Ntokens<300] = 1
    
    n_remove = n
    
    # rejoin
    n = df
            
    n.Notes = ' ' + n.Notes.astype(str) + ' '
    
    n = n.groupby(['PatientID', 'SexDSC', 'Age', 'HospitalAdmitDTS',
                   'HospitalDischargeDTS', 'Date_mrs_post_discharge','mrs', 
                   'DischargeDispositionDSC', 'NoteID','follow_up']).Notes.sum().reset_index()
    
    n.Notes = n.Notes.apply(lambda x: " ".join(x.split())) # removes duplicated spaces  
     
    n = pd.merge(n, n_remove[['PatientID', 'SexDSC', 'Age','HospitalAdmitDTS',
                   'HospitalDischargeDTS', 'Date_mrs_post_discharge','mrs', 
                   'DischargeDispositionDSC','remove','follow_up']], on=['PatientID', 'SexDSC', 'Age', 'HospitalAdmitDTS',
                                  'HospitalDischargeDTS', 'Date_mrs_post_discharge','mrs', 
                                  'DischargeDispositionDSC','follow_up'])
                                                                
    n = n[n.remove != 1].drop(columns='remove')                                                        
    
    n['Expired'] = 0
    n['Expired'][n['mrs'] == 6] = 1
    
    n['SexDSC'][n['SexDSC'] == 'Male'] = 0
    n['SexDSC'][n['SexDSC'] == 'Female'] = 1
      
    return n              

n = n[~(n.Notes.astype(str)=='nan')]

n = arrange(n, n)
  

feature = 'Notes'
outcome = 'mrs' 

#------------------------------------------------------
# Create train and test sets
#------------------------------------------------------

#stratified random sampling 
X_train, X_test = train_test_encode(n, outcome, feature)

#------------------------------------------------------------------------
# Remove misrepresented features in train
#------------------------------------------------------------------------
 
X_train2 = filter_features(X_train, outcome, feature, path) # only train in filter_features()

X_train = pd.concat([X_train.drop(columns='Notes'),X_train2[['Notes']]], axis=1)

#------------------------------------------------------------------------
# Merge notes
#------------------------------------------------------------------------

def merge(d):
    
    d = d[~(d.Notes.astype(str) == '')]
    
    d = d.sort_values(['HospitalAdmitDTS',"NoteID"], ascending=[True,True])

    #added
    d.Notes = ' ' + d.Notes.astype(str) + ' '
   
    d = d.groupby(['PatientID','SexDSC','Age','HospitalAdmitDTS','HospitalDischargeDTS',
                   'mrs','Expired','follow_up']).Notes.sum().reset_index()
 
    d.Notes = d.Notes.apply(lambda x: " ".join(x.split())) # removes duplicated spaces    

    return d

X_train = merge(X_train)
X_test = merge(X_test)

y_train = X_train.mrs
y_test = X_test.mrs

# include Date_mrs_post_discharge

X_train = pd.merge(X_train, n[['PatientID','SexDSC','Age','HospitalAdmitDTS',
                               'HospitalDischargeDTS','mrs','Expired',
                               'follow_up','Date_mrs_post_discharge']].drop_duplicates(), 
                               on=['PatientID','SexDSC','Age','HospitalAdmitDTS',
                               'HospitalDischargeDTS','mrs','Expired','follow_up'])

X_test = pd.merge(X_test, n[['PatientID','SexDSC','Age','HospitalAdmitDTS',
                               'HospitalDischargeDTS','mrs','Expired',
                               'follow_up','Date_mrs_post_discharge']].drop_duplicates(), 
                               on=['PatientID','SexDSC','Age','HospitalAdmitDTS',
                               'HospitalDischargeDTS','mrs','Expired','follow_up'])

# save X_train and X_test

X_train.to_csv(os.path.join(path,'X_train_mrs_demo.csv'), index=False)
X_test.to_csv(os.path.join(path,'X_test_mrs_demo.csv'), index=False)

#------------------------------------------------------------------------
# Vectorization
#------------------------------------------------------------------------

# vectorization for combinations of n-grams

x_train11, x_test11 = ngram(X_train, X_test, feature, ngram_range=(1, 1))
print(1)
x_train12, x_test12 = ngram(X_train, X_test, feature, ngram_range=(1, 2))
print(2)
x_train13, x_test13 = ngram(X_train, X_test, feature, ngram_range=(1, 3))
print(3)
x_train22, x_test22 = ngram(X_train, X_test, feature, ngram_range=(2, 2))
print(4)
x_train23, x_test23 = ngram(X_train, X_test, feature, ngram_range=(2, 3))
print(5)
x_train33, x_test33 = ngram(X_train, X_test, feature, ngram_range=(3, 3))
print(6)

#join all    
x_tr = pd.concat([x_train11, x_train12, x_train13, x_train22, x_train23, x_train33], axis = 1)
x_t = pd.concat([x_test11, x_test12, x_test13, x_test22, x_test23, x_test33], axis = 1)   

#remove repeated columns
columns = x_tr.columns.drop_duplicates()
x_train = x_tr.loc[:,~x_tr.columns.duplicated()]
x_test = x_t.loc[:,~x_t.columns.duplicated()]

x_train = x_train.astype(int)
x_test = x_test.astype(int)

x_train[x_train >0] = 1 
x_test[x_test >0] = 1 


#------------------------------------------------------------------------
# Add other variables besides text
#------------------------------------------------------------------------

def scale_range (input, min, max):
    input += -(np.min(input))
    input /= np.max(input) / (max - min)
    input += min
    return input

cols = ['Age']

for col in cols:
    x_test[col] = np.round(scale_range (X_test[col], np.min(X_train[col]), np.max(X_train[col])))
    x_test[col] = (x_test[col] - np.min(X_train[col])) / (np.max(X_train[col]) - np.min(X_train[col]))
    x_train[col] = (X_train[col] - np.min(X_train[col])) / (np.max(X_train[col]) - np.min(X_train[col]))

x_train['SexDSC'] = X_train['SexDSC']
x_test['SexDSC'] = X_test['SexDSC']

x_train['Expired'] = X_train['Expired']
x_test['Expired'] = X_test['Expired']

x_train['follow_up'] = X_train['follow_up']
x_test['follow_up'] = X_test['follow_up']

# save x_train and x_test

x_train.to_csv(os.path.join(path,'x_train_mrs_features_demo.csv'), index=False)
x_test.to_csv(os.path.join(path,'x_test_mrs_features_demo.csv'), index=False)
