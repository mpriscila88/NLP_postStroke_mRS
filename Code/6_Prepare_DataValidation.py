###############################
# NOTES PREPROCESSING
##############################

# Only code is provided in this script for reproducibility

# Only validation datasets output from this script are provided

import sys
import os
import pandas as pd
import numpy as np


##############################################
# DISCHARGE AND FOLLOW_UP NOTES -- validation
##############################################


# path = path here
sys.path.insert(0, path) # insert path


from Preprocessing_function import create_types, preprocessing

df_dis0 = pd.read_csv(os.path.join(path,'notes_discharge_before_preprocessing_bidmc.csv')) 

df_dis0['mrs'] = df_dis0['mrs_discharge']

df_dis1 = create_types(df_dis0, 'discharge')

df_dis1.to_csv(os.path.join(path, 'dis1.csv'), index=False)

df_dis = preprocessing(df_dis1)

df_dis['follow_up'] = 0

# follow up

df_fup0 = pd.read_csv(os.path.join(path,'notes_mrs90days_before_preprocessing_bidmc.csv')) 

df_fup0['mrs'] = df_fup0['mrs_discharge']

df_fup1 = create_types(df_fup0, 'follow_up')

df_fup1.to_csv(os.path.join(path, 'fup1.csv'), index=False)

df_fup = preprocessing(df_fup1)

df_fup['follow_up'] = 1
      
      
# Join both

n = pd.concat([df_dis,df_fup[list(df_dis.columns)]], axis=0).reset_index(drop=True)

n.PatientID =n.PatientID.astype(float)

n.to_csv(os.path.join(path,'df_with_lemma_mrs_bidmc.csv'), index=False)


#############################
# PREPARE DATA FOR MODELING
###########################

#------------------------------------------------------------------------
# Load data
#------------------------------------------------------------------------

import sys
import os
import pandas as pd
import random
random.seed(42)


# path = path here
sys.path.insert(0, path) # insert path


#------------------------------------------------------------------------
# Load data
#------------------------------------------------------------------------
 
n = pd.read_csv(os.path.join(path,'df_with_lemma_mrs_bidmc.csv'))


def arrange(n, df):
             
    n.Notes = ' ' + n.Notes.astype(str) + ' '
    
    n = n.groupby(list(n.drop(columns=['NoteID','Notes']).columns)).Notes.sum().reset_index() 
    
    n.Notes = n.Notes.apply(lambda x: " ".join(x.split())) # removes duplicated spaces  
        
    
    # Expired at hospital discharge
    n['Expired'] = 0
    # n['Expired'][n['mrs'] == 6] = 1
    n['Expired'][n['mrs'].astype(str).str.contains('Deceased')] = 1
    n['Expired'][n.follow_up == 1] = 0
    
    n['gender'][n['gender'] == 'M'] = 0
    n['gender'][n['gender'] == 'F'] = 1
      
    return n              

X_test = arrange(n, n)

  
# save X_test
X_test.to_csv(os.path.join(path,'X_val_mrs.csv'), index=False) # de-identified dataset provided

#------------------------------------------------------------------------
# Vectorization
#------------------------------------------------------------------------

def ngram(X_train, X_test, feature, ngram_range):
    #------------------------------------------------------------------------
    # Vectorization
    #------------------------------------------------------------------------
        
    import pandas as pd
    from sklearn.feature_extraction.text import CountVectorizer
    
    # word level tf-idf
    
    tfidf_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=None, ngram_range=ngram_range)
    
    #tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=None, ngram_range=ngram_range)
    tfidf_vect.fit(X_train[feature])
    xtrain_tfidf =  tfidf_vect.transform(X_train[feature])
    xvalid_tfidf =  tfidf_vect.transform(X_test[feature])  
  
    xtrain_tfidf2 = xtrain_tfidf.astype('uint8')
    
    # Train data

    x_train=pd.DataFrame.sparse.from_spmatrix(xtrain_tfidf2, columns = tfidf_vect.get_feature_names_out())
    x_test=pd.DataFrame.sparse.from_spmatrix(xvalid_tfidf, columns = tfidf_vect.get_feature_names_out())
    
    #sparsity
    a = (x_train == 0).astype(int).sum(axis=0)/len(x_train)*100
  
    #a.hist()
    a = a[a<90]
    a = pd.DataFrame(a.index,columns = ['Features'])
  
    x_train = x_train[a.Features]
    x_test = x_test[a.Features]

    return x_train, x_test

# get X_train

X_train = pd.read_csv(os.path.join(path,'X_train_mrs.csv'))

feature = 'Notes'
outcome = 'mrs' 

# vectorization for combinations of n-grams -- X_test here is validation set

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
x_t = pd.concat([x_test11, x_test12, x_test13, x_test22, x_test23, x_test33], axis = 1)   

#remove repeated columns
x_test = x_t.loc[:,~x_t.columns.duplicated()]

x_test = x_test.astype(int)

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
   
x_test['SexDSC'] = X_test['gender']

x_test['Expired'] = X_test['Expired']

x_test['follow_up'] = X_test['follow_up']

# save x_test

x_test.to_csv(os.path.join(path,'x_val_mrs_features.csv'), index=False) # de-identified dataset provided

