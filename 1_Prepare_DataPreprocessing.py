#################################
# PREPARE DATA FOR PREPROCESSING
################################

import sys
import os
import pandas as pd

# path = path here
sys.path.insert(0, path) # insert path


##################################
# NOTES + demograhics - DISCHARGE
##################################

# dataset with demographics + hospital admission and discharge + mrs outcomes at discharge

df = pd.read_csv(os.path.join(path,'df_mrs.csv')) 

# dataset with notes for patients
notes = pd.read_csv(os.path.join(path,'notes.csv'), encoding='cp1252') 

notes = pd.merge(df, notes, on='PatientID')

# for mrs at discharge - initial pre-selection of notes within the hospital stay
notes = notes[(notes.ContactDTS.astype('datetime64[ns]') >= notes.HospitalAdmitDTS.astype('datetime64[ns]')) &
              (notes.ContactDTS.astype('datetime64[ns]') <= notes.HospitalDischargeDTS.astype('datetime64[ns]'))]

# create dummy column Date_mrs_post_discharge
notes['Date_mrs_post_discharge'] = notes['HospitalDischargeDTS']

# remove column from dataset if it exists
notes = notes.drop(columns='mrs_90days')

notes = notes[notes.Age>=18]

notes.to_csv(os.path.join(path,'notes_discharge_before_preprocessing.csv'), index=False)


#######################################
# NOTES + demograhics - POST-DISCHARGE
#######################################

# dataset with demographics + hospital admission and discharge and post-discharge + mrs outcomes at follow-up

df = pd.read_csv(os.path.join(path,'df_mrs.csv')) 

# dataset with notes for patients
notes = pd.read_csv(os.path.join(path,'notes.csv'), encoding='cp1252') 

notes = pd.merge(df, notes, on='PatientID')

# for mrs at follow-up - initial pre-selection of notes between discharge and follow-up
notes = notes[(notes.ContactDTS.astype('datetime64[ns]') >= notes.HospitalAdmitDTS.astype('datetime64[ns]'))]
notes = notes[(notes.ContactDTS.astype('datetime64[ns]') <= notes.Date_mrs_post_discharge.astype('datetime64[ns]'))]


notes['mrs'] = notes['mrs_90days']

notes = notes.drop(columns='mrs_90days')

notes.to_csv(os.path.join(path,'notes_mrs90days_before_preprocessing.csv'), index=False)

###############################
# NOTES PREPROCESSING
##############################

from Preprocessing_function import create_types, preprocessing

df_dis0 = pd.read_csv(os.path.join(path,'notes_discharge_before_preprocessing.csv')) 

df_dis1 = create_types(df_dis0, 'discharge')

# df_dis1.to_csv(os.path.join(path, 'dis1.csv'), index=False)

df_dis = preprocessing(df_dis1)

df_dis['follow_up'] = 0

df_dis.to_csv(os.path.join(path, 'dis.csv'), index=False)


df_fup0 = pd.read_csv(os.path.join(path,'notes_mrs90days_before_preprocessing.csv')) 

df_fup1 = create_types(df_fup0, 'follow_up')

# df_fup1.to_csv(os.path.join(path, 'fup1.csv'), index=False)

df_fup = preprocessing(df_fup1)

df_fup['follow_up'] = 1

df_fup.to_csv(os.path.join(path, 'fup.csv'), index=False)

# Join both

n = pd.concat([df_dis,df_fup[list(df_dis.columns)]], axis=0).reset_index(drop=True)

n.to_csv(os.path.join(path,'df_with_lemma_mrs.csv'), index=False)
