#####################################
# mRS EXTRACTION FOR FOLLOW-UP VISITS
#####################################

import sys
import os
import pandas as pd
import numpy as np
import re

# path = path here
sys.path.insert(0, path) # insert path


df = pd.read_csv(os.path.join(path,'notes_mrs90days_before_preprocessing.csv'))

n = pd.read_csv(os.path.join(path,'df_with_lemma_mrs.csv'))

n = n[n.follow_up==1]

df = df[df.NoteID.isin(n.NoteID)]

df['mrs_90days'] = df['mrs']
df=df.drop(columns='mrs')

# Death

expressions = ['notice of death', 'death note', 'patient deceased',  
               'patient is deceased', 'patient is dead', 
               'deceased patient', 'physician deceased', 'report of death',
               'patient passed away', 'patient died', 'patient has died',
               'patient expired', 'patient is expired', 'patient as deceased',
               'time of death', 'death was pronounced', 'disposition death', 
               'expired discharge', 'discharge expired', 'death discharge', 
               'discharge death', 'deceased discharge', 'discharge deceased']

df['mrs6'] = 0

for i in expressions:
    df['mrs6'][df.NoteTXT.astype(str).str.lower().str.contains(i)] = 1


expressions2 = ['pronounced (her)?(him)?(his)?(as)?\s?death', 
                'pronounced (as)?\s?dea(th)?d?', 
                'pronounced (her)?(him)?(his)?(as)?\s?dead', 
                'pronounced (her)?(him)?(his)?(as)?\s?deceased',
                'patient (is)?\s?expired', 'he (is)?\s?expired', 
                'patient ([\w]).{1,50} expired at', 
                'patient ([\w]).{1,50} expired on', 
                'patient (is)?\s?deceased', 'patient pass(ed)? away']

for i in expressions2:
    df.NoteTXT = df.NoteTXT.astype(str).str.lower().apply(lambda x: re.sub(i,'mrs 6',x))
 
df['mrs6'][df.NoteTXT.astype(str).str.lower().str.contains('mrs 6')] = 1
    

# Start preprocessing

# Preprocessing

notes = df

notes['Notes'] = notes.NoteTXT


def notes_select(o):   

    # Group notes per order of time and note id
    # ------------------------------------------------------

    o = o.sort_values(['ContactDTS',"NoteID","LineNBR"], ascending=[True,True,True])

    o['Notes'] = o['NoteTXT']

    #added
    o.Notes = ' ' + o.Notes.astype(str) + ' '

    # join notes by note date
    
    o = o.groupby(['PatientID', 'HospitalAdmitDTS', 'HospitalDischargeDTS', 
                   'Date_mrs_post_discharge', 'ContactDTS', "NoteID", 'mrs6']).Notes.sum().reset_index()

    # Repeat removing duplicated spaces and words in a row

    # Remove duplicated spaces
    o.Notes = o.Notes.apply(lambda x: " ".join(x.split())) 

    # Remove duplicated words in row
    o.Notes = o.Notes.astype(str).apply(lambda x: re.sub(r'\b(\w+)( \1\b)+', r'\1', x))
    
    o = o.drop(columns='ContactDTS')
    
    o=o.drop_duplicates()
    
    return o

notes = notes_select(notes)


a = notes
a['mrs'] = a.Notes
a['mrs'] = a['mrs'].astype(str).str.lower().apply(lambda x: re.sub('x\]?\s?0\s?-?:?\s?(the patient has)?\s?no\s?(residual)? symptoms','mrs_0',x))
a['mrs'] = a['mrs'].astype(str).str.lower().apply(lambda x: re.sub('x\]?\s?1\s?-?:?\s?(the patient has)?\s?no significant disability','mrs_1',x))
a['mrs'] = a['mrs'].astype(str).str.lower().apply(lambda x: re.sub('x\]?\s?2\s?-?:?\s?(the patient has)?\s?slight disability','mrs_2',x))
a['mrs'] = a['mrs'].astype(str).str.lower().apply(lambda x: re.sub('x\]?\s?3\s?-?:?\s?(the patient has)?\s?moderate disability','mrs_3',x))
a['mrs'] = a['mrs'].astype(str).str.lower().apply(lambda x: re.sub('x\]?\s?4\s?-?:?\s?(the patient has)?\s?moderately severe disability','mrs_4',x))
a['mrs'] = a['mrs'].astype(str).str.lower().apply(lambda x: re.sub('x\]?\s?5\s?-?:?\s?(the patient has)?\s?severe disability','mrs_5',x))
a['mrs'] = a['mrs'].astype(str).str.lower().apply(lambda x: re.sub('x\]?\s?6\s?-?:?\s?(the patient has)?\s?expired','mrs_6',x))
a['mrs'] = a['mrs'].astype(str).str.lower().apply(lambda x: re.sub('x\]?\s?6\s?-?:?\s?(the patient has)?\s?dead','mrs_6',x))

a['mrs'][a['mrs'].astype(str).str.contains('mrs_0')] = 0
a['mrs'][a['mrs'].astype(str).str.contains('mrs_1')] = 1
a['mrs'][a['mrs'].astype(str).str.contains('mrs_2')] = 2
a['mrs'][a['mrs'].astype(str).str.contains('mrs_3')] = 3
a['mrs'][a['mrs'].astype(str).str.contains('mrs_4')] = 4
a['mrs'][a['mrs'].astype(str).str.contains('mrs_5')] = 5
a['mrs'][(a['mrs'].astype(str).str.contains('mrs_6')) | (a.mrs6 == 1)] = 6

a = a[((a.mrs.astype(str) == '0') | (a.mrs.astype(str) == '1') | 
        (a.mrs.astype(str) == '2') | (a.mrs.astype(str) == '3') |
        (a.mrs.astype(str) == '4') | (a.mrs.astype(str) == '5') |
        (a.mrs.astype(str) == '6') )]

a['DD_days'] = a.Date_mrs_post_discharge.astype('datetime64[ns]') - a.HospitalDischargeDTS.astype('datetime64[ns]')

a['DD_days'] = a['DD_days'].astype(str).str.replace(' days','').astype(int)
  
a = a[a.DD_days<=30].drop(columns='DD_days').drop_duplicates() 

a.to_csv(os.path.join(path,'extractions1.csv'), index=False)


dfb = pd.read_csv(os.path.join(path,'notes_mrs90days_before_preprocessing.csv'))

dfb['mrs_90days'] = dfb['mrs']
dfb=dfb.drop(columns='mrs')

aux = dfb[dfb.NoteID.isin(a.NoteID)]
aux['remove'] = 1

dfb = pd.merge(dfb, aux[['PatientID', 'HospitalAdmitDTS', 'HospitalDischargeDTS', 'Date_mrs_post_discharge',
       'mrs_90days','remove']].drop_duplicates(), on=['PatientID', 'HospitalAdmitDTS', 'HospitalDischargeDTS', 'Date_mrs_post_discharge',
       'mrs_90days'], how='outer')
dfb = dfb[~(dfb.remove == 1)]

dfb = dfb[dfb.mrs_90days.astype(int) != 6]

notes = dfb

notes['Notes'] = notes.NoteTXT

# ------------------------------------------------------
# Remove special characters
# ------------------------------------------------------

notes.Notes = notes.Notes.astype(str).apply(lambda x: re.sub('[^a-zA-Z0-9 \n\.]', ' ', x)) 

notes.Notes = notes.Notes.astype(str).apply(lambda x: x.replace('.',' ')) 

notes.Notes = notes.Notes.astype(str).apply(lambda x: x.replace(',',' ')) 

notes.Notes = notes.Notes.astype(str).apply(lambda x: x.replace('score',' ')) 

notes.Notes = notes.Notes.astype(str).apply(lambda x: x.replace('scale',' '))

notes.Notes = notes.Notes.astype(str).apply(lambda x: x.replace('Score',' ')) 

notes.Notes = notes.Notes.astype(str).apply(lambda x: x.replace('Scale',' '))

# Remove duplicated spaces
notes.Notes = notes.Notes.astype(str).apply(lambda x: " ".join(x.split())) 

# ------------------------------------------------------
# Notes between discharge and follow-up date (remaining)
# ------------------------------------------------------

# Notes between discharge and follow-up day

dd = notes[(notes.ContactDTS.astype('datetime64[ns]') >= notes.HospitalDischargeDTS.astype('datetime64[ns]'))]

dd.Notes = dd.Notes.astype(str).str.lower().apply(lambda x: x.replace('mrs','rankin'))


def notes_select(o, col):   
 
    # Select notes closer to follow-up
    
    o.ContactDTS = o.ContactDTS.astype('datetime64[ns]')

    o[col+'_days'] = o.Date_mrs_post_discharge.astype('datetime64[ns]') - o.ContactDTS.astype('datetime64[ns]')

    o[col+'_days'] = o[col+'_days'].astype(str).str.replace(' days','').astype(int)
    
   
    # Group notes per order of time and note id
    # ------------------------------------------------------

    o = o.sort_values(['ContactDTS',"NoteID","LineNBR"], ascending=[True,True,True])

    #added
    o.Notes = ' ' + o.Notes.astype(str) + ' '

    # join notes by note date
    
    o = o.groupby(['PatientID', 'HospitalAdmitDTS', 'HospitalDischargeDTS', 
                   'Date_mrs_post_discharge', 'ContactDTS', col+'_days', "NoteID", 'mrs_90days']).Notes.sum().reset_index()

    o.Notes = o.Notes.astype(str).str.lower().apply(lambda x: x.replace('remained at',' '))

    o.Notes = o.Notes.astype(str).str.lower().apply(lambda x: x.replace('of',' '))

    o.Notes = o.Notes.astype(str).str.lower().apply(lambda x: x.replace('premorbid rankin',' '))

    # Repeat removing duplicated spaces and words in a row

    # Remove duplicated spaces
    o.Notes = o.Notes.apply(lambda x: " ".join(x.split())) 

    # Remove duplicated words in row
    o.Notes = o.Notes.astype(str).apply(lambda x: re.sub(r'\b(\w+)( \1\b)+', r'\1', x))
    
    o = o.drop(columns='ContactDTS')

    
    return o

dd = notes_select(dd, 'DD')

#Find expression
def find_(s, d, ntokens):     
    s = pd.Series(s).str.extractall('('+ d + '(:?\??[?)(\s?)([\w]).{1,'+ntokens+'})').iloc[:,0].str.cat(sep=' -#- ')
    return s

########################################################################
# Extract last mrs for each NoteID, then select the closest to follow-up
#-----------------------------------------------------------------------

# last mrs in the NoteID

a = dd

a['rankin'] =  pd.Series(a['Notes'].astype(str).str.lower()).apply(lambda x: find_(x,'rankin',ntokens='50'))

a = a[a.rankin.astype(str)!='']

# select last of noteID

a['last'] = a.rankin.apply(lambda x: re.search(r"([^#]*$)", x)[0])

a['mrs'] = np.nan

a['mrs'][a['last'].astype(str).str.contains('rankin 0')] = 0
a['mrs'][a['last'].astype(str).str.contains('rankin 1')] = 1
a['mrs'][a['last'].astype(str).str.contains('rankin 2')] = 2
a['mrs'][a['last'].astype(str).str.contains('rankin 3')] = 3
a['mrs'][a['last'].astype(str).str.contains('rankin 4')] = 4
a['mrs'][a['last'].astype(str).str.contains('rankin 5')] = 5

a['mrs_is'] = np.nan
a['mrs_is'][a['rankin'].astype(str).str.contains('rankin is 0')] = 0
a['mrs_is'][a['rankin'].astype(str).str.contains('rankin is 1')] = 1
a['mrs_is'][a['rankin'].astype(str).str.contains('rankin is 2')] = 2
a['mrs_is'][a['rankin'].astype(str).str.contains('rankin is 3')] = 3
a['mrs_is'][a['rankin'].astype(str).str.contains('rankin is 4')] = 4
a['mrs_is'][a['rankin'].astype(str).str.contains('rankin is 5')] = 5            

a['mrs'][a['mrs_is'].astype(str)!='nan'] = a['mrs_is'][a['mrs_is'].astype(str)!='nan']

# select closest to follow-up

aux = a.groupby(['PatientID','HospitalAdmitDTS', 'HospitalDischargeDTS', 'Date_mrs_post_discharge']).DD_days.min().reset_index()

a = pd.merge(a[['PatientID', 'HospitalAdmitDTS', 'HospitalDischargeDTS',
       'Date_mrs_post_discharge', 'DD_days', 'mrs_90days',
       'rankin',  'mrs']].drop_duplicates(), aux, on=list(aux.columns), how='outer')

a = a[a.DD_days<=30].drop(columns='DD_days').drop_duplicates()

cols = ['PatientID', 'HospitalAdmitDTS', 'HospitalDischargeDTS',
       'Date_mrs_post_discharge','mrs']

aa = pd.read_csv(os.path.join(path,'extractions1.csv'))

a = pd.concat([a[cols].drop_duplicates(),aa[cols].drop_duplicates()],axis=0)

a=a.rename(columns={'mrs':'mrs_extractions','mrs_90days':'mrs'})

a= a[~(a.mrs_extractions.astype(str)=='nan')]

a.to_csv(os.path.join(path,'extractions_demo.csv'), index=False)


