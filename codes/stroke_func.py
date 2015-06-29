import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import Imputer
from sklearn import preprocessing
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Imputer
from scipy.stats import pearsonr
from sklearn.cross_validation import StratifiedKFold

## get the labels of dataset
def get_label():
     file = open('stroke.filtered.dominant.raw')
     train_df = []
     first=1
     i=1
     for line in file:
          if first:
               id_columns = line
               id_columns = id_columns.split(" ")
               id_columns = id_columns[:2]
               id_columns = list(id_columns)
               first = 0
          else:
               line = line.split(" ")
               line = line[:2]
               line = list(line)
               train_df.append(line)
               print("finished {number}th row".format(number = i))
               i+=1
     train_df = pd.DataFrame(train_df,index=None)
     train_df.columns = id_columns
     train_df['FID']= train_df['FID'].astype(int)
     phenotype = pd.read_csv("phenotype.txt",header = 0, index_col = False, delimiter = '\t')
     merged = pd.merge(train_df,phenotype,left_on='FID', right_on='subject_id', how = 'inner')
     labels = merged['CCStype']
     return labels

## make two classes balanced and obtain the dataframe of balanced classes
def balance_type(train_df,label_df,label_1,index = None):
     df_1 = train_df.loc[(label_df==label_1)]
     df_2 = train_df.loc[(label_df!=label_1)]
     df_1_index = train_df.loc[(label_df==label_1)].index
     len_df_1 = len(df_1_index)
     df_2_index = train_df.loc[(label_df!=label_1)].index
     len_df_2 = len(df_2_index)
     if index==None:
          df_2_index_ = random.sample(df_2_index,len_df_1)
          non_train_df = train_df.loc[df_2_index_,:]
          df_1 = df_1.append(non_train_df)
          df_1 = df_1.sort_index()
          df_1_index = list(df_1_index)
          df_1_index.extend(df_2_index_)
          df_1_index.sort()
          labels = label_df.loc[df_1_index]
     else:
          df_1_index = index
          df_1 = train_df.loc[index]
          labels = label_df.loc[df_1_index]
     return df_1, labels, df_1_index

##binarize labels
def binary_label(column, label1, label2):
                  response = column.copy()
                  response.loc[(response!=label1)] = label2
                  return response

## get correlation of each feature and the labels
def get_corr(train_df_,labels):
     corr_p = []
     for id_ in train_df_.columns:
          s = train_df_[id_].astype(int)
          s = np.array(s)
          p_value = pearsonr(s,labels)[0]
          if np.isnan(p_value):
               p_value = 0
          elif p_value<0:
               p_value = 0-p_value
          corr_p.append(p_value)
     corr_p=pd.DataFrame(corr_p)
     corr_p['features'] = train_df_.columns
     corr_p.columns = ['p_value','features']
     return corr_p

## read data
def read_df(geno_file_path,pheno_file_path,feature_index=None,part=None, select_feature = False):
     end = 0
     f=0
     i = part
     file = open(geno_file_path,'r')
     phenotype = pd.read_csv(pheno_file_path,header=0,index_col = False, delimiter = '\t')
     train_df = []
     imp = Imputer(missing_values ='NaN',strategy = 'most_frequent',axis=0)
     first = 1
     for line in file:
          if first:
               id_columns = line
               id_columns = id_columns.split(" ")
               if select_feature == False:
                    if (1000004+1000001*i)>(len(id_columns)-1):
                         end = 1
                         max_len = len(id_columns)-1
                         id_columns = id_columns[4+1000001*i:max_len]
                         id_columns = map(lambda x:x.strip(),id_columns)
                    else:
                         id_columns = id_columns[4+1000001*i:1000004+1000001*i]
                         id_columns = list(id_columns)
               elif select_feature == True:
                    id_columns = list(id_columns[i] for i in feature_index)
                    print("selected {number} features".format(number = len(id_columns)))
               first= 0
          else:
               line = line.split(" ")
               if select_feature == False:
                    if end:
                         line = line[4+1000001*i:max_len]
                         line = map(lambda x:x.strip(),line)
                         line = map(lambda x:x if x!= 'NA' else 'NaN',line)
                         train_df.append(line)
                    else:
                         line = line[4+1000001*i:1000004+1000001*i]
                         line = list(line)
                         line = map(lambda x:x if x!= 'NA' else 'NaN',line)
                         train_df.append(line)
                         print("finished {number}th sample".format(number = f))
               elif select_feature == True:
                    line = list(line)
                    line = list(line[i] for i in feature_index)
                    line = map(lambda x:x if x!= 'NA' else 'NaN',line)
                    train_df.append(line)
                    print("finished {number}th sample".format(number = f))
               f += 1
     train_df = imp.fit_transform(train_df)
     print("Imputed")
     train_df = pd.DataFrame(train_df,index=None)
     train_df.columns = id_columns
     file.close()
     return train_df,id_columns

## impute phenotype data where the value is not available, impute with most frequent value
def miximputer(df):
     colnames = df.columns
     for col in colnames:
          l = []
          mf = list(df[col].value_counts().index)
          for i in df[col]:
               if isinstance(i,float):
                    if np.isnan(i):
                         i = mf[0]
               l.append(i)
          df[col]=l
     return df

## categorize phenotype features in phenotype data
def categorize(df):
     le = LabelEncoder()
     colnames = df.columns
     for col in colnames:
          if not isinstance(df[col].loc[0],float):
               df[col] = le.fit_transform(df[col])
     return df
