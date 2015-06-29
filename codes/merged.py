import numpy as np
import padnas as pd
import stroke_func

def merge():
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
     merged = merged.drop(['FID','IID','dbGaP_Subject_ID','subject_id','studyID'],axis=1)
     labels = merged['CCStype']
     merged = merged.drop(['CCStype'],1)     
     merged_imp = miximputer(merged)    
     merged_cat = categorize(merged_imp)
     merged_array = np.array(merged_cat)
     return merged_array
