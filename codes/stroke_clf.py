import pandas as pd
import numpy as np
from sklearn import preprocessing
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import StratifiedKFold
from stroke_func import *
from select import *
from merged import *
from test_clf import *

file = open('stroke.filtered.dominant.raw')
geno = 'stroke.filtered.dominant.raw'
pheno = 'phenotype.txt'
 
le = LabelEncoder()
labels = get_label()
labels_t= le.fit_transform(labels)

skf=list(StratifiedKFold(labels_t,n_folds = 8))
tr,te = skf[0]

## replace 'SAO' with other labels to test other classes
labels_s = binary_label(labels,'SAO','NON-SAO')
labels_s_t = le.fit_transform(labels_s)
label_type = 'SAO'

tr_labels = labels_s_t[tr]
te_labels = labels_s_t[te]

skf_2 = list(StratifiedKFold(labels_s_t,8))
tr_,va_ = skf_2[0]
tr_labels_tr = tr_labels[tr_]
tr_labels_va = tr_labels[va_]

#tr_df = train_df.loc[tr,:]
#te_df = train_df.loc[te,:]

selection= 'p_value' ## or ='l1'

## select by p-value:
if selection=='p_value':
     chosen_data = selectByp_value()
elif selection =='l1':
     chosen_data = selectByL1()

#chosen_data.to_csv(path,quoting = 3)
train_ = chosen_data.loc[tr,:][tr_,]
valid_=chosen_data.loc[tr,:][va_,]
test_ = chosen_data.loc[te,:]

## merge with phenotype data
merged_array = merge()
merged_data = np.append(train_data,merged_array,axis=1)
merged_train = merged_data[tr,][tr_,]
merged_valid = merged_data[tr,][va_,]
merged_test = merged_data[te,]

## test logistic regression
C_list = {0.01, 0.1, 1.0, 10}
test_logit_classifier(train_,valid_,tr_labesl_tr,tr_labels_tr,C_list, pca=False)
test_logit_classifier(train_,valid_,tr_labesl_tr,tr_labels_tr,C_list, pca=True)
## with merged data
test_logit_classifier(merged_train,merged_valid,tr_labesl_tr,tr_labels_tr,C_list, pca=False)
test_logit_classifier(merged_train,merged_valid,tr_labesl_tr,tr_labels_tr,C_list, pca=True)

## test gbc classifier
param_n = {10, 20, 30, 40, 50, 100}
test_gbc_classifier(train_,valid_,tr_labesl_tr,tr_labels_tr,param_n, pca=False)
test_gbc_classifier(train_,valid_,tr_labesl_tr,tr_labels_tr,param_n, pca=True)
## with merged data
test_logit_classifier(merged_train,merged_valid,tr_labesl_tr,tr_labels_tr,param_n, pca=False)
test_logit_classifier(merged_train,merged_valid,tr_labesl_tr,tr_labels_tr,param_n, pca=True)

## use the classifier with the best performance to obtain predicted probabilities
best_clf =
best_param =
pred_test_proba = best_clf.fit(train, tr_labels_tr)
pred_test_proba_laa = pd.DataFrame(pred_test_proba_laa)
pred_test_proba_laa.columns = [label_type,'NON-'+label_type]
pred_test_proba_laa.to_csv(path, quoting =3, index=False)
