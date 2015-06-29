from sklearn.svm import LinearSVC
from sklearn.grid_search import GridSearchCV
from scipy.stats import pearsonr

def selectByp_value():
     for i in [0,1,2,3,4]:
          train_df,id_columns = read_df(geno,pheno,part=i,select_feature = False)
          tr_df = train_df.loc[tr,:]
          ## calculate p-value with balanced classes
          if i ==0:
               tr_df_,labels_,tr_df_index = balance_type(tr_df,labels_s,label_type)
          else:
               tr_df_,labels_,tr_df_index = balance_type(tr_df,labels_s,label_type,tr_df_index)
          ## encode labels
          labels_ = le.fit_transform(labels_)
          if i ==0:
               p_value_ = get_corr(tr_df_,labels_)
               p_value = p_value_
               print("----initial p value list-----")
          else:
               p_value_=get_corr(tr_df_,labels_)
               p_value_.index = p_value_.index+i*1000000
               p_value = p_value.append(p_value_)
               print('----append p value list----')
     p_value['index']=p_value.index
     p_value = p_value.sort('p_value',ascending = 0)
     p_value.index =range(p_value.shape[0])
     select_feat = p_value.loc[:100000-1]
     select_index = sorted(list(select_feat['index']))
     select_index = map(lambda x:x+4,select_index)
     chosen_data = read_df(geno,pheno,feature_index = select_index, select_feature = True )
     chosen_data = pd.DataFrame(chosen_data[0])
     return chosen_data

def selectByL1():
     tuned_parameters = [{ 'C': [1, 5,10, 20]}]
     C=[1,5,10,20]
     l1 = LinearSVC(C=C, penalty="l1",class_weight='auto',dual=False)
     for i in [0,1,2,3,4]:
          train_df,id_columns = read_df(geno,pheno,part=i,select_feature = False)
          tr_df = train_df.loc[tr,:]
          te_df = train_df.loc[te,:]
          tr_df.index =range(tr_df.shape[0])
          tr_df_tr = tr_df.loc[tr_,:]
          tr_df_va = tr_df.loc[va_,:]
          tr_labels_tr = tr_labels[tr_]
          tr_labels_va = tr_labels[va_]
          clf = GridSearchCV(LinearSVC(penalty='l1',class_weight='auto',dual=False), tuned_parameters, cv=5)
          clf.fit(tr_df_tr,tr_labels_tr)
          clf.best_params_
          param = clf.best_params_
          l1 = LinearSVC(C=param['C'], penalty="l1",class_weight='auto',dual=False)
          tr_df_l1 = l1.fit_transform(tr_df_tr,tr_labels_tr)
          train_df_l1 = l1.transform(train_df)
          if i ==0:
               train_data = train_df_l1
               print("----initial l1-----")
               train_data.shape
          else:
               train_data = np.append(train_data,train_df_l1,axis=1)
               #transfomer.append(l1)
               #test_data[id_columns] = te_df
               print('----append l1----')
               train_data.shape
          return train_data
