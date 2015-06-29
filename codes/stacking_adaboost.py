from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix as cm

## class "CE" stacking:
clf = LogisticRegression(C=0.01, penalty='l1', class_weight = 'auto')
clfs = [clf]
blend_train = np.zeros((train_.shape[0],len(clfs)))
blend_train_pca = np.zeros((train_pca.shape[0],len(clfs)))
blend_test = np.zeros((test_.shape[0],len(clfs)))
blend_test_pca = np.zeros((test_.shape[0],len(clfs)))

skf = list(StratifiedKFold(tr_labels_tr,10))

# non-transformed
for j,clf in enumerate(clfs):
     print 'Training classifier [%s]'%(j):
          blend_test_ = np.zeros((test_.shape[0],len(skf)))
          for i,(train_index,cv_index) in enumerate(skf):
               print 'Fold [%s]'%(i)
               X_train = train_[train_index]
               Y_train = tr_labels_tr[train_index]
               X_cv = train_[cv_index]
               Y_cv = tr_labels_tr[cv_index]
               clf.fit(X_train, Y_train)
               blend_train[cv_index,j] = clf.predict(X_cv)
               blend_test_[:,j] = clf.predict(test_)
          blend_test[:,j] = blend_test_j.mean(1)
# pca
for j,clf in enumerate(clfs):
     print 'Training classifier [%s]'%(j):
          blend_test_pca_ = np.zeros((test_pca.shape[0],len(skf)))
          for i,(train_index,cv_index) in enumerate(skf):
               print 'Fold [%s]'%(i)
               X_train = train_pca[train_index]
               Y_train = tr_labels_tr[train_index]
               X_cv = train_pca[cv_index]
               Y_cv = tr_labels_tr[cv_index]
               clf.fit(X_train, Y_train)
               blend_train_pca[cv_index,j] = clf.predict(X_cv)
               blend_test_pca[:,j] = clf.predict(test_pca)
          blend_test_pca[:,j] = blend_test_pca.mean(1)

blend_test_train= np.append(blend_train,blend_train_pca,axis=1)
blend_test_test = np.append(blend_test,blend_test_pca,axis=1)

bclf = LogisticRegression()
pred = bclf.fit(blend_test_train,tr_labels_tr)
cm(pred,te_labels)

## class "OTHER" adaboost:
base_clf = DecisionTreeClassifier(criterion='entropy', splitter='best', max_depth=None,class_weight='auto')
n_estimators = [10, 20, 30, 40, 50]
for n in n_estimators:
     clf = AdaBoostClassifier(base_estimator=base_clf, n_estimators=n, learning_rate=1.0, algorithm='SAMME.R', random_state=None)
     clf.fit(train_,tr_labels_tr)
     pred = clf.predict(valid_)
     print("Validation data:")
     cm(pred, tr_labels_va)
     print("with {n} estimators:".format(n=n))
     pred_test = clf.predict(test_)
     print("Test data:")
     cm(pred_test, te_labels)
     

