from sklearn.metrics import confusion_matrix as cm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier

def test_logit_classifier(X_train, X_valid, X_test, y_train, y_valid, y_test,param_c, pca = False):
     if pca = True:
          pca_transform = PCA()
          X_train = pca_transform.fit_transform(X_train)
          X_valid = pca_transform.transform(X_valid)
          X_test = pca_transform.transform(X_test)
          print("--PCA-transformed data--")
     for c in param_c:
          clf =LogisticRegression(C=c,class_weight='auto',penalty='l1',dual=False)
          clf.fit(X_train,y_train)
          pred = clf.predict(X_valid)
          print("Validation data:")
          print("  parameter C = {num}".format(num = c))
          cm(pred,y_valid)
          pred_test = clf.predict(X_test)
          print("Test data:)
          print("  parameter C = {num}".format(num=c))
          cm(pred_test,y_test)

def test_rf_classifier(X_train, X_valid, X_test, y_train, y_valid, y_test,param_n, pca = False):
     if pca = True:
          pca_transform = PCA()
          X_train = pca_transform.fit_transform(X_train)
          X_valid = pca_transform.transform(X_valid)
          X_test = pca_transform.transform(X_test)
          print("--PCA-transformed data--")
     for n_estimators in param_n:
          clf =RandomForestClassifier(n_estimators = n_estimators)
          clf.fit(X_train,y_train)
          pred = clf.predict(X_valid)
          print("Validation data:")
          print("  parameter n_estimators = {num}".format(num = c))
          cm(pred,y_valid)
          pred_test = clf.predict(X_test)
          print("Test data:)
          print("  parameter n_estimators = {num}".format(num=c))
          cm(pred_test,y_test)

def test_gbc_classifier(X_train, X_valid, X_test, y_train, y_valid, y_test,param_n, pca = False):
     if pca = True:
          pca_transform = PCA()
          X_train = pca_transform.fit_transform(X_train)
          X_valid = pca_transform.transform(X_valid)
          X_test = pca_transform.transform(X_test)
          print("--PCA-transformed data--")
     for n_estimators in param_n:
          clf =GradientBoostingClassifier(loss='deviance', learning_rate= 0.1, n_estimators = n_estimators)
          clf.fit(X_train,y_train)
          pred = clf.predict(X_valid)
          print("Validation data:")
          print("  parameter n_estimators = {num}".format(num = c))
          cm(pred,y_valid)
          pred_test = clf.predict(X_test)
          print("Test data:)
          print("  parameter n_estimators = {num}".format(num=c))
          cm(pred_test,y_test)
