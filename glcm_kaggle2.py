# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 16:49:43 2018

@author: karde
"""

import cv2 as cv
from skimage import feature
import numpy as np
from matplotlib import pyplot as plt
import glob
import pandas as pd
from skimage.feature import greycomatrix, greycoprops
from featureDescriptor import LocalBinaryPatterns as LBP


from sklearn.metrics import average_precision_score as aps
from sklearn.metrics import precision_score as ps
from sklearn.metrics import recall_score as rs
from sklearn.metrics import f1_score as f1s
from sklearn.metrics import classification_report as creport

from sklearn.linear_model import LinearRegression

pathname = "D:\\4th_year\\Project\\Datasets\\IDC_regular_ps50_idx5\\*\\";


data = []
labels = []

c = 1
lbpModel = LBP.LocalBinaryPatterns(24, 8)

for folderName in ["0", "1"]:
    for imagePath in glob.glob(pathname+folderName+"\\*.png"):#paths.list_images(args["training"]):
	   # load the image, convert it to grayscale, and describe it
       print(str(c))
       image = cv.imread(imagePath)
       gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
       hist = greycomatrix(gray, [5], [0], 256, symmetric=False, normed=True)
       contrast = greycoprops(hist, 'contrast')
       dissimilarity = greycoprops(hist, 'dissimilarity')
       homogeneity = greycoprops(hist, 'homogeneity')
       ASM = greycoprops(hist, 'ASM')
       energy = greycoprops(hist, 'energy')
       correlation = greycoprops(hist, 'correlation')
       hist = hist[:,:,0,0]
       hist = lbpModel.get_histogram(hist)
       gray = feature.local_binary_pattern(gray, 24, 8, method="uniform")
       #lbpModel = LBP.LocalBinaryPatterns(24, 8)
       #gray = lbpModel.lbp(hist, hist.shape)
       #hist, _ = np.histogram(hist.ravel(), bins = 256)
       #hist = hist.astype("float")
       #hist /= (hist.sum() + 1e-07)
       #hist1 = lbpModel.get_histogram(gray)
       #hist = np.concatenate((hist, hist1), axis = 0)
       hist = np.append(hist, np.array([contrast, dissimilarity, homogeneity, ASM, energy, correlation]))
       '''np.append(dissimilarity)
       np.append(homogeneity)
       np.append(ASM)
       np.append(energy)
       np.append(correlation)'''
       #plt.plot(hist.flatten()) 
       #plt.show()
       means, std = cv.meanStdDev(image)
       hist = np.concatenate((hist, means.transpose().flatten(), std.transpose().flatten()), 
                             axis = 0)
       labels.append(int(imagePath.split("\\")[-2]))
       data.append(hist)
       c = c + 1


X = pd.DataFrame(data = data, dtype='float32').values
#X.to_csv('all_data_2.csv')

X = pd.read_csv('all_data_2.csv')
X = pd.DataFrame(data = X[:, 1:], dtype='float32').values

#X=X.values

y = pd.DataFrame(data=labels).values

#trainFrame = pd.concat([X, y], axis = 1)

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Encoding the Dependent Variable
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

##Accumulating all classifier results

from sklearn.linear_model import LogisticRegression#0.72 #0.75
lr = LogisticRegression(random_state = 0)
lr.fit(X_train, y_train)           

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
knn.fit(X_train, y_train)

from sklearn.svm import LinearSVC#0.74 #0.77
model = LinearSVC(C=100.0, random_state=0)
model.fit(X_train, y_train)

from sklearn.svm import SVC
ksvm = SVC(kernel = 'rbf', random_state = 0)
ksvm.fit(X_train, y_train)

lsvm = SVC(kernel='linear', random_state = 0)
lsvm.fit(X_train, y_train)

from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X_train, y_train)

from sklearn.tree import DecisionTreeClassifier#0.67 #.68
dt = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
dt.fit(X_train, y_train)

from sklearn.ensemble import RandomForestClassifier#0.74 #0.78
rf = RandomForestClassifier(n_estimators = 500, criterion = 'entropy', random_state = 42)
rf.fit(X_train, y_train)

from sklearn.ensemble import ExtraTreesClassifier
et = ExtraTreesClassifier(n_estimators = 500, criterion = 'entropy', random_state = 42)
et.fit(X_train, y_train)

## Selecting features based on relative importance
print(rf.feature_importances_)

prob = rf.feature_importances_ #random forest

prob = xgb.feature_importances_  #XGBoost defined later

#saving the indexes
imp_index = []
for i in range(0, len(prob)):
    if(prob[i] > 1e-2):
        imp_index.append(i)
#New feature vector
X_train_ = []
X_test_ = []
for row in range(0, X_train.shape[0]):
    temp = [X_train[row, :][i] for i in imp_index]
    X_train_.append(temp)
    
for row in range(0, X_test.shape[0]):
    temp1 = [X_test[row, :][i] for i in imp_index]
    X_test_.append(temp1)
    
X_train_ = pd.DataFrame(data = X_train_, dtype='float32').values
X_test_ = pd.DataFrame(data = X_test_, dtype='float32').values

#training new classifier
rf1 = RandomForestClassifier(n_estimators = 500, criterion = 'entropy', random_state = 42)
rf1.fit(X_train_, y_train)


model1 = LinearSVC(C=100.0, random_state=0)
model1.fit(X_train_, y_train)


lsvm1 = SVC(kernel='linear', random_state = 0)
lsvm1.fit(X_train, y_train)


from xgboost import XGBClassifier
xgb1 = XGBClassifier()
xgb1.fit(X_train_, y_train)


y_predict = rf1.predict(X_test_) 
y_predict = model1.predict(X_test_)
y_predict = lsvm1.predict(X_test_)
y_predict = xgb1.predict(X_test_)

y_score = rf1.predict_proba(X_test_)
score = []
for row in y_score:
    if row[0] > row[1]:
        score.append(row[0])
    else:
        score.append(row[1])
         
score = pd.DataFrame(data = score, dtype ='float64').values
avg = aps(y_test, score)
rsc = rs(y_test, y_predict)
psc = ps(y_test, y_predict)
f1sc = f1s(y_test, y_predict)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_predict)
tn, fp, fn, tp = confusion_matrix(y_test, y_predict).ravel()
specificity = tn / (tn+fp)
print(specificity)
acc = (cm[0, 0] + cm[1, 1])/cm.sum()
print('accuracy : ' + str(acc))



from sklearn.model_selection import learning_curve
train_sizes, train_scores, test_scores = learning_curve(model1, 
                                                        X_train_, 
                                                        y_train,
                                                        # Number of folds in cross-validation
                                                        cv=None,
                                                        # Evaluation metric
                                                        scoring='accuracy',
                                                        # Use all computer cores
                                                        n_jobs=-1, 
                                                        # 50 different sizes of the training set
                                                        train_sizes=np.linspace(0.01, 1.0, 5))

# Create means and standard deviations of training set scores
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)

# Create means and standard deviations of test set scores
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# Draw lines
plt.plot(train_sizes, train_mean, '--', color="#111111",  label="Training score")
plt.plot(train_sizes, test_mean, color="#111111", label="Cross-validation score")

# Draw bands
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="#DDDDDD")
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color="#DDDDDD")

# Create plot
plt.title("Learning Curve")
plt.xlabel("Training Set Size"), plt.ylabel("Accuracy Score"), plt.legend(loc="best")
plt.tight_layout()
plt.show()
plt.savefig('svm_plot.jpg')

from sklearn.model_selection import validation_curve
param_range = np.arange(1, 200, 1)
train_scores, test_scores = validation_curve(model1, 
                                             X_train_, 
                                             y_train, 
                                             param_name="max_iter", 
                                             param_range=param_range,
                                             cv=None, 
                                             scoring="accuracy", 
                                             n_jobs=-1)


# Calculate mean and standard deviation for training set scores
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)

# Calculate mean and standard deviation for test set scores
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# Plot mean accuracy scores for training and test sets
plt.plot(param_range, train_mean, label="Training score", color="black")
plt.plot(param_range, test_mean, label="Cross-validation score", color="dimgrey")

# Plot accurancy bands for training and test sets
plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, color="gray")
plt.fill_between(param_range, test_mean - test_std, test_mean + test_std, color="gainsboro")

# Create plot
plt.title("Validation Curve With Linear SVM")
plt.xlabel("Iterations")
plt.ylabel("Accuracy Score")
plt.tight_layout()
plt.legend(loc="best")
plt.show()
plt.savefig('svm_val1.jpg')

from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

gpc = GaussianProcessClassifier(copy_X_train = False, random_state = 0).fit(X_train[0:10000, 0:10000], y_train[0:10000, :])
  
from xgboost import XGBClassifier
xgb = XGBClassifier()
xgb.fit(X_train, y_train)



pred = lr.predict(X_test)

pred = knn.predict(X_test)

pred = model.predict(X_test)

pred = ksvm.predict(X_test)

pred = nb.predict(X_test)

pred = dt.predict(X_test)

pred = rf.predict(X_test)

pred = gpc.predict(X_test[0:8000])

pred = xgb.predict(X_test)
'''
262 features for xgb --> 0.8509 4:1 partition
262 0.84767 2:1
'''

from sklearn.metrics import confusion_matrix


cm = confusion_matrix(y_test, pred)
acc = (cm[0, 0] + cm[1, 1])/cm.sum()
print('accuracy : ' + str(acc)) #0.845

from sklearn.metrics import accuracy_score
score = accuracy_score(y_test, pred)
print(str(score))

#0.80 random forest

#0.795 before
from sklearn.feature_selection import RFE
rfe = RFE(rf)
rfe = rfe.fit(X_train, y_train)

print(rfe.support_)
print(rfe.ranking_)

#Ensemble Learning

from sklearn.model_selection import StratifiedKFold

def Stacking(model,train,y,test,n_fold):
   #train = pd.DataFrame(data = train, dtype='float32')
   #test = pd.DataFrame(data = test, dtype='float32')
   print(type(train))
   folds=StratifiedKFold(n_splits=n_fold,random_state=1)
   test_pred=np.empty((0,1),float)  #test.shape[0]instead of 0
   train_pred=np.empty((0,1),float)
   for train_indices,val_indices in folds.split(train,y):
      x_train,x_val=train.iloc[train_indices].values,train.iloc[val_indices].values
      y_train,y_val=y.iloc[train_indices].values,y.iloc[val_indices].values

      model.fit(X=x_train,y=y_train)
      train_pred=np.append(train_pred,model.predict(x_val))
      
   model.fit(train, y)
   test_pred=np.append(test_pred,model.predict(test))
   return test_pred.reshape(-1,1),train_pred


# First Model is Random Forest
rf2 = RandomForestClassifier(n_estimators = 500, criterion = 'entropy', random_state = 42)   

x_train = pd.DataFrame(data = X_train_, dtype ='float32')
x_test = pd.DataFrame(data = X_test_, dtype='float32')
y_train_ = pd.DataFrame(data = y_train, dtype='int64')

test_pred1 ,train_pred1=Stacking(model=rf2,n_fold=10, train=x_train,test=x_test,y=y_train_)

rf2.fit(x_train, y_train_)
train_pred1 = rf2.predict(x_train)
test_pred1 = rf2.predict(x_test)

train_pred1=pd.DataFrame(train_pred1)
test_pred1=pd.DataFrame(test_pred1)

# Second Model is Linear SVC
model2 = LinearSVC(C=100.0, random_state=0)
test_pred2 ,train_pred2=Stacking(model=model2,n_fold=10,train=x_train,test=x_test,y=y_train_)

train_pred2=pd.DataFrame(train_pred2)
test_pred2=pd.DataFrame(test_pred2)


df = pd.concat([train_pred1, train_pred2], axis=1)
df_test = pd.concat([test_pred1, test_pred2], axis=1)

# Finally using XGBoost
emodel = XGBClassifier(random_state=1)

emodel.fit(df.values,y_train)
emodel.score(df_test.values, y_test)    #0.856409

emodel.fit(x_train, y_train_)
train_pred2 = emodel.predict(x_train)
test_pred2 = emodel.predict(x_test)
train_pred2=pd.DataFrame(train_pred2)
test_pred2=pd.DataFrame(test_pred2)

df = np.c_[train_pred1, train_pred2]
lr = LinearRegression()
lr.fit(df, y_train)

df_test = np.c_[test_pred1, test_pred2]
pred_val = lr.predict(df_test)
pred = np.array([])
for i in range(0, len(pred_val)):
    if(pred_val.item(i) < 0.5):
        pred = np.append(pred, [0])
    else:
        pred = np.append(pred, [1])

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, pred)
acc = (cm[0, 0] + cm[1, 1])/cm.sum()
print('accuracy : ' + str(acc)) #0.845

'''
stacked 262: 0.84586
stacked 85: 0.85327
stackes 36: 0.85327
'''

from sklearn.linear_model import LogisticRegression
lre = LogisticRegression(random_state = 0)
lre.fit(df.values, y_train) 
lre.score(df_test.values, y_test) 

import pickle

filename = './models/best_lgb1.sav'
pickle.dump(best_lgb, open(filename, 'wb'))

