# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 11:18:49 2019

@author: user
"""

import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
%matplotlib qt
import glob
import mahotas as mt
from skimage.feature import greycomatrix, greycoprops, hog, local_binary_pattern
from sklearn.model_selection import cross_validate
from scipy.stats import pearsonr, skew, kurtosis
import seaborn as sns
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

def get_histogram(output):
        hist, _ = np.histogram(output.ravel(), bins=256)
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-07)
        return hist
    
def plot_imp(feature_dataframe):
    trace = go.Scatter(
        y = feature_dataframe['RF feature importances'].values,
        x = feature_dataframe['features'].values,
        mode='markers',
        marker=dict(
            sizemode = 'diameter',
            sizeref = 1,
            size = 25,
    #       size= feature_dataframe['AdaBoost feature importances'].values,
            #color = np.random.randn(500), #set color equal to a variable
            color = feature_dataframe['RF feature importances'].values,
            colorscale='Portland',
            showscale=True
        ),
        text = feature_dataframe['features'].values
    )
    data = [trace]
    
    layout= go.Layout(
        autosize= True,
        title= 'RF Feature Importance',
        hovermode= 'closest',
    #     xaxis= dict(
    #         title= 'Pop',
    #         ticklen= 5,
    #         zeroline= False,
    #         gridwidth= 2,
    #     ),
        yaxis=dict(
            title= 'Feature Importance',
            ticklen= 5,
            gridwidth= 2
        ),
        showlegend= False
    )
    fig = go.Figure(data=data, layout=layout)
    py.plot(fig,filename='har_hu', image = 'png', image_filename='har_glcm_hu_feat_rf')


pathname = "D:\\DL_test\\Project\\dataset\\IDC_regular_ps50_idx5\\*\\";

channel = ['R-mean', 'G-mean', 'B-mean', 'R-sd', 'G-sd', 'B-sd']

# SIFT features extraction
data = []
labels = []
c = 1
for folderName in ["0", "1"]:
    for imagePath in glob.glob(pathname+folderName+"\\*.png"):
        print(str(c))
        hist = ''
        image = cv2.imread(imagePath)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sift = cv2.xfeatures2d.SIFT_create()
        kp, des = sift.detectAndCompute(gray,None)
        #img=cv2.drawKeypoints(gray,kp, None)
        if des is None:
            hist = get_histogram(np.zeros(128, dtype = 'float'))
            #hist = np.full((128, 1), 255, dtype = 'float').flatten()
        else:
            hist = get_histogram(des)
        means, std = cv2.meanStdDev(image)
        hist = np.concatenate((hist, means.transpose().flatten(), std.transpose().flatten()), axis = 0)
        labels.append(int(imagePath.split("\\")[-2]))
        data.append(hist)
        c += 1

cols = ['sift_feat{}'.format(i) for i in range(0, 256)]
cols.extend(channel)

X = pd.DataFrame(data = data, dtype='float32')
X.fillna( method ='ffill', inplace = True)

colormap = plt.cm.RdBu
plt.figure(figsize=(14,12))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(X.astype(float).corr(),linewidths=0.1,vmax=1.0, 
            square=True, cmap=colormap, linecolor='white', annot=True)
X = X.values
y = pd.DataFrame(data=labels, columns = ['class_label']).values

feature_data = pd.concat([X, y], axis = 1)
feature_data.to_csv('sift.csv')
#X.fillna( method ='ffill', inplace = True) 

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators = 700, criterion = 'entropy', random_state = 42)
rf.fit(X_train, y_train)



rf_features = rf.feature_importances_
x_train = pd.DataFrame(data = X_train, columns = cols)
cols = x_train.columns.values
# Create a dataframe with features
feature_dataframe = pd.DataFrame( {'features': cols,
     'feature importances': rf_features
    })
plot_imp(feature_dataframe)



from xgboost import XGBClassifier
xgb = XGBClassifier()
xgb.fit(X_train, y_train)

xgb_features = xgb.feature_importances_
feature_dataframe = pd.DataFrame( {'features': cols,
     'feature importances': xgb_features
    })
plot_imp(feature_dataframe)


from catboost import CatBoostClassifier as cbc
cb = cbc(depth = 10, task_type = 'GPU', learning_rate = 0.05)
cb.fit(X_train, y_train.ravel())

cb_features = cb.feature_importances_
x_train = pd.DataFrame(data = X_train, columns = cols)
cols = x_train.columns.values
# Create a dataframe with features
feature_dataframe = pd.DataFrame( {'features': cols,
     'CB feature importances': cb_features
    })
plot_imp(feature_dataframe)

from sklearn.svm import LinearSVC
model = LinearSVC(C=100.0, random_state=0)
model.fit(X_train, y_train)

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler() 
x_train = scaler.fit_transform(X_train) 
x_test = scaler.transform(X_test)
#clf = MLPClassifier(hidden_layer_sizes = (200,100,50), activation = 'relu', solver='adam', alpha=1e-4, max_iter = 2000, random_state=1)
clf = MLPClassifier(hidden_layer_sizes = (500,), activation = 'logistic', solver='sgd', 
                    learning_rate = 'adaptive', alpha=1e-5, max_iter = 1000, random_state=1)
clf.fit(x_train, y_train)


pred = rf.predict(X_test)   #0.83
pred = xgb.predict(X_test)  #0.8501 
pred = clf.predict(x_test)  #0.7923
pred = cb.predict(X_test)   #0.8695
pred = model.predict(X_test)#0.7337

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, pred)
acc = (cm[0, 0] + cm[1, 1])/cm.sum()
print('accuracy : ' + str(acc)) 

from sklearn.metrics import accuracy_score
score = accuracy_score(y_test, pred)
print(str(score))

sift = pred

plt.scatter(y_test.flatten(), pred)
plt.xlabel('True Values')
plt.ylabel('Predictions')



# SURF features extraction
data = []
labels = []
c = 1
for folderName in ["0", "1"]:
    for imagePath in glob.glob(pathname+folderName+"\\*.png"):
        print(str(c))
        hist = ''
        image = cv2.imread(imagePath)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        surf = cv2.xfeatures2d.SURF_create()
        surf.setUpright(True)
        surf.setHessianThreshold(200)
        kp, des = surf.detectAndCompute(gray,None)
        if des is None:
            hist = get_histogram(np.zeros(128, dtype = 'float'))
        else:
            hist = get_histogram(des)
        means, std = cv2.meanStdDev(image)
        hist = np.concatenate((hist, means.transpose().flatten(), std.transpose().flatten()), axis = 0)
        labels.append(int(imagePath.split("\\")[-2]))
        data.append(hist)
        c += 1

cols = ['surf_feat{}'.format(i) for i in range(0, 256)]
cols.extend(channel)

X = pd.DataFrame(data = data, dtype='float32', columns = cols)
X.fillna( method ='ffill', inplace = True)
X = X.values
y = pd.DataFrame(data=labels, columns = ['class_label']).values

feature_data = pd.concat([X, y], axis = 1)
feature_data.to_csv('surf.csv')

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators = 500, criterion = 'entropy', random_state = 42)
rf.fit(X_train, y_train)

from xgboost import XGBClassifier
xgb = XGBClassifier()
xgb.fit(X_train, y_train)

from sklearn.svm import LinearSVC
model = LinearSVC(C=100.0, random_state=0)
model.fit(X_train, y_train)

pred = rf.predict(X_test)# 0.8355
pred = xgb.predict(X_test)# all 0s ---> 0.845
pred = model.predict(X_test)

from catboost import CatBoostClassifier as cbc
cb = cbc(depth = 10, task_type = 'GPU', learning_rate = 0.05)
cb.fit(X_train, y_train.ravel())

cb_features = cb.feature_importances_
x_train = pd.DataFrame(data = X_train, columns = cols)
cols = x_train.columns.values
# Create a dataframe with features
feature_dataframe = pd.DataFrame( {'features': cols,
     'CB feature importances': cb_features
    })
plot_imp(feature_dataframe)

pred = cb.predict(X_test) # 0.8615


from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler() 
x_train = scaler.fit_transform(X_train) 
x_test = scaler.transform(X_test)
clf = MLPClassifier(hidden_layer_sizes = (100,), activation = 'logistic', solver='adam', alpha=1e-4,
                    max_iter = 1000, random_state=1)
clf.fit(x_train, y_train)


pred = clf.predict(x_test)


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, pred)
acc = (cm[0, 0] + cm[1, 1])/cm.sum()
print('accuracy : ' + str(acc))

from sklearn.metrics import accuracy_score
score = accuracy_score(y_test, pred)
print(str(score))
surf = pred

plt.scatter(y_test, pred)
plt.xlabel('True Values')
plt.ylabel('Predictions')





# ORB features extraction
data = []
labels = []
c = 1
for folderName in ["0", "1"]:
    for imagePath in glob.glob(pathname+folderName+"\\*.png"):
        print(str(c))
        image = cv2.imread(imagePath)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h,l, = gray.shape
        if h != 50 or l != 50:
            gray = cv2.resize(gray,(50,50))
        orb = cv2.ORB_create(nfeatures = 100, edgeThreshold=10)
        kp = orb.detect(gray, None)
        kp, des = orb.compute(gray, kp)
        if des is None:
            hist = get_histogram(np.zeros(128, dtype = 'float'))
        else:
            hist = get_histogram(des)
        means, std = cv2.meanStdDev(image)
        #hist = np.concatenate((hist, means.transpose().flatten(), std.transpose().flatten()),axis = 0)
        labels.append(int(imagePath.split("\\")[-2]))
        data.append(hist)
        c += 1

cols = ['orb_feat{}'.format(i) for i in range(0, 256)]
cols.extend(channel)

X = pd.DataFrame(data = data, dtype='float32', columns = cols)
X.fillna( method ='ffill', inplace = True)
X = X.values
y = pd.DataFrame(data=labels, columns = ['class_label']).values

feature_data = pd.concat([X, y], axis = 1)
feature_data.to_csv('orb.csv')

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

from xgboost import XGBClassifier
xgb = XGBClassifier()
xgb.fit(X_train, y_train)

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators = 500, criterion = 'entropy', random_state = 42)
rf.fit(X_train, y_train)

from catboost import CatBoostClassifier as cbc
cb = cbc(depth = 10, task_type = 'GPU')
cb.fit(X_train, y_train.ravel())

cb_features = cb.feature_importances_
x_train = pd.DataFrame(data = X_train, columns = cols)
cols = x_train.columns.values
# Create a dataframe with features
feature_dataframe = pd.DataFrame( {'features': cols,
     'CB feature importances': cb_features
    })
plot_imp(feature_dataframe)


pred = cb.predict(X_test)#0.8599

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler() 
x_train = scaler.fit_transform(X_train) 
x_test = scaler.transform(X_test)
clf = MLPClassifier(hidden_layer_sizes = (100,), activation = 'logistic', solver='adam', alpha=1e-4,
                    max_iter = 1000, random_state=1)
clf.fit(x_train, y_train)


pred = clf.predict(x_test)


pred = rf.predict(X_test)# 0.8316
pred = xgb.predict(X_test)#all 0s ---> 0.8502

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, pred)
acc = (cm[0, 0] + cm[1, 1])/cm.sum()
print('accuracy : ' + str(acc)) 

from sklearn.metrics import accuracy_score
score = accuracy_score(y_test, pred)
print(str(score))
orb = pred


plt.scatter(surf, orb)
plt.xlabel('surf')
plt.ylabel('orb')



# 14 HARALICK features extraction
data = []
labels = []
c = 1
for folderName in ["0", "1"]:
    for imagePath in glob.glob(pathname+folderName+"\\*.png"):
        print(str(c))
        image = cv2.imread(imagePath)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h,l, = gray.shape
        if h != 50 or l != 50:
            gray = cv2.resize(gray,(50,50))
        hist = mt.features.haralick(gray, compute_14th_feature=True, return_mean=True)
        #HOG = hog(gray, feature_vector = True)
        means, std = cv2.meanStdDev(image)
        #hist = np.concatenate((hist, means.transpose().flatten(), std.transpose().flatten()), axis = 0)
        labels.append(int(imagePath.split("\\")[-2]))
        data.append(hist)
        c += 1

cols = ['har_feat{}'.format(i) for i in range(0, 14)]
#cols1 = ['lbp_feat{}'.format(i) for i in range(0, 256)]
cols.extend(channel)
        
X = pd.DataFrame(data = data, dtype='float32', columns = cols)
X.fillna( method ='ffill', inplace = True)

colormap = plt.cm.RdBu
plt.figure(figsize=(14,12))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(X.astype(float).corr(),linewidths=0.1,vmax=1.0, 
            square=True, cmap=colormap, linecolor='white', annot=True)

X = X.values
y = pd.DataFrame(data=labels, columns = ['class_label']).values

feature_data = pd.concat([X, y], axis = 1)
feature_data.to_csv('har.csv')



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

from xgboost import XGBClassifier
xgb = XGBClassifier()
xgb.fit(X_train, y_train)

xgb_features = xgb.feature_importances_
feature_dataframe = pd.DataFrame( {'features': cols,
     'XGB feature importances': xgb_features
    })
plot_imp(feature_dataframe)

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators = 500, criterion = 'entropy', random_state = 42)
rf.fit(X_train, y_train)

from sklearn.ensemble import AdaBoostClassifier
ada = AdaBoostClassifier(n_estimators = 500, learning_rate = 0.8, random_state = 42)
ada.fit(X_train, y_train)


from sklearn.svm import SVC
ksvm = SVC(C = 100, kernel = 'sigmoid', random_state = 68)
ksvm.fit(X_train, y_train)

from sklearn.svm import LinearSVC
model = LinearSVC(C=100.0, random_state=0)
model.fit(X_train, y_train)

from catboost import CatBoostClassifier as cbc
cb = cbc(depth = 10, task_type = 'GPU', learning_rate = 0.05)
cb.fit(X_train, y_train.ravel())



rf_features = rf.feature_importances_
feature_dataframe = pd.DataFrame( {'features': cols,
     'RF feature importances': rf_features
    })
plot_imp(feature_dataframe)



pred = cb.predict(X_test)#0.8666 har,hog

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler() 
x_train = scaler.fit_transform(X_train) 
x_test = scaler.transform(X_test)
clf = MLPClassifier(hidden_layer_sizes = (100,50,20, 10), activation = 'logistic', solver='adam', alpha=1e-4,
                    max_iter = 1000, random_state=1) #logistic, relu,
clf.fit(x_train, y_train)


pred = model.predict(X_test)#0.7611
pred = ksvm.predict(X_test)
pred = rf.predict(X_test)# 0.8689
pred = xgb.predict(X_test)# 0.8529 \ har, lbp 0.8484
pred = clf.predict(x_test)#0.8755 | 0.8827 | har,lbp 0.8672
pred = ada.predict(X_test)#0.85

plt.scatter(y_test, pred)
plt.xlabel('True Values')
plt.ylabel('Predictions')

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, pred)
acc = (cm[0, 0] + cm[1, 1])/cm.sum()
print('accuracy : ' + str(acc)) 

from sklearn.metrics import accuracy_score
score = accuracy_score(y_test, pred)
print(str(score))

'''

'''
# gabor wavelet feature extraction
kernels = []
for theta in range(4):
    theta = theta / 4. * np.pi
    for sigma in (1, 3):
        for frequency in (0.05, 0.25):
            kernel = np.real(gabor_kernel(frequency, theta=theta,
                                          sigma_x=sigma, sigma_y=sigma))
            kernels.append(kernel)

def compute_feats(image, kernels):
    feats = np.zeros((len(kernels), 2), dtype=np.double)
    for k, kernel in enumerate(kernels):
        filtered = ndi.convolve(image, kernel, mode='wrap')
        feats[k, 0] = filtered.mean()
        feats[k, 1] = filtered.var()
    return np.reshape(feats.flatten(), (1, 32))

# Combining all features

data = []
labels = []
c = 1
for folderName in ["0", "1"]:
    for imagePath in glob.glob(pathname+folderName+"\\*.png"):
        print(str(c))
        image = cv2.imread(imagePath)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h,l, = gray.shape
        if h != 50 or l != 50:
            gray = cv2.resize(gray,(50,50))
        
        #hist = np.array([], dtype = 'float')
        hist = mt.features.haralick(gray, compute_14th_feature=True, return_mean=True)
        '''
        #hist = hist.flatten()
        glcm = greycomatrix(gray, [2], [ np.pi/4, 3*np.pi/4, 5*np.pi/4, 7*np.pi/4], 256, symmetric=False, normed=True) #0
        contrast = greycoprops(glcm, 'contrast')
        dissimilarity = greycoprops(glcm, 'dissimilarity')
        homogeneity = greycoprops(glcm, 'homogeneity')
        ASM = greycoprops(glcm, 'ASM')
        energy = greycoprops(glcm, 'energy')
        correlation = greycoprops(glcm, 'correlation')
        #hist = np.append(hist, np.array([contrast, dissimilarity, homogeneity, ASM, energy, correlation]))
        hist = np.append(hist, [contrast, dissimilarity, homogeneity, ASM, energy, correlation])
        '''
        hu_moment = cv2.HuMoments(cv2.moments(gray)).flatten()
        hist = np.append(hist, hu_moment)
        '''
        #ref_feats = compute_feats(gray, kernels)
        #hist = np.append(hist, ref_feats)
        #lbp = local_binary_pattern(gray, 24, 3, method='uniform')
        #HOG = get_histogram(lbp)
        #hist = np.append(hist, HOG, axis = 0)
        
        hist = np.array([], dtype = 'float')
        
        orb = cv2.ORB_create(nfeatures = 100, edgeThreshold=12)
        kp = orb.detect(gray, None)
        kp, des = orb.compute(gray, kp)
        if des is None:
            hist = np.concatenate((hist,  get_histogram(np.zeros(128, dtype = 'float'))),axis = 0)
        else:
            hist = np.concatenate((hist, get_histogram(des)), axis = 0)
        
        surf = cv2.xfeatures2d.SURF_create()
        surf.setUpright(True)
        surf.setHessianThreshold(200)
        kp, des = surf.detectAndCompute(gray,None)
        if des is None:
            hist = np.concatenate((hist,  get_histogram(np.zeros(128, dtype = 'float'))), 
                             axis = 0)
        else:
            hist = np.concatenate((hist,get_histogram(des)), 
                             axis = 0)
        
        sift = cv2.xfeatures2d.SIFT_create()
        kp, des = sift.detectAndCompute(gray,None)
        #img=cv2.drawKeypoints(gray,kp, None)
        if des is None:
            hist = np.concatenate((hist,  get_histogram(np.zeros(128, dtype = 'float'))), 
                             axis = 0)
        else:
            hist = np.concatenate((hist, get_histogram(des)), axis = 0)
        ''''''
        means, std = cv2.meanStdDev(image)
        hist = np.concatenate((hist, means.transpose().flatten(), std.transpose().flatten()), axis = 0)
        '''
        labels.append(int(imagePath.split("\\")[-2]))
        data.append(hist)
        c += 1

cols = ['har_feat{}'.format(i) for i in range(0, 14)]
cols.extend(['glcm_feat{}'.format(i) for i in range(0, 24)])
cols.extend(['hu_feat{}'.format(i) for i in range(0, 7)])
#cols.extend(['gabor_feat{}'.format(i) for i in range(0, 32)])
cols.extend(channel)
        
X = pd.DataFrame(data = data, dtype='float32')
X.fillna( method ='ffill', inplace = True)

colormap = plt.cm.RdBu
plt.figure(figsize=(14,12))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(X.astype(float).corr(),linewidths=0.1,vmax=1.0, 
            square=True, cmap=colormap, linecolor='white', annot=True)


X = X.values
y = pd.DataFrame(data=labels, columns = ['class_labels']).values

feature_data = pd.concat([X, y], axis = 1)
feature_data.to_csv('har_surf.csv')

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

from xgboost import XGBClassifier
#xgb = XGBClassifier(n_estimators= 2000, max_depth= 4, objective= 'binary:logistic', nthread= -1)
xgb = XGBClassifier()
xgb.fit(X_train, y_train)

xgb_features = xgb.feature_importances_
feature_dataframe = pd.DataFrame( {'features': cols,
     'XGB feature importances': xgb_features
    })
plot_imp(feature_dataframe)

from sklearn.ensemble import AdaBoostClassifier
ada = AdaBoostClassifier(n_estimators = 500, random_state = 42)
ada.fit(X_train, y_train)

scores = cross_validate(xgb, X, y, cv=10, n_jobs = -1)

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators = 500, criterion = 'entropy', random_state = 42)
rf.fit(X_train, y_train)

rf_features = rf.feature_importances_
feature_dataframe = pd.DataFrame( {'features': cols,
     'RF feature importances': rf_features
    })
plot_imp(feature_dataframe)


from sklearn.ensemble import ExtraTreesClassifier
et = ExtraTreesClassifier(n_estimators=600, random_state=0)
et.fit(X_train, y_train)

from sklearn.svm import SVC
ksvm = SVC(C = 100, kernel = 'rbf', random_state = 68, gamma = 'scale')
ksvm.fit(X_train, y_train)

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler() 
x_train = scaler.fit_transform(X_train) 
x_test = scaler.transform(X_test)
clf = MLPClassifier(hidden_layer_sizes = (100,50,20,10), activation = 'logistic', solver='adam', alpha=1e-4,
                    max_iter = 1000, random_state=1) #logistic, relu,
clf.fit(x_train, y_train)

pred = xgb.predict(X_test)# 0.8528 har, orb| 0.8520 har, glcm | 0.8481 hu+| 0.8113 hu(glcm)| 0.81351 hu
pred = rf.predict(X_test)#0.8697 | hu+ 0.8635\ hu 0.8281
pred = clf.predict(x_test)#0.8832 har, glcm| hu+ 0.8801| hu 0.8364
pred = ada.predict(X_test)#0.8526 |0.8496 hu
pred = et.predict(X_test)# har, glcm, surf, orb  0.8256| 0.8586 hu

prob = rf.feature_importances_
imp_index = []
for i in range(0, len(prob)):
    if(prob[i] > 0.001):
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

xgb.fit(X_train_, y_train)
pred = xgb.predict(X_test)

rf.fit(X_train_, y_train)
pred = rf.predict(X_test)

from catboost import CatBoostClassifier as cbc
cb = cbc(depth = 10, task_type = 'GPU', learning_rate = 0.05)
cb.fit(X_train, y_train.ravel())

pred = cb.predict(X_test)# 0.865 har,orb | 0.87 har,sift | 0.8608 orb, sift, surf | 0.8733 har,glcm| 0.8689 hu+|0.8275 hu(glcm)|0.8282 hu

from sklearn.metrics import accuracy_score
score = accuracy_score(y_test, pred)
print(str(score))

###Holding all train and test predictions

#xgb_test = xgb.predict(X_test)
cb_test = cb.predict(X_test)
rf_test = rf.predict(X_test)
#et_test = et.predict(X_test)
mlp_test = clf.predict(X_test)
ada_test = ada.predict(X_test)

from sklearn.model_selection import StratifiedKFold

def Stacking(model,train,y,test,n_fold):
   folds=StratifiedKFold(n_splits=n_fold,random_state=1)
   test_pred=np.empty((test.shape[0],1),float)
   train_pred=np.empty((0,1),float)
   for train_indices,val_indices in folds.split(train,y.values):
      x_train,x_val=train.iloc[train_indices],train.iloc[val_indices]
      y_train,y_val=y.iloc[train_indices],y.iloc[val_indices]

      model.fit(X=x_train,y=y_train)
      train_pred=np.append(train_pred,model.predict(x_val))
      #test_pred=np.append(test_pred,model.predict(test))
   return train_pred

xtrn = pd.DataFrame(data = X_train, dtype='float32')
ytrn = pd.DataFrame(data = y_train, dtype='float32', columns = ['class_labels'])
xtst = pd.DataFrame(data = X_test, dtype='float32')
ytst = pd.DataFrame(data = y_test, dtype='float32', columns = ['class_labels'])

#Generating new traning data
#xgb_train = Stacking(xgb, xtrn, ytrn, xtst, 5)
cb_train = Stacking(cb, xtrn, ytrn, xtst, 5)
rf_train = Stacking(rf, xtrn, ytrn, xtst, 5)
#et_train = Stacking(et, xtrn, ytrn, xtst, 5)
mlp_train = Stacking(clf, xtrn, ytrn, xtst, 5)
ada_train = Stacking(ada, xtrn, ytrn, xtst, 5)

#v2
xgb_train = xgb.fit(X_train, y_train).predict(X_train)
cb_train = cb.fit(X_train, y_train.ravel()).predict(X_train)
rf_train = rf.fit(X_train, y_train).predict(X_train)
et_train = et.fit(X_train, y_train).predict(X_train)
mlp_train = clf.fit(X_train, y_train).predict(X_train)
ada_train = ada.fit(X_train, y_train).predict(X_train)

#Ensembling Models

##Stacking

base_predictions_train = pd.DataFrame( {'RandomForest': rf_train.ravel(),
     'ExtraTrees': et_train.ravel(),
     'AdaBoost': ada_train.ravel(),
      'XGradientBoost': xgb_train.ravel(),
      'Catboost': cb_train.ravel(),
      'Multi_LayerPerceptron': mlp_train.ravel()
    })
base_predictions_train.head()
base_predictions_train.to_csv('nfeat_base_predictions_trainV2.csv')

#compare training models
data = [
    go.Heatmap(
        z= base_predictions_train.astype(float).corr().values ,
        x=base_predictions_train.columns.values,
        y= base_predictions_train.columns.values,
          colorscale='Viridis',
            showscale=True,
            reversescale = True
    )
]
py.plot(data, filename='labelled-heatmap', image = 'png', image_filename='model_heatmap_all')

x_train = np.concatenate((rf_train.reshape((222019, 1)), ada_train.reshape((222019, 1)),
                          mlp_train.reshape((222019, 1))), axis=1)
x_test = np.concatenate((rf_test.reshape((55505, 1)), ada_test.reshape((55505, 1)),
                         mlp_test.reshape((55505, 1))), axis=1)


x_train = np.concatenate((rf_train.reshape((222019, 1)), xgb_train.reshape((222019, 1))), axis=1)
x_test = np.concatenate((rf_test.reshape((55505, 1)), xgb_test.reshape((55505, 1))), axis=1)

#Fitting final model
gbm = XGBClassifier(
    #learning_rate = 0.02,
 n_estimators= 2000,
 max_depth= 4,
 min_child_weight= 2,
 #gamma=1,
 gamma=0.9,                        
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread= -1,
 scale_pos_weight=1).fit(x_train, y_train)
gbm = XGBClassifier().fit(x_train, y_train)
predictions = gbm.predict(x_test)#0.87466

cb = cbc(depth = 10, task_type = 'GPU', learning_rate = 0.05).fit(x_train, y_train.ravel())
predictions = cb.predict(x_test)#0.87466

clf1 = MLPClassifier(hidden_layer_sizes = (100,50,20,10), activation = 'logistic', solver='adam', alpha=1e-4,
                    max_iter = 1000, random_state=1).fit(x_train, y_train)
predictions = clf1.predict(x_test)#0.87466

from sklearn.linear_model import LogisticRegression
reg = LogisticRegression().fit(x_train, y_train)
predictions = reg.predict(x_test)

score = accuracy_score(y_test, predictions)
print(str(score))

'''
cb, rf, ada, mlp: 0.87466 | 0.8915 -> [xgb, cb, mlp]

'''

from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

train_sizes, train_scores, test_scores = learning_curve(ada, 
                                                        X_train, 
                                                        y_train.ravel(),
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
plt.savefig('ada_plot.jpg')
   

