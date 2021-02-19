#!/usr/bin/env python
# coding: utf-8

# Ιωάννης Παλιάκης 03114877
# ομάδα 63 
# Πρώτο εργαστήριο 
# Μεγάλο data set 
# 

# In[1]:


import numpy as np
import requests
import os
import matplotlib.pyplot as plt
import csv
import pandas as pd


# In[2]:



with open('1_2_3_4.data') as fp: 
    data = fp.read() 

with open('5.data') as fp: 
    data1 = fp.read()
    
names = []
f = open("isolet.names", "r")
for x in f:
    x= x.split(':')
    names.append(x[0])

data+=data1

with open ('big_final_file.dat', 'w') as fp: 
    fp.write(data) 


# In[3]:


names = names[1:]
names.append("class")


# In[4]:


with open('big_final_file.dat') as dat_file, open('big_final_file.csv', 'w') as csv_file:
    csv_writer = csv.writer(csv_file)
    row = names
    
    csv_writer.writerow(row)
    
    for line in dat_file:
        stripped=line.strip()
        row=stripped.split(',')
        csv_writer.writerow(row)


# In[5]:


data = pd.read_csv('big_final_file.csv', delimiter=',',header=None)
data.head()


# In[6]:


tm = 0
for i in range(617):
    tm+=data[i].isnull().sum()
print(tm)


# To dataset όπως παραπάνω βλέπω έχει 617 features +1 την κλάση . Συνολικά το data set έχει 7798 samples . Τα samples είναι ήχοι από γράμματα που προφέρουν ομιλητές. Υπάρχουν 150 ομιλητές και ο καθένας λέει όλα τα γράμματα από 2 φορές. Τα γράμματα είναι της αγγλικής αλφαβήτου όπως μπορώ να δω και από τις κλάσεις που είναι 26. Τα features είναι η συχνότητα κανονικοποιημένη στο -1 1. Δεν υπάρχουν μη διατεταγμένα χαρακτηριστικά. ΟΙ κολόνες των κλάσεων βρίσκονται στην κολόνα 617

# In[7]:


# print(data)
print(data.loc[0:,:616])


# In[8]:


samples = data.loc[1:,:617].values #values are in string type
r = len(samples)
c = len(samples[1])    
for i in range(r):  
    for j in range(c):
        samples[i,j]=float(samples[i,j])


# In[9]:


data.iloc[1:,617].value_counts(normalize=True)


# In[10]:


from sklearn.model_selection import train_test_split
data=data.loc[1:,:] #exclude the 1st row with the names of the features


labels = samples[:,617]
samples = samples[:,:617]
X_train, X_test, y_train, y_test = train_test_split(samples, labels, test_size=0.3, random_state=42)


# Δεν λείπουν δεδομένα μόνο λείπουν 3 samples. Όλα τα γράμματα αντιπροσωπεύονται το ίδιο και συγκεκριμένα 3.8% το καθένας (σχεδόν  ,λόγω των 3 samples που λείπουν κάποια γράμματα αντιπροσωπεύονται λιγότερα αλλά αυτή η διαφορά είναι ήσσονος σημασίας)

# In[11]:


from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


# In[12]:


dc_uniform = DummyClassifier(strategy="uniform")
dc_most_frequent = DummyClassifier(strategy="most_frequent")
dc_stratified = DummyClassifier(strategy="stratified")


# In[13]:


for ind,i in enumerate(y_test):
    y_test[ind]  = int(i)
for ind,i in enumerate(y_train):
    y_train[ind]  = int(i)

for ind,i in enumerate(y_test):
    y_test[ind]  = str(i)
for ind,i in enumerate(y_train):
    y_train[ind]  = str(i)


# In[14]:


def test_the_dummies(which,name):
    which.fit(X_train, y_train)

    preds_uniform = which.predict(X_test)

    conf_uniform = confusion_matrix(y_test, preds_uniform)
    print("Infos about DummyClassifier Uniform\n",conf_uniform)

    precision, recall, f1_uniform_micro, _ = precision_recall_fscore_support(y_test, preds_uniform, average='micro')
    print("Precision: {}\nRecall: {}\nF1-micro: {}".format(precision, recall, f1_uniform_micro))

    precision, recall, f1_uniform_macro, _ = precision_recall_fscore_support(y_test, preds_uniform, average='macro')
    print("\nPrecision: {}\nRecall: {}\nF1-macro: {}".format(precision, recall, f1_uniform_macro))
    
    return [name,precision, recall, f1_uniform_micro,f1_uniform_macro]


# In[15]:


classifiers = []


# In[16]:


classifiers.append(test_the_dummies(dc_uniform,"uniform"))


# In[17]:


for i in range(1,27):
    dc_constant = DummyClassifier(strategy="constant", constant=str(i))
    classifiers.append(test_the_dummies(dc_constant,"constant_"+str(i)))


# Τρέχω όλους τους σταθερούς κλάσεων dummies με την παραπάνω λούπα

# In[18]:


classifiers.append(test_the_dummies(dc_most_frequent,'most_frequent'))


# In[19]:


classifiers.append(test_the_dummies(dc_stratified,'stratified'))


# In[20]:


def do_it_all(clf,clf_name):
    if clf_name=='mlp':
        clf = MLPClassifier(random_state=1, max_iter=300).fit(X_train, y_train)
    else:
        clf.fit(X_train, y_train)
    clf.predict(X_test)
    preds_clf = clf.predict(X_test)

    conf = confusion_matrix(y_test, preds_clf)
    print("Infos about {}\n".format(clf_name))
    print(conf)

    precision, recall, f1_clf_micro, _ = precision_recall_fscore_support(y_test, preds_clf, average='micro')
    print("Precision: {}\nRecall: {}\nF1-micro: {}".format(precision, recall, f1_clf_micro))

    precision, recall, f1_clf_macro, _ = precision_recall_fscore_support(y_test, preds_clf, average='macro')
    print("\nPrecision: {}\nRecall: {}\nF1-macro: {}".format(precision, recall, f1_clf_macro))
    
    return [clf_name,precision, recall, f1_clf_micro,f1_clf_macro]


# In[21]:


from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
classifiers.append(do_it_all(gnb,"Gaussian Naive"))


# In[22]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
classifiers.append(do_it_all(knn,'Knn'))


# In[23]:


from sklearn import svm
svm = svm.SVC()
classifiers.append(do_it_all(svm,'Support Vector Machines'))


# In[24]:


from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(random_state=1, max_iter=300)
classifiers.append(do_it_all(mlp,'Multi Layer Perceptrons'))


# Σχολιασμός για τα recall το Precision και το confusion matrix:
# Για τους constant ταξινομητές ισχύει το προφανές ότι έχω πάρα πολύ χαμηλά και precision και recall , τώρα για τους "καλούς ταξινομητές" έχω αρκετά παρόμοιο recall και precision που σημαίνιε ότι δεν έχω bias προς μία συγκεκριμένη κλάση.
# 

# In[25]:


for i in classifiers:
    print(i[0])


# In[26]:


f1_macro_scores = []
f1_micro_scores = []
clfnames =[]

for i in classifiers:
    f1_macro_scores.append(i[-1])
    f1_micro_scores.append(i[-2])
    clfnames.append(i[0])


# In[27]:


x = range(len(classifiers))
plt.figure(figsize=(40,20))
plt.locator_params(nbins=20, axis='y')
plt.bar(x, f1_macro_scores)
plt.xticks(x,clfnames)
plt.title('F1 macro scores')
plt.show()


# In[28]:


x = range(len(classifiers))
plt.figure(figsize=(40,20))
plt.locator_params(nbins=20, axis='y')
plt.bar(x, f1_micro_scores)
plt.xticks(x,clfnames)
plt.title('F1 micro scores')
plt.show()


# Οι αλγόριθμοι (επειδή είναι πολλοί και δεν φαίνονται ) είναι με την σειρά η εξής: Uniform , costant(1-26) , Most frequent ,  Stratified, Gaussian Naive , Knn, Svm και Mlp 
# 
# Πρατηρώ ότι Svm και Mlp είναι σχεδόν οι ίδιοι σε απόδοση. Κάτι πολύ λογικό γιατί είναι αλγόριθμοι ειδικεύομενοι σε τέτοια προβλήματα . Για τους dummy ο σχολιασμός είναι τετριμένος .
# 

# In[29]:


from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import RandomOverSampler
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import time


# In[30]:


train_variance = X_train.var(axis=0)
print(train_variance)
print(np.max(train_variance))
print(np.median(train_variance))


# In[34]:


def FullPipe(clf,name='classifer',pcan=0,ncomp=[5,10,20,50]):
    selector = VarianceThreshold()
    scaler = StandardScaler()
    ros = RandomOverSampler()
    pca = PCA()
    n_components = ncomp
    vthreshold = [0,0.01,0.02,0.05,0.08,0.1,0.2,0.5] #προσαρμόζουμε τις τιμές μας στο variance που παρατηρήσαμε

    if pcan==1:
        pipe = Pipeline(steps=[('selector', selector),('scaler', scaler),('pca',pca),(name, clf)])
        estimator = GridSearchCV(pipe,
                             dict(selector__threshold=vthreshold,
                                  pca__n_components=n_components),
                             cv=5,
                             scoring='f1_macro',
                             n_jobs=-1)
    else:
        pipe = Pipeline(steps=[('selector', selector),('scaler', scaler), (name, clf)])
        estimator = GridSearchCV(pipe,
                         dict(selector__threshold=vthreshold),
                         cv=5,
                         scoring='f1_macro',
                         n_jobs=-1)
    start_time = time.time()
    estimator.fit(X_train, y_train)
    preds = estimator.predict(X_test)
    print("Συνολικός χρόνος fit και predict: %s seconds" % (time.time() - start_time))
    print(classification_report(y_test, preds))
    print(estimator.best_estimator_)
    print(estimator.best_params_)
    cnf_matrix = confusion_matrix(y_test, preds)
    print("Confusion Matrix")
    print(cnf_matrix)
    precision, recall, f1_micro, _ = precision_recall_fscore_support(y_test, preds, average='micro')
    print("Precision: {}\nRecall: {}\nF1-micro: {}".format(precision, recall, f1_micro))

    precision, recall, f1_macro, _ = precision_recall_fscore_support(y_test, preds, average='macro')
    
    print("\nPrecision: {}\nRecall: {}\nF1-macro: {}".format(precision, recall, f1_macro))
    return f1_macro


# In[39]:


knq = KNeighborsClassifier()
selector = VarianceThreshold()
scaler = StandardScaler()
ros = RandomOverSampler()
pca = PCA()
n_components = [20,40,50,100,200,500]
k = [1,3, 5, 11, 21, 31, 41]
vthreshold = [0,0.01,0.02,0.05,0.08,0.1,0.2,0.5] #προσαρμόζουμε τις τιμές μας στο variance που παρατηρήσαμε

pipe = Pipeline(steps=[('selector', selector),('scaler', scaler),('pca',pca),('kNN', knq)])
estimator = GridSearchCV(pipe,
                     dict(selector__threshold=vthreshold,
                          pca__n_components=n_components,
                          kNN__n_neighbors=k),
                     cv=5,
                     scoring='f1_macro',
                     n_jobs=-1)

start_time = time.time()
estimator.fit(X_train, y_train)
preds = estimator.predict(X_test)
print("Συνολικός χρόνος fit και predict: %s seconds" % (time.time() - start_time))
print(classification_report(y_test, preds))
print(estimator.best_estimator_)
print(estimator.best_params_)
cnf_matrix = confusion_matrix(y_test, preds)
print("Confusion Matrix")
print(cnf_matrix)
precision, recall, f1_micro, _ = precision_recall_fscore_support(y_test, preds, average='micro')
print("Precision: {}\nRecall: {}\nF1-micro: {}".format(precision, recall, f1_micro))

precision, recall, f1_macro, _ = precision_recall_fscore_support(y_test, preds, average='macro')

print("\nPrecision: {}\nRecall: {}\nF1-macro: {}".format(precision, recall, f1_macro))


# In[40]:


f1s = []
f1s.append([precision, recall, f1_macro])


# In[47]:


mlp = MLPClassifier()
selector = VarianceThreshold()
scaler = StandardScaler()
ros = RandomOverSampler()
pca = PCA()
n_components = [20,40,50,100,200,500]
activation = ['identity','logistic','tanh','relu']
solver = ['lbfgs','sgd','adam']
alpha = [0.0001, 0.05]
max_iter = [100,200,300]
learning_rate = ['constant','adaptive']

vthreshold = [0,0.01,0.02,0.05,0.08,0.1,0.2,0.5] #προσαρμόζουμε τις τιμές μας στο variance που παρατηρήσαμε

pipe = Pipeline(steps=[('selector', selector),('scaler', scaler),('pca',pca),('mlp', mlp)])
estimator = GridSearchCV(pipe,
                     dict(selector__threshold=vthreshold, pca__n_components=n_components, 
                                        mlp__hidden_layer_sizes=(7,), mlp__solver=solver, mlp__activation=activation,
                                        mlp__max_iter=max_iter, mlp__learning_rate=learning_rate, mlp__alpha = alpha),
                                        cv=5, scoring='f1_macro', n_jobs=-1)

start_time = time.time()
estimator.fit(X_train, y_train)
preds = estimator.predict(X_test)
print("Συνολικός χρόνος fit και predict: %s seconds" % (time.time() - start_time))
print(classification_report(y_test, preds))
print(estimator.best_estimator_)
print(estimator.best_params_)
cnf_matrix = confusion_matrix(y_test, preds)
print("Confusion Matrix")
print(cnf_matrix)
precision, recall, f1_micro, _ = precision_recall_fscore_support(y_test, preds, average='micro')
print("Precision: {}\nRecall: {}\nF1-micro: {}".format(precision, recall, f1_micro))

precision, recall, f1_macro, _ = precision_recall_fscore_support(y_test, preds, average='macro')

print("\nPrecision: {}\nRecall: {}\nF1-macro: {}".format(precision, recall, f1_macro))


# In[48]:


f1s.append([precision, recall, f1_macro])


# In[50]:


def dummyPipe(clf,name='classifer'):
    selector = VarianceThreshold()
    scaler = StandardScaler()
    ros = RandomOverSampler()
    pca = PCA()
    n_components = [30,50,100,200,500]
    vthreshold = [0,0.01,0.02,0.05,0.08,0.1,0.2,0.5] #προσαρμόζουμε τις τιμές μας στο variance που παρατηρήσαμε
    pipe = Pipeline(steps=[('selector', selector),('scaler', scaler),('pca',pca),(name, clf)])
    estimator = GridSearchCV(pipe,
                         dict(selector__threshold=vthreshold,
                              pca__n_components=n_components),
                         cv=5,
                         scoring='f1_macro',
                         n_jobs=-1)
    start_time = time.time()
    estimator.fit(X_train, y_train)
    preds = estimator.predict(X_test)
    print("Συνολικός χρόνος fit και predict: %s seconds" % (time.time() - start_time))
    print(classification_report(y_test, preds))
    print(estimator.best_estimator_)
    print(estimator.best_params_)
    cnf_matrix = confusion_matrix(y_test, preds)
    print("Confusion Matrix")
    print(cnf_matrix)
    precision, recall, f1_micro, _ = precision_recall_fscore_support(y_test, preds, average='micro')
    print("Precision: {}\nRecall: {}\nF1-micro: {}".format(precision, recall, f1_micro))

    precision, recall, f1_macro, _ = precision_recall_fscore_support(y_test, preds, average='macro')
    
    print("\nPrecision: {}\nRecall: {}\nF1-macro: {}".format(precision, recall, f1_macro))
    return [f1_micro,f1_macro]


# In[51]:


for i in range(1,27):
    dc_constant = DummyClassifier(strategy="constant", constant=str(i))
    f1s.append(dummyPipe(dc_constant,"constant_"+str(i)))


# In[53]:


dc_uniform = DummyClassifier(strategy="uniform")
dc_most_frequent = DummyClassifier(strategy="most_frequent")
dc_stratified = DummyClassifier(strategy="stratified")


# In[54]:


dum = [dc_uniform,dc_most_frequent,dc_stratified]
dumnam = ["dc_uniform","dc_most_frequent","dc_stratified"]


for ind,i in enumerate(dum):
    f1s.append(dummyPipe(i,dumnam[ind]))


# In[82]:


from sklearn import svm
svc = svm.SVC()
selector = VarianceThreshold()
scaler = StandardScaler()
ros = RandomOverSampler()
pca = PCA()
n_components = [20,40,50,100,200,500]
vthreshold = [0,0.05,0.1,0.2,0.5] #προσαρμόζουμε τις τιμές μας στο variance που παρατηρήσαμε
kernel = ['linear']
c=[1, 10, 100, 1000]

pipe = Pipeline(steps=[('selector', selector),('scaler', scaler),('pca',pca),('SVM', svc)])
estimator = GridSearchCV(pipe,
                            dict(
                              pca__n_components=n_components,
                                SVM__kernel=kernel,
                                SVM__C=c),
                                        cv=5, scoring='f1_macro', n_jobs=-1)

start_time = time.time()
estimator.fit(X_train, y_train)
preds = estimator.predict(X_test)
print("Συνολικός χρόνος fit και predict: %s seconds" % (time.time() - start_time))
print(classification_report(y_test, preds))
print(estimator.best_estimator_)
print(estimator.best_params_)
cnf_matrix = confusion_matrix(y_test, preds)
print("Confusion Matrix")
print(cnf_matrix)
precision, recall, f1_micro, _ = precision_recall_fscore_support(y_test, preds, average='micro')
print("Precision: {}\nRecall: {}\nF1-micro: {}".format(precision, recall, f1_micro))

precision, recall, f1_macro, _ = precision_recall_fscore_support(y_test, preds, average='macro')

print("\nPrecision: {}\nRecall: {}\nF1-macro: {}".format(precision, recall, f1_macro))


# In[83]:


f1s.append([precision, recall, f1_macro])


# In[89]:


from sklearn import svm
svc = svm.SVC()
selector = VarianceThreshold()
scaler = StandardScaler()
ros = RandomOverSampler()
pca = PCA()
n_components = [20,40,50,100,200,500]
vthreshold = [0,0.05,0.1,0.2,0.5] #προσαρμόζουμε τις τιμές μας στο variance που παρατηρήσαμε
kernel = ['rbf','poly']
c=[1, 10, 100, 1000]
gam=[1e-3, 1e-4]
deg = [3,5,7,8]
tol = [0.001,0.002]

pipe = Pipeline(steps=[('selector', selector),('scaler', scaler),('pca',pca),('SVM', svc)])
estimator = GridSearchCV(pipe,
                            dict(
                              pca__n_components=n_components,
                                SVM__kernel=kernel,
                                SVM__C=c,
                                SVM__gamma=gam,
                            SVM__degree = deg,
                            SVM__tol = tol),
                                        cv=5, scoring='f1_macro', n_jobs=-1)

start_time = time.time()
estimator.fit(X_train, y_train)
preds = estimator.predict(X_test)
print("Συνολικός χρόνος fit και predict: %s seconds" % (time.time() - start_time))
print(classification_report(y_test, preds))
print(estimator.best_estimator_)
print(estimator.best_params_)
cnf_matrix = confusion_matrix(y_test, preds)
print("Confusion Matrix")
print(cnf_matrix)
precision, recall, f1_micro, _ = precision_recall_fscore_support(y_test, preds, average='micro')
print("Precision: {}\nRecall: {}\nF1-micro: {}".format(precision, recall, f1_micro))

precision, recall, f1_macro, _ = precision_recall_fscore_support(y_test, preds, average='macro')

print("\nPrecision: {}\nRecall: {}\nF1-macro: {}".format(precision, recall, f1_macro))


# In[90]:


f1s.append([precision, recall, f1_macro])


# In[91]:


print(f1s)


# In[92]:


ff =[]
ff.append(f1s[0][2])
ff.append(f1s[1][2])
ff.append(f1s[-1][2])
ff.append(f1s[-2][2])


# In[98]:


for i in f1s[2:-2]:
    ff.append(i[1])


# In[99]:


print(len(ff))


# In[102]:


x = range(len(ff))
plt.figure(figsize=(40,20))
plt.locator_params(nbins=20, axis='y')
plt.bar(x, ff)
plt.xticks(x,["Knn","Mlp","Svmlinear","Svmrbf"])
plt.title('F1 macro scores')
plt.show()


# Παρατηρώ μία αύξηση της απόδοσης στους "καλούς" (όχι dummy ) ταξινομητές που ξεκινάνε από αριστερά με την σειρά KNN MLP Svm με linear kernel και Svm με rbf . Κάτι που είναι αναμενόμενο. Οι αλθόριθμοι και ειδικότερα ο Mlp κάνανε πάρα πολύ ώρα να τρέξουν γιατί έβαλα πάρα πολλά ορίσματα στον grid search.
