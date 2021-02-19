#!/usr/bin/env python
# coding: utf-8

# # Στοιχεία ομάδας
# 
# ###  Ομάδα 63 
# 
# ###  Ιωάννης Παλιάκης 03114877
# 

# ##  *Dataset S10*
# 
# 

# # Βασικές Πληροφορίες
# 

# 
# ### 1. Σύντομη παρουσίαση του dataset (τι περιγράφει).
# 
# Το συγκεριμένο dataset περιέχει δείγματα από τρισδιάστατα αντικείμενα μέσα σε μια εικόνα και ο σκοπός είναι να ταξινομηθεί ενα δεδομένο αντικείμενο ως ένας από τέσσερις τύπους οχημάτων (double decker bus, Cheverolet van, Saab 9000 και Opel Manta 400), χρησιμοποιώντας ένα σύνολο χαρακτηριστικών που εξάγονται από το εκάστοτε αντικείμενο. 
# 

# ### 2. Αριθμός δειγμάτων και χαρακτηριστικών, είδος χαρακτηριστικών. Υπάρχουν μη διατεταγμένα χαρακτηριστικά και ποια είναι αυτά;
# 
# Όπως προκύπτει από τον παρακάτω κώδικα, διαθέτουμε 847 δείγματα και 18 χαρακτηριστικά. Τα χαρακτηριστικά είναι όλα ίδιου είδους συγκεκριμένα αριθμητικού τύπου, αρα και διατεταγμένα. 
# 

# #### *Λήψη αρχείων, συννένωση και ανάγνωση*

# In[42]:


import requests
import os

# The path containing your notebook
path_data = './'
# The name of the file
filenames = ['xaa.dat','xab.dat','xac.dat','xad.dat','xae.dat','xaf.dat','xag.dat','xah.dat','xai.dat']

for name in filenames:
    if os.path.exists(os.path.join(path_data, name)):
        print('The file %s already exists.' % os.path.join(path_data, name))
    else:
        url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/statlog/vehicle/' + name
        r = requests.get(url)
        with open(os.path.join(path_data, name), 'wb') as f:
            f.write(r.content)
        print('Downloaded file %s.' % os.path.join(path_data, name))


# In[43]:


#concatination of .dat files

with open('xaa.dat') as fp: 
    data = fp.read() 

with open('xab.dat') as fp: 
    data2 = fp.read()

with open('xac.dat') as fp: 
    data3 = fp.read()
    
with open('xad.dat') as fp: 
    data4 = fp.read()
    
with open('xae.dat') as fp: 
    data5 = fp.read()
    
with open('xaf.dat') as fp: 
    data6 = fp.read()
    
with open('xag.dat') as fp: 
    data7 = fp.read()
    
with open('xah.dat') as fp: 
    data8 = fp.read()

with open('xai.dat') as fp: 
    data9 = fp.read()

data += data2+data3+data4+data5+data6+data7+data8+data9
  
with open ('final_file.dat', 'w') as fp: 
    fp.write(data) 


# In[85]:



import csv

with open('final_file.dat') as dat_file, open('final_file.csv', 'w') as csv_file:
    csv_writer = csv.writer(csv_file)
    
    line = "COMPACTNESS,CIRCULARITY,DISTANCE CIRCULARITY,RADIUS RATIO,PR.AXIS ASPECT RATIO,MAX.LENGTH ASPECT RATIO,SCATTER RATIO,ELONGATEDNESS,PR.AXIS RECTANGULARITY,MAX.LENGTH RECTANGULARITY,SCALED VARIANCE MAJOR,SCALED VARIANCE MINOR,SCALED RADIUS OF GYRATION,SKEWNESS MAJOR,SKEWNESS MINOR,KURTOSIS MAJOR,KURTOSIS MINOR,HOLLOWS RATIO,CLASS"
#     line=[i for i in range(19)]
    row = [field for field in line.split(',')]
    csv_writer.writerow(row)
    
    for line in dat_file:
        stripped=line.strip()
        row=stripped.split(' ')
        csv_writer.writerow(row)


# In[ ]:





# In[45]:


import pandas as pd
data = pd.read_csv('final_file.csv', delimiter=',',header=None)
data.head()


# Τσεκάτω ποιά μπορούν να γίνουν ints . Βλέπω ότι τα μόνο που δεν μπορούνε είναι στην στήλη 18, δηλαδή στην στήλη κλάση . Συνεπάγω ότι όλα μου τα features είναι ints. Στην συνέχεια επειδή τα έχω διαβάσει σαν strings τα μετατρέπω σε ints.

# In[46]:


samples = data.loc[1:,:18].values #values are in string type
r = len(samples)
c = len(samples[1])


for i in range(r):     # I check if features can be converted to int . If a feature cannot be converted into 
    for j in range(c):   # an int it means it is a string. i see that in the column 18 they are all strings
        try:
            int(samples[i][j]) 
        except:
            print ("Not an int with i {} and j {}".format(i,j))

samples = data.loc[1:,:17].values
            
for i in range(r):      #convert values to int type
    for j in range(0,c-1):
        samples[i,j]=int(samples[i,j])
        
print(samples)


# ### 3. Υπάρχουν επικεφαλίδες; Αρίθμηση γραμμών;
# 
# Το αρχικό dataset δεν περιείχε αρίθμηση γραμμών όυτε στηλών. Ωστόσο το pandas εισάγει αυτόματη αρίθμηση.
# 
# 

# ### 4. Ποιες είναι οι ετικέτες των κλάσεων και σε ποιά στήλη βρίσκονται; 
# 

# In[47]:


data.loc[1:, 18].unique()


# Η 18η στήλη όπως παρατηρούμε αναφέρεται στην αληθινή κλάση του δείγματος που είναι μία από αυτές που προαναφέραμε, χρησιμοποιώντας τις ετικέτες: 'van' ή 'saab' ή 'bus' ή 'opel'.

# ### 5. Χρειάστηκε να κάνετε μετατροπές στα αρχεία text και ποιες? 
# 
# Δεν χρειάστηκε να γίνουν μετατροπές στα αρχεία .dat καθώς όλα τα στοιχεία μας είναι αριθμητικά.
# 

# ### 6. Υπάρχουν απουσιάζουσες τιμές; Πόσα είναι τα δείγματα με απουσιάζουσες τιμές και ποιο το ποσοστό τους επί του συνόλου; 

# In[48]:


data.iloc[:,:18].isnull().any().sum()


# Όπως φαίνεται δεν απουσιάζει καμία τιμή.

# ### 7. Ποιος είναι ο αριθμός των κλάσεων και τα ποσοστά δειγμάτων τους επί του συνόλου;  Αν θεωρήσουμε ότι ένα dataset είναι μη ισορροπημένο αν μια οποιαδήποτε κλάση είναι 1.5 φορά πιο συχνή από κάποια άλλη (60%-40% σε binary datasets) εκτιμήστε την ισορροπία του dataset.

# In[49]:


data.iloc[1:,18].value_counts(normalize=True)


# Παρατηρούμε ότι όπως προαναφέραμε διαθέτουμε 4 κλάσεις οι οποίες εμφανίζονται με την ίδια συχνότητα στο dataset μας και κατ' επέκταση, όπως φαίνεται και από τα παραπάνω ποσοστά, πρόκειται για ένα πλήρως ισορροπημένο dataset.

# ### 8. Διαχωρισμός σε train και test set.
# 

# In[81]:


from sklearn.model_selection import train_test_split
data=data.loc[1:,:] #exclude the 1st row with the names of the features


labels = data[18].values
X_train, X_test, y_train, y_test = train_test_split(samples, labels, test_size=0.2, random_state=42)
print (samples)




# # Baseline Classification

# ### 1. Εκπαίδευση και αξιολόγηση των ταξινομητών με default τιμές.

# In[51]:


from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


# ### Εκπαιδεύση classifiers confusion matrix, f1-micro average και f1-macro average
# 
# #### Dummy Classifiers

# In[52]:


dc_uniform = DummyClassifier(strategy="uniform")
dc_constant_bus = DummyClassifier(strategy="constant", constant='bus')
dc_constant_saab = DummyClassifier(strategy="constant", constant='saab')
dc_constant_opel = DummyClassifier(strategy="constant", constant='opel')
dc_constant_van = DummyClassifier(strategy="constant", constant='van')
dc_most_frequent = DummyClassifier(strategy="most_frequent")
dc_stratified = DummyClassifier(strategy="stratified")


# In[53]:


dc_uniform.fit(X_train, y_train)

preds_uniform = dc_uniform.predict(X_test)

conf_uniform = confusion_matrix(y_test, preds_uniform)
print("Infos about DummyClassifier Uniform\n",conf_uniform)

precision, recall, f1_uniform_micro, _ = precision_recall_fscore_support(y_test, preds_uniform, average='micro')
print("Precision: {}\nRecall: {}\nF1-micro: {}".format(precision, recall, f1_uniform_micro))

precision, recall, f1_uniform_macro, _ = precision_recall_fscore_support(y_test, preds_uniform, average='macro')
print("\nPrecision: {}\nRecall: {}\nF1-macro: {}".format(precision, recall, f1_uniform_macro))


# In[54]:


dc_constant_bus.fit(X_train, y_train)

preds_constant_bus = dc_constant_bus.predict(X_test)

conf_constant_bus = confusion_matrix(y_test, preds_constant_bus)
print("Infos about DummyClassifier constant_bus\n",conf_constant_bus)


precision, recall, f1_constant_bus_micro, _ = precision_recall_fscore_support(y_test, preds_constant_bus, average='micro')
print("Precision: {}\nRecall: {}\nF1-micro: {}".format(precision, recall, f1_constant_bus_micro))

precision, recall, f1_constant_bus_macro, _ = precision_recall_fscore_support(y_test, preds_constant_bus, average='macro')
print("\nPrecision: {}\nRecall: {}\nF1-macro: {}".format(precision, recall, f1_constant_bus_macro))


# In[55]:


dc_constant_saab.fit(X_train, y_train)

preds_constant_saab = dc_constant_saab.predict(X_test)

conf_constant_saab = confusion_matrix(y_test, preds_constant_saab)
print("Infos about DummyClassifier constant_saab\n",conf_constant_saab)


precision, recall, f1_constant_saab_micro, _ = precision_recall_fscore_support(y_test, preds_constant_saab, average='micro')
print("Precision: {}\nRecall: {}\nF1-micro: {}".format(precision, recall, f1_constant_saab_micro))

precision, recall, f1_constant_saab_macro, _ = precision_recall_fscore_support(y_test, preds_constant_saab, average='macro')
print("\nPrecision: {}\nRecall: {}\nF1-macro: {}".format(precision, recall, f1_constant_saab_macro))



# In[56]:


dc_constant_opel.fit(X_train, y_train)

preds_constant_opel = dc_constant_opel.predict(X_test)

conf_constant_opel = confusion_matrix(y_test, preds_constant_opel)
print("Infos about DummyClassifier constant_opel\n",conf_constant_opel)


precision, recall, f1_constant_opel_micro, _ = precision_recall_fscore_support(y_test, preds_constant_opel, average='micro')
print("Precision: {}\nRecall: {}\nF1-micro: {}".format(precision, recall, f1_constant_opel_micro))

precision, recall, f1_constant_opel_macro, _ = precision_recall_fscore_support(y_test, preds_constant_opel, average='macro')
print("\nPrecision: {}\nRecall: {}\nF1-macro: {}".format(precision, recall, f1_constant_opel_macro))




# In[57]:


dc_constant_van.fit(X_train, y_train)

preds_constant_van = dc_constant_van.predict(X_test)

conf_constant_van = confusion_matrix(y_test, preds_constant_van)
print("Infos about DummyClassifier constant_van\n",conf_constant_van)


precision, recall, f1_constant_van_micro, _ = precision_recall_fscore_support(y_test, preds_constant_van, average='micro')
print("Precision: {}\nRecall: {}\nF1-micro: {}".format(precision, recall, f1_constant_van_micro))

precision, recall, f1_constant_van_macro, _ = precision_recall_fscore_support(y_test, preds_constant_van, average='macro')
print("\nPrecision: {}\nRecall: {}\nF1-macro: {}".format(precision, recall, f1_constant_van_macro))





# In[58]:


dc_most_frequent.fit(X_train, y_train)

preds_most_frequent = dc_most_frequent.predict(X_test)

conf_most_frequent = confusion_matrix(y_test, preds_most_frequent)
print("Infos about DummyClassifier frequent\n",conf_most_frequent)

precision, recall, f1_most_frequent_micro, _ = precision_recall_fscore_support(y_test, preds_most_frequent, average='micro')
print("Precision: {}\nRecall: {}\nF1-micro: {}".format(precision, recall, f1_most_frequent_micro))

precision, recall, f1_most_frequent_macro, _ = precision_recall_fscore_support(y_test, preds_most_frequent, average='macro')
print("\nPrecision: {}\nRecall: {}\nF1-macro: {}".format(precision, recall, f1_most_frequent_macro))


# In[59]:


dc_stratified.fit(X_train, y_train)

preds_stratified = dc_stratified.predict(X_test)

conf_stratified = confusion_matrix(y_test, preds_stratified)
print("Infos about DummyClassifier stratified\n",conf_stratified)

precision, recall, f1_stratified_micro, _ = precision_recall_fscore_support(y_test, preds_stratified, average='micro')
print("Precision: {}\nRecall: {}\nF1-macro: {}".format(precision, recall, f1_stratified_micro))

precision, recall, f1_stratified_macro, _ = precision_recall_fscore_support(y_test, preds_stratified, average='macro')
print("\nPrecision: {}\nRecall: {}\nF1-macro: {}".format(precision, recall, f1_stratified_macro))


# #### Gaussian Naive Bayes

# In[60]:


from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)
preds_gnb = gnb.predict(X_test)

conf_gnb = confusion_matrix(y_test, preds_gnb)
print("Infos about gnb\n",conf_gnb)

precision, recall, f1_gnb_micro, _ = precision_recall_fscore_support(y_test, preds_gnb, average='micro')
print("Precision: {}\nRecall: {}\nF1-micro: {}".format(precision, recall, f1_gnb_micro))

precision, recall, f1_gnb_macro, _ = precision_recall_fscore_support(y_test, preds_gnb, average='macro')
print("\nPrecision: {}\nRecall: {}\nF1-macro: {}".format(precision, recall, f1_gnb_macro))


# #### kNN classifier

# In[61]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
preds_knn = knn.predict(X_test)

conf_knn = confusion_matrix(y_test, preds_knn)
print("Infos about knn\n",conf_knn)

precision, recall, f1_knn_micro, _ = precision_recall_fscore_support(y_test, preds_knn, average='micro')
print("Precision: {}\nRecall: {}\nF1-micro: {}".format(precision, recall, f1_knn_micro))

precision, recall, f1_knn_macro, _ = precision_recall_fscore_support(y_test, preds_knn, average='macro')
print("\nPrecision: {}\nRecall: {}\nF1-macro: {}".format(precision, recall, f1_knn_macro))


# ### 2. Bar plots σύγκρισης

# In[62]:


import matplotlib.pyplot as plt


# In[63]:


f1_micro_scores = [f1_uniform_micro, f1_constant_bus_micro, f1_constant_saab_micro, f1_constant_opel_micro,f1_constant_van_micro,
                   f1_most_frequent_micro, f1_stratified_micro, f1_gnb_micro, f1_knn_micro] 
x = range(9)
plt.figure(figsize=(15,9))
plt.locator_params(nbins=20, axis='y')
plt.bar(x, f1_micro_scores)
plt.xticks(x, ('uniform', 'constant bus', 'constant saab', 'constant opel', 'constant van', 'most frequent', 'stratified', 'gnb','knn'))
plt.title('F1 micro scores')
plt.show()


# In[64]:


f1_macro_scores = [f1_uniform_macro, f1_constant_bus_macro, f1_constant_saab_macro, f1_constant_opel_macro,f1_constant_van_macro,
                   f1_most_frequent_macro, f1_stratified_macro, f1_gnb_macro, f1_knn_macro]
x = range(9)
plt.figure(figsize=(15,9))
plt.locator_params(nbins=20, axis='y')
plt.bar(x, f1_macro_scores)
plt.xticks(x, ('uniform', 'constant bus', 'constant saab', 'constant opel', 'constant van', 'most frequent', 'stratified', 'gnb','knn'))
plt.title('F1 macro scores')
plt.show()


# In[82]:


np.unique(y_train, return_counts=True)


# ### 3. Σχολιασμός αποτελεσμάτων 
# 
Συμβαίνει το εξής παράδοξο. Ο τρόπος που χωρίσαμε τα data σε train και set μας άφησε περισσότερα opel στον train set παρόλο που τα opel δεν είναι πιο συχνά στην πραγματικότητα . Αυτό έχει ως αποτέλεσμα το most frequent να επιλέγει πάντα opel που όμως στο test set παρουσιάζονται ακόμη λιγότερο . Έτσι το most frequent έχει αρκετά μικρό f1_μmicro και f1_macro . Βλέπουμε ότι οι καλύτεροι αλγόριθμοι είναι προφανώς ο Gaussian bayse και ο Knn. Το f1_micro και f1_macro δεν διαφέρουν αισθητά διότι οι κλάσεις παρουσιάζονται σε αρκετά παρόμοιο ποσοστό. Οπότε μπορούμε να δούμε και τις δύο μετρικές σαν μία. 
# # Βελτιστοποίηση ταξινομητών
# 
# 

# Σαν πρώτο βήμα θα κωδικοποιήσουμε τις ετικέτες των κλάσεων:

# In[65]:


import numpy as np
import sklearn
from sklearn.pipeline import Pipeline


# In[66]:


f1s = []
train_variance = X_train.var(axis=0)
print(train_variance)
print(np.max(train_variance))
print(np.median(train_variance))


# In[67]:


from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import RandomOverSampler
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
selector = VarianceThreshold()
scaler = StandardScaler()
ros = RandomOverSampler()
pca = PCA()
n_components = [5, 10, 15, 18]
clf = sklearn.neighbors.KNeighborsClassifier(n_jobs=-1) 
vthreshold = [0,7,10,20,40,60,70,80] #προσαρμόζουμε τις τιμές μας στο variance που παρατηρήσαμε
k = [1,3, 5, 11, 21, 31, 41] # η υπερπαράμετρος του ταξινομητή

pipe = Pipeline(steps=[('selector', selector),('scaler', scaler), ('kNN', clf)])


# In[68]:


estimator = GridSearchCV(pipe,
                         dict(selector__threshold=vthreshold,
                              kNN__n_neighbors=k),
                         cv=5,
                         scoring='f1_macro',
                         n_jobs=-1)


# In[69]:


import time
start_time = time.time()
estimator.fit(X_train, y_train)
preds = estimator.predict(X_test)
print("Συνολικός χρόνος fit και predict: %s seconds" % (time.time() - start_time))
print(classification_report(y_test, preds))


# In[70]:


print(estimator.best_estimator_)
print(estimator.best_params_)


# In[72]:


from sklearn.metrics import confusion_matrix
cnf_matrix = confusion_matrix(y_test, preds)
print("Confusion Matrix\n")
print(cnf_matrix)
precision, recall, f1_micro, _ = precision_recall_fscore_support(y_test, preds, average='micro')
print("Precision: {}\nRecall: {}\nF1-micro: {}".format(precision, recall, f1_micro))

precision, recall, f1_macro_k, _ = precision_recall_fscore_support(y_test, preds, average='macro')
print("\nPrecision: {}\nRecall: {}\nF1-macro: {}".format(precision, recall, f1_macro_k))


# In[73]:


def FullPipe(clf,name='classifer',pcan=0):
    selector = VarianceThreshold()
    scaler = StandardScaler()
    ros = RandomOverSampler()
    pca = PCA()
    n_components = [5, 10, 15, 18]
    vthreshold = [0,7,10,20,40,60,70,80] #προσαρμόζουμε τις τιμές μας στο variance που παρατηρήσαμε

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


# In[74]:




dummies = [dc_uniform,dc_constant_bus,dc_constant_saab,dc_constant_opel,dc_constant_van,dc_most_frequent,
           dc_stratified]
dummynames = ["dc_uniform","dc_constant_bus","dc_constant_saab","dc_constant_opel","dc_constant_van",
              "dc_most_frequent",
           "dc_stratified"]

for i in range(len(dummies)):
    f1s.append(FullPipe(dummies[i],dummynames[i]))


# In[ ]:


from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
f1s.append(FullPipe(gnb,'gnb'))


# In[ ]:


# f1_macro_scores = [f1_uniform_macro, f1_constant_bus_macro, f1_constant_saab_macro, f1_constant_opel_macro,f1_constant_van_macro,
#                    f1_most_frequent_macro, f1_stratified_macro, f1_gnb_macro, f1_knn_macro]
f1s.append(f1_macro_k)
x = range(9)
plt.figure(figsize=(15,9))
plt.locator_params(nbins=20, axis='y')
plt.bar(x, f1s)
plt.xticks(x, ('uniform', 'constant bus', 'constant saab', 'constant opel', 'constant van', 'most frequent', 'stratified', 'gnb','knn'))
plt.title('F1 macro scores')
plt.show()


# In[ ]:


dummynames.append('gnb')
dummynames.append('knn')


# In[ ]:


x = range(2)
fig, ax = plt.subplots(nrows=3, ncols=3,figsize=(15,20))
for i,a in enumerate(ax):
    for j,x1 in enumerate(a):
        if j+i*3<=25:
            x1.bar(x,[f1_macro_scores[j+i*3],f1s[j+i*3]])
            x1.set_title(dummynames[j+i*3])

Σχολιασμός αποτελεστμάτων:
Παρατηρώ ότι οι dummy classifiers παραμένουν σταθεροί , κάτι που είναι απολύτως λογικό. Μόνο ο stratified αλλάζει κάτι το οποίο είναι τυχαίο διότι ο stratified τυχαία κάθε φορά κάνει predict δηλαδή όχι πάντα με τον ίδιο τρόπο. 
Βλέπουμε μία σαφή βελτίωση στον knn διότι βρήκαμε τις καλύτερες παραμέτρους (γείτονες = 5) αλλά και ο μετασχηματισμός βοήθησε. Στον Gaussian βλέπουμε μία μικρή βελτίωση που οφέιλεται στον μετασχηματισμό . Pca έκανα στην αρχή , αλλά δεν βοήθησε κάτι που είναι απολύτως λογικό γιατί έχω λίγα σχετικά features . 