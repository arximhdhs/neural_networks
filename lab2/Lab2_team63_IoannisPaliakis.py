#!/usr/bin/env python
# coding: utf-8

# # Εργαστηριακή Άσκηση 2. Μη επιβλεπόμενη μάθηση. 
# ## Σύστημα συστάσεων βασισμένο στο περιεχόμενο
# ## Σημασιολογική απεικόνιση δεδομένων με χρήση SOM 
# Ημερομηνία εκφώνησης της άσκησης: 23 Νοεμβρίου 2020
# 
# Ιωάννης Παλιάκης 03114877
# Ομάδα 63
# 

# In[1]:


get_ipython().system('pip install --upgrade pip')
get_ipython().system('pip install --upgrade numpy')
get_ipython().system('pip install --upgrade pandas')
get_ipython().system('pip install --upgrade nltk')
get_ipython().system('pip install --upgrade scikit-learn')
get_ipython().system('pip install --upgrade joblib')


# ## Εισαγωγή του Dataset

# In[2]:


import pandas as pd

dataset_url = "https://drive.google.com/uc?export=download&id=1PdkVDENX12tQliCk_HtUnAUbfxXvnWuG"
df_data_1 = pd.read_csv(dataset_url, sep='\t',  header=None, quoting=3, error_bad_lines=False)


# In[75]:


import numpy as np

# βάλτε το seed που αντιστοιχεί στην ομάδα σας
team_seed_number = 63

movie_seeds_url = "https://drive.google.com/uc?export=download&id=1EA_pUIgK5Ub3kEzFbFl8wSRqAV6feHqD"
df_data_2 = pd.read_csv(movie_seeds_url, header=None, error_bad_lines=False)

# επιλέγεται 
my_index = df_data_2.iloc[team_seed_number,:].values

titles = df_data_1.iloc[:, [2]].values[my_index] # movie titles (string)
categories = df_data_1.iloc[:, [3]].values[my_index] # movie categories (string)
bins = df_data_1.iloc[:, [4]]
catbins = bins[4].str.split(',', expand=True).values.astype(np.float)[my_index] # movie categories in binary form (1 feature per category)
summaries =  df_data_1.iloc[:, [5]].values[my_index] # movie summaries (string)
corpus = summaries[:,0].tolist() # list form of summaries


# In[76]:


print(categories)


# In[5]:


ID = 99
print(titles[ID])
print(categories[ID])
print(catbins[ID])
print(corpus[ID])


# In[6]:


for i in range(5000):
    print(titles[i],my_index[i])


# # Εφαρμογή 1. Υλοποίηση συστήματος συστάσεων ταινιών βασισμένο στο περιεχόμενο
# <img src="http://clture.org/wp-content/uploads/2015/12/Netflix-Streaming-End-of-Year-Posts.jpg" width="70%">

# ## Μετατροπή σε TFIDF
# 
# Το πρώτο βήμα θα είναι λοιπόν να μετατρέψετε το corpus σε αναπαράσταση tf-idf:

# In[4]:


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_df=0.5, min_df=3,stop_words='english')
corpus_tf_idf=vectorizer.fit_transform(corpus)


# In[8]:


print(corpus_tf_idf.shape)


# In[9]:


print(vectorizer.get_feature_names())


# In[10]:


from sklearn.metrics.pairwise import cosine_similarity
def content_recommender(target_movie,max_recommendations):
    target_movie_id=0;
    for i in range(0,5000):
        if my_index[i]==target_movie:
            target_movie_id=i
            break;
    #print id_in_corpus
    cos_sim=[];
    for j in range(0, 5000):
        if my_index[j]!=target_movie_id:
            hd=cosine_similarity(corpus_tf_idf[target_movie_id],corpus_tf_idf[j])  
            cos_sim.append((hd[0][0],j))   
    sorted_by_first = sorted(cos_sim, key=lambda tup: tup[0])
    sorted_by_first.reverse()
    second_elts = [x[1] for x in sorted_by_first]
    myl1= second_elts[1:max_recommendations+1]
    print ("*** Target movie",target_movie," ***")
    print ("Title: ",titles[target_movie_id][0])
    print ("Summary: ",summaries[target_movie_id][0])
    print ("Genres: " ,categories[target_movie_id][0])
    print ("--- ",max_recommendations, "most related movies based on content ---")
    for i in range(0, max_recommendations):
        print ("--- Recommended movie No. ",i+1 , " ---")
        print ("Movie ID: ",my_index[myl1[i]])
        print ("Title: ",titles[myl1[i]][0])
        print ("Summary: ",summaries[myl1[i]][0])
        print ("Genres: " ,categories[myl1[i]][0])


# In[11]:


content_recommender(my_index[50],20)


# Μία γρήγορη παρατήρη του παραπάνω αλγορίθμου είναι ότι θεωρεί σχετικό περιεχόμενο Ονόματα. Κάτι τέτοιο χαλάει αρκετά την ποιότητα του recommendation

# ## Βελτιστοποίηση
# 
# Αφού υλοποιήσετε τη συνάρτηση `content_recommender` χρησιμοποιήστε τη για να βελτιστοποιήσετε την `TfidfVectorizer`. Συγκεκριμένα, αρχικά μπορείτε να δείτε τι επιστρέφει το σύστημα για τυχαίες ταινίες-στόχους και για ένα μικρό `max_recommendations` (2 ή 3). Αν σε κάποιες ταινίες το σύστημα μοιάζει να επιστρέφει σημασιολογικά κοντινές ταινίες σημειώστε το `ID` τους. Δοκιμάστε στη συνέχεια να βελτιστοποιήσετε την `TfidfVectorizer` για τα συγκεκριμένα `ID` ώστε να επιστρέφονται σημασιολογικά κοντινές ταινίες για μεγαλύτερο αριθμό `max_recommendations`. Παράλληλα, όσο βελτιστοποιείτε την `TfidfVectorizer`, θα πρέπει να λαμβάνετε καλές συστάσεις για μεγαλύτερο αριθμό τυχαίων ταινιών. Μπορείτε επίσης να βελτιστοποιήσετε τη συνάρτηση παρατηρώντας πολλά φαινόμενα που το σύστημα εκλαμβάνει ως ομοιότητα περιεχομένου ενώ επί της ουσίας δεν είναι επιθυμητό να συνυπολογίζονται (δείτε σχετικά το [FAQ](https://docs.google.com/document/d/1hou1gWXQuHAB7J2aV44xm_CtAWJ63q6Cu1V6OwyL_n0/edit?usp=sharing)). Ταυτόχρονα, μια άλλη κατεύθυνση της βελτιστοποίησης είναι να χρησιμοποιείτε τις παραμέτρους του `TfidfVectorizer` έτσι ώστε να μειώνονται οι διαστάσεις του Vector Space Model μέχρι το σημείο που θα αρχίσει να εμφανίζονται επιπτώσεις στην ποιότητα των συστάσεων. 
# 
# 
# 

# In[8]:


stop = []
for i in range(5000):
    temp = corpus[i].split()
    for j in range(1,len(temp)):
        if temp[j][0].isupper():
            stop.append(temp[j])
print(stop)
        


# Αυτό που έχω κάνει παραπάνω είναι να βγάλω τα ονόματα, αυτό το πέτυχα αφαιρώντας τις λέξεις που ξεκινάνε με κεφαλαίο γράμμα και δεν είναι πρώτες λέξεις μετά από τελεία. Κάτι τέτοιο έχει ρίσκο να χάσουμε σημαντικές λέξεις, αλλά σε μία ταινία τα περισσότερα ονόματα είναι χαρακτήρες τις ταινίες κάτι το οποίο είναι σχεδόν τελειώς ανεξάρτητο με το είδος της ταινίας

# In[13]:


from sklearn.feature_extraction import text
my_stop_words = text.ENGLISH_STOP_WORDS.union(stop)
for i in range(1,11):
    j=i/10;
    vectorizer = TfidfVectorizer(max_df=j, min_df=3,stop_words=my_stop_words)
    corpus_tf_idf=vectorizer.fit_transform(corpus)
    content_recommender(8906,7)


# In[5]:


import nltk

# απαραίτητα downloads για τους stemmer/lemmatizer
nltk.download('wordnet') 
#nltk.download('rslp')
nltk.download('punkt')

from nltk import word_tokenize          
from nltk.stem import WordNetLemmatizer 


# In[6]:


class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
         return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]


# In[10]:


from sklearn.feature_extraction import text
my_stop_words = text.ENGLISH_STOP_WORDS.union(stop)
vectorizer = TfidfVectorizer(max_df=0.5, min_df=3,stop_words=my_stop_words,tokenizer= LemmaTokenizer())
corpus_tf_idf=vectorizer.fit_transform(corpus)
content_recommender(8906,7)


# ## Επεξήγηση επιλογών και ποιοτική ερμηνεία
# 
# Σε markdown περιγράψτε πώς προχωρήσατε στις επιλογές σας για τη βελτιστοποίηση της `TfidfVectorizer`. Επίσης σε markdown δώστε 10 παραδείγματα (IDs) από τη συλλογή σας που επιστρέφουν καλά αποτελέσματα μέχρι `max_recommendations` (5 και παραπάνω) και σημειώστε συνοπτικά ποια είναι η θεματική που ενώνει τις ταινίες.
# 
# Δείτε [εδώ](https://pastebin.com/raw/ZEvg5t3z) ένα παράδειγμα εξόδου του βελτιστοποιημένου συστήματος συστάσεων για την ταίνία ["Q Planes"](https://en.wikipedia.org/wiki/Q_Planes) με την κλήση της συνάρτησης για κάποιο seed `content_recommender(529,3)`. Είναι φανερό ότι η κοινή θεματική των ταινιών είναι τα αεροπλάνα, οι πτήσεις, οι πιλότοι, ο πόλεμος.

# Παραπάνω έχω κάνει τα εξής:
# Έχω βάλει max_fd 0.5 , δηλαδή όταν μία λέξη υπάρχει σε πάνω από το μισό dataset θεωρείται irrelevant. Αυτό διαισθητικά το καταλαβαίνουμε διότι με πάνω από το μισό dataset είναι μία λέξη αρκετή κοινή δεν χαρακτηρίζει κάτι. Το σημαντικό τώρα είναι στα stopwords. Αρχικά είχα βάλει stopwords english που είναι οι λέξεις την Αγγλικής γλώσσας που είναι κοινές και δεν δείχνουν κάτι σπουδαίο όπως is this, that, the κτλπ, τέτοιες λέξεις προφανώς δεν μπορούν να μαρτηρήσουν κάτι για το κείμενο. Στην συνέχεια αυτό που πρόσθεσα ήταν το εξής. Επειδή πολλές ταινίες χαρακτηρίζονταν κοινές με βάση ονόματα χαρακτήρων ή πόλεων κτλπ κάτι που προφανώς δεν το θέλουμε, πρόσθεσα στα stop words όλες τις λέξεις του συνολικού dataset που ξεκινάνε με κεφαλαίο και δεν είναι στην αρχή της πρότασης. Τέτοιες λέξεις είναι κυρίως ονόματα.Ακόμη έχω κάνει lemmatokenize το οποίο κρατάει την ρίζα κάθε λέξης, σημαντικό για να μειώσουμε τον όγκο των λέξεων και επίσης να μην διαφοροποιούνται λέξεις όπως computer, compouters Έχω βάλει για test την barbie mariposa. Παρατηρώ ότι τα περισσότερα recommendations που παίρνω είναι σχετικά εκτός από ένα 

# In[6]:


import joblib

joblib.dump(corpus_tf_idf, 'corpus_tf_idf.pkl') 


# In[11]:


import joblib


# 
# 
# Μπορείτε με ένα απλό `!ls` να δείτε ότι το αρχείο `corpus_tf_idf.pkl` υπάρχει στο filesystem σας (== persistence):

# In[12]:


get_ipython().system('ls -lh')


# και μπορούμε να τα διαβάσουμε με `joblib.load`

# In[13]:


corpus_tf_idf = joblib.load('corpus_tf_idf.pkl')


# # Εφαρμογή 2.  Τοπολογική και σημασιολογική απεικόνιση της ταινιών με χρήση SOM
# <img src="https://i.imgur.com/Z4FdurD.jpg" width="60%">

# ## Δημιουργία dataset
# Στη δεύτερη εφαρμογή θα βασιστούμε στις τοπολογικές ιδιότητες των Self Organizing Maps (SOM) για να φτιάξουμε ενά χάρτη (grid) δύο διαστάσεων όπου θα απεικονίζονται όλες οι ταινίες της συλλογής της ομάδας με τρόπο χωρικά συνεκτικό ως προς το περιεχόμενο και κυρίως το είδος τους (ο παραπάνω χάρτης είναι ενδεικτικός, δεν αντιστοιχεί στο dataset μας). 
# 
# Η `build_final_set` αρχικά μετατρέπει την αραιή αναπαράσταση tf-idf της εξόδου της `TfidfVectorizer()` σε πυκνή (η [αραιή αναπαράσταση](https://en.wikipedia.org/wiki/Sparse_matrix) έχει τιμές μόνο για τα μη μηδενικά στοιχεία). 
# 
# Στη συνέχεια ενώνει την πυκνή `dense_tf_idf` αναπαράσταση και τις binarized κατηγορίες `catbins` των ταινιών ως επιπλέον στήλες (χαρακτηριστικά). Συνεπώς, κάθε ταινία αναπαρίσταται στο Vector Space Model από τα χαρακτηριστικά του TFIDF και τις κατηγορίες της.
# 
# Τέλος, δέχεται ένα ορισμα για το πόσες ταινίες να επιστρέψει, με default τιμή όλες τις ταινίες (5000). Αυτό είναι χρήσιμο για να μπορείτε αν θέλετε να φτιάχνετε μικρότερα σύνολα δεδομένων ώστε να εκπαιδεύεται ταχύτερα το SOM.

# In[14]:


def build_final_set(doc_limit = 5000, tf_idf_only=False):
    # convert sparse tf_idf to dense tf_idf representation
    dense_tf_idf = corpus_tf_idf.toarray()[0:doc_limit,:]
    if tf_idf_only:
        # use only tf_idf
        final_set = dense_tf_idf
    else:
        # append the binary categories features horizontaly to the (dense) tf_idf features
        final_set = np.hstack((dense_tf_idf, catbins[0:doc_limit,:]))
        # η somoclu θέλει δεδομένα σε float32
    return np.array(final_set, dtype=np.float32)


# In[15]:


final_set = build_final_set()


# Τυπώνουμε τις διαστάσεις του τελικού dataset μας. Χωρίς βελτιστοποίηση του TFIDF θα έχουμε περίπου 50.000 χαρακτηριστικά.

# In[22]:


final_set.shape


# Βλέπω ότι τα χαρακτηριστικά μου είναι αισθητά μειωμένα 

# Με βάση την εμπειρία σας στην προετοιμασία των δεδομένων στην επιβλεπόμενη μάθηση, υπάρχει κάποιο βήμα προεπεξεργασίας που θα μπορούσε να εφαρμοστεί σε αυτό το dataset; 

# ## Εκπαίδευση χάρτη SOM
# 
# Θα δουλέψουμε με τη βιβλιοθήκη SOM ["Somoclu"](http://somoclu.readthedocs.io/en/stable/index.html). Εισάγουμε τις somoclu και matplotlib και λέμε στη matplotlib να τυπώνει εντός του notebook (κι όχι σε pop up window).

# In[16]:


# install somoclu
get_ipython().system('pip install --upgrade somoclu')
# import sompoclu, matplotlib
import somoclu
import matplotlib
# we will plot inside the notebook and not in separate window
get_ipython().run_line_magic('matplotlib', 'inline')


# In[17]:


import time


# In[25]:


som1=somoclu.Somoclu(10,10)
print ("1st training som 10x10")
c_s_t = time.time()
som1.train(final_set,epochs=100)
c_f_t = time.time()
print ("took %s seconds" % (c_f_t - c_s_t))


# In[26]:


print("hello")


# In[27]:


som2=somoclu.Somoclu(20,20)
print ("2nd training som 20x20")
c_s_t = time.time()
som2.train(final_set,epochs=50)
c_f_t = time.time()
print ("took %s seconds" % (c_f_t - c_s_t))


# In[28]:


som3=somoclu.Somoclu(25,25)
print ("3rd training som 25x25")
c_s_t = time.time()
som3.train(final_set,epochs=50)
c_f_t = time.time()
print ("took %s seconds" % (c_f_t - c_s_t))


# 
# ## Best matching units
# 
# Μετά από κάθε εκπαίδευση αποθηκεύστε σε μια μεταβλητή τα best matching units (bmus) για κάθε ταινία. Τα bmus μας δείχνουν σε ποιο νευρώνα ανήκει η κάθε ταινία. Προσοχή: η σύμβαση των συντεταγμένων των νευρώνων είναι (στήλη, γραμμή) δηλαδή το ανάποδο από την Python. Με χρήση της [np.unique](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.unique.html) (μια πολύ χρήσιμη συνάρτηση στην άσκηση) αποθηκεύστε τα μοναδικά best matching units και τους δείκτες τους (indices) προς τις ταινίες. Σημειώστε ότι μπορεί να έχετε λιγότερα μοναδικά bmus από αριθμό νευρώνων γιατί μπορεί σε κάποιους νευρώνες να μην έχουν ανατεθεί ταινίες. Ως αριθμό νευρώνα θα θεωρήσουμε τον αριθμό γραμμής στον πίνακα μοναδικών bmus.
# 

# In[29]:


bmus1 = som1.bmus
ubmus1, indices1 = np.unique(bmus1, return_inverse=True, axis=0)
print(indices1)


# In[37]:


ls = [0]*100
for i in indices1:
    ls[i]+=1
print(ls)


# In[31]:


bmus2 = som2.bmus
ubmus2, indices2 = np.unique(bmus2, return_inverse=True, axis=0)
print(indices2)


# In[32]:


bmus3 = som3.bmus
ubmus3, indices3 = np.unique(bmus3, return_inverse=True, axis=0)
print(indices3)


# In[46]:


class mySOMS:
    def __init__(self, SOMs_dict):
        self.SOMS = SOMs_dict

mySoms = mySOMS(SOMs)
joblib.dump(mySoms,'mySOMS.pkl')


# In[18]:


class mySOMS:
    def __init__(self, SOMs_dict):
        self.SOMS = SOMs_dict


# In[40]:


SOMs = {}
SOMs['som10'] = som1
SOMs['som20'] = som2
SOMs['som25'] = som3


# In[42]:


get_ipython().system('ls')


# 
# ## Ομαδοποίηση (clustering)
# 
# Τυπικά, η ομαδοποίηση σε ένα χάρτη SOM προκύπτει από το unified distance matrix (U-matrix): για κάθε κόμβο υπολογίζεται η μέση απόστασή του από τους γειτονικούς κόμβους. Εάν χρησιμοποιηθεί μπλε χρώμα στις περιοχές του χάρτη όπου η τιμή αυτή είναι χαμηλή (μικρή απόσταση) και κόκκινο εκεί που η τιμή είναι υψηλή (μεγάλη απόσταση), τότε μπορούμε να πούμε ότι οι μπλε περιοχές αποτελούν clusters και οι κόκκινες αποτελούν σύνορα μεταξύ clusters.
# 
# To somoclu δίνει την επιπρόσθετη δυνατότητα να κάνουμε ομαδοποίηση των νευρώνων χρησιμοποιώντας οποιονδήποτε αλγόριθμο ομαδοποίησης του scikit-learn. Στην άσκηση θα χρησιμοποιήσουμε τον k-Means. Για τον αρχικό σας χάρτη δοκιμάστε ένα k=20 ή 25. Οι δύο προσεγγίσεις ομαδοποίησης είναι διαφορετικές, οπότε περιμένουμε τα αποτελέσματα να είναι κοντά αλλά όχι τα ίδια.
# 

# In[19]:


from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from copy import copy


# In[3]:


print("hello")


# In[20]:


mySoms = joblib.load('mySOMS.pkl')


# In[21]:


SOMs = mySoms.SOMS


# Τώρα κάνουμε κατηγοροποιήση(clusetering) σε κάθε SOM για k=20,25
ks = { "k_20":20, "k_25": 25} 
for som in SOMs:
    for K in ks:
        k_means = KMeans(n_clusters= ks[K])
        SOMs[som].cluster(algorithm = k_means)
        act_map = SOMs[som].get_surface_state()   
        file_name = som + "_" + K + ".pkl"
        joblib.dump(SOMs[som], file_name)


# In[ ]:





# 
# ## Αποθήκευση του SOM
# 
# Επειδή η αρχικοποίηση του SOM γίνεται τυχαία και το clustering είναι και αυτό στοχαστική διαδικασία, οι θέσεις και οι ετικέτες των νευρώνων και των clusters θα είναι διαφορετικές κάθε φορά που τρέχετε τον χάρτη, ακόμα και με τις ίδιες παραμέτρους. Για να αποθηκεύσετε ένα συγκεκριμένο som και clustering χρησιμοποιήστε και πάλι την `joblib`. Μετά την ανάκληση ενός SOM θυμηθείτε να ακολουθήσετε τη διαδικασία για τα bmus.
# 

# In[26]:


soms={"som10_k_20.pkl":20,"som10_k_25.pkl":25,"som20_k_20.pkl":20,"som20_k_25.pkl":25,"som25_k_20.pkl":20,"som25_k_25.pkl":25}
for som in soms:
    current_som = joblib.load(som)
    current_bmus = current_som.get_bmus(current_som.activation_map)
    current_ubmus = np.unique(current_bmus, axis=1)


    # Εκτύπωση του τρέχοντος SOM 
    print (som)
    current_som.view_umatrix(figsize=(10,10), bestmatches = True, colorbar = True)
    print ("\n")
    print (current_som.clusters)
    print ('\n')
    num_neuro_cluster= np.unique(current_som.clusters, return_counts = True)
    num_neuro_cluster_sorted =  np.argsort(num_neuro_cluster)
    print ("Number of neuros in each cluster")
    print (num_neuro_cluster_sorted[1])
    print ('\n')


# In[23]:


## 1. Ορισμός της print_categories_stats
# Για τις ταινίες των οποίων το id περιέχεται στην λίστα που περνάμε σαν όρισμα 
# στη συνάρτηση αυτή, βρίσκουμε σε ποιες κατηγορίες ανήκουν και πόσες από αυτές 
# τις ταινίες ανήκουν σε κάθε κατηγορία
def print_categories_stats(ids):
    categ = []
    for Id in ids:
        catStr = categories.item(Id)
        catStr = catStr.strip() 
        curMovCats = catStr.split(",")
        for indx in range(0,len(curMovCats)):
            curMovCats[indx] = curMovCats[indx].strip()
        categ.extend(curMovCats)
    uniqCats = np.unique(categ, return_counts = True)
    uniqCatsSort = np.argsort(uniqCats[1])
    #print uniqCats
    #print  uniqCatsSort
    listPrint = []
    for i in range(0, len(uniqCatsSort)):
        #print str(i) + str(uniqCats[0][uniqCatsSort[1][i]])+ " "+str(uniqCats[1][uniqCatsSort[1][i]])
        mvTuple=(uniqCats[0][uniqCatsSort[i]], uniqCats[1][uniqCatsSort[i]])
        #print mvTuple
        listPrint.append(mvTuple)
    listPrint.reverse()
    print (listPrint) 



# Ορισμός της print_cluster_neuros_movies_report
# Δέχεται σαν όρισμα το Id κάποιου cluster φτιάχνει μια λίστα
# με τις συντεταγμένες των νευρώνων που ανήκουν στο cluster,
# ψάχνει να βρει ποιες ταινίες αντιστοιχούν σε κάθε νευρώνα(bmus) 
# και τις βάζει στην αντίστοιχη λίστα. 
def print_cluster_neurons_movies_report(clustΙd,curSom):
    movList=[]
    neurons_per_cluster_coords = np.where(curSom.clusters == clustΙd)
    neurons_per_cluster = np.column_stack(neurons_per_cluster_coords)
    movList = neuron_movies_report(neurons_per_cluster)
    print_categories_stats(movList)
    #cur_movie_bmus = np.where(best_matching_units = clustΙd)
    #cur_movie_bmus_stack = np.column_stack(cur_movies_bmus)

# Δέχεται σαν είσοδο μία λίστα συντεταγμένων νευρώνων και βρίσκει 
# ποιες τανίες ανήκουν σε αυτούς τους νευρώνες και επιστρέφει μία 
# λίστα με τα ids των ταινίων που αντιστοιχούν σε αυτούς τους νευρώνες
def neuron_movies_report(neurons):
    movListId = []
    falseVal = len(current_bmus)+1
    for neuron in neurons:
        Id = check_bmus(neuron, current_bmus)
        if Id != falseVal:
            movListId.append(Id)
    return movListId

def check_bmus(bestΜatch, bestMatches):
    for Id in range(len(bestMatches)):
        row = bestMatches[Id][1]
        column = bestMatches[Id][0]
        bmrow = bestΜatch[0]
        bmcolumn = bestΜatch[1]
        if (row == bmrow and column == bmcolumn):
            return Id
    return (len(bestMatches)+1)


# In[27]:


for i in [10,20,25]:
    for j in [20,25]:
        som_name = 'som'+ str(i)+'_k_'+  str(j)
        filename1 = som_name + '.pkl'
        current_som = joblib.load(filename1)
        current_bmu = current_som.get_bmus(current_som.activation_map)
        unique_current_bmu = np.unique(current_bmu, axis=1)


        # Εκτύπωση του τρέχοντος SOM 
        print (som_name)
        current_som.view_umatrix(figsize=(8,8), bestmatches = True, colorbar = True)
        print ("\n")
        rand = np.random.randint(0,j, [5,1])
        for index in rand:
            print ('cluster index  ' + str(index) + '\n')
            print_cluster_neurons_movies_report(index, current_som)
            print ('\n')
            


# In[33]:


som_final = 'som25_k_20.pkl'
current_som_final = joblib.load(som_final)
best_matching_units = current_som_final.get_bmus(current_som_final.activation_map)
unique_best_matching_units = np.unique(best_matching_units, axis=1)


# Εκτύπωση του τρέχοντος SOM 
print (som_final)
current_som_final.view_umatrix(figsize=(10,10), bestmatches = True, colorbar = True)
print ("\n")
print (current_som_final.clusters)
print ("\n")
objects_per_cluster = np.unique(current_som_final.clusters, return_counts = True)
objects_per_cluster_sorted =  np.argsort(objects_per_cluster)
print ("Number of neuros in each cluster")
print (objects_per_cluster_sorted[1])
print ('\n')

for index in range(0,20):
    print ('cluster index  ' + str(index))
    print ('__________________')
    print_cluster_neurons_movies_report(index, current_som_final)
    print ('\n')


# 
# 
# ## Ανάλυση τοπολογικών ιδιοτήτων χάρτη SOM
# 
# 

# Παρακάτω παρουσιάζονται οι κατηγορίες των ταινιών ταξινομιμένες ως προς το πλήθος των ταινιών
# 

# In[100]:


temp = []
for i in categories:
    lst = list(i)
    new = lst[0].split(",")
    temp+=new
for i in range(len(temp)):
    temp[i] = ''.join(ch for ch in temp[i] if ch not in exclude)
    temp[i]=temp[i].replace(' ','')


# In[101]:


import re, string, timeit
stats = {}
exclude = set(string.punctuation)
for i in temp:
    if i in stats:
        t = stats[i]
        t+=1
        stats[i] = t
    else:
        stats[i]=1


# In[106]:


k=dict(sorted(stats.items(), key=lambda item: item[1],reverse=True))


# In[107]:


print(k)


# - Αρχικά παρατηρώ ότι όσο μεγαλύτερη η πιθανότητα εμφάνισης μίας κατηγορίας τόσο αυξημένο είναι το μέγεθος των clusters που την εκπροσωπούν. Η κατηγορία "Δράμα" που όπως βλέπω παραπάνω είναι και η πιο συχνή κατηγορία, παρουσιάζεται σε όλα τα clusters σχεδόν και από τις πρώτες κατηφορίες. Βλέπω ότι τους περισσότερους νευρώνες τους έχει η κατηγορία 12 (number of neurons είναι μία λίστα με τα indexes των clusters ταξινομημένη σύμφωνα με το πόσους νευρώνες έχει το κάθε cluster από την μικρότερη τιμή στην μεγαλύτερη). Αυτό είναι απολύτως λογικό αφού η κατηγορία 12 έχει μεγάλη πιθανότητα λόγω του ότι έχει πάρα πολλές ταινίες. Το ίδιο παρατηρούμαι και για την κατηγορία 2 που είναι η δεύτερη μεγαλύτερη σε νευρώνες.
#    
# - Όσον αφορά τα απομακρυσμένα σημεία στο χάρτη ορισμένα παραδείγματα είναι τα εξής:
# Το cluster 9 (κυρίως 'Indie') βρίσκεται σε απομακρσμένο σημείο στο χάρτη. Αντίστοιχο φαινόμενο παρατηρείται και σε μικρότερους χάρτες. 
# 
# - Παρατηρώ πληθώρα κοινών κατηγοριών που απεικονίζονται σε κοντινά σημεία στο χάρτη. Παραδείγματα είναι:
# Οι κατηγορίες 19 και 10 που είναι για horror movies και thrillers, οι κατηγορίες 11 και 13 που είναι Crime Fiction και World cinema και δράσης. Οι κατηγορίες 0 και 16 που είναι action και thriller η μία και action και drama η άλλη 
