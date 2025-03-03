"""
The evaluation.py is used to evaluation 

Runs in Python 3.12.4.

@author: Jin-Zhou Lee
"""

#%% pacakges
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%% read data
# 
# 
#   

#%% read data
content = pd.read_csv('data/content.csv', usecols=['科別', '問題主旨', '內文'])
content['內文擴展'] = content['問題主旨'] + '\n' + content['內文']

#%% train and test
from sklearn.model_selection import train_test_split

# 70% train, 30% test
train, test = train_test_split(content, test_size=0.3, random_state=1, stratify=content["科別"])

#%% Preprocess 
#
#
#

#%% Put train ans often question together
def train_explode():
    '''
    Put train ans often question together

    Returns
    -------
    None.

    '''
    global train
    
    # check the departmet in the train
    depart = train['科別'].unique()

    # read often question 
    often_question = pd.read_pickle('data/often_question.pkl')
    # select depart
    often_question = often_question[often_question['科別'].isin(depart)]
    # explode questinos
    often_question = often_question.explode('問題', ignore_index=True)
    # rename
    often_question = often_question.rename(columns = {'問題':'內文擴展'})
    
    # combain data
    train = pd.concat([train, often_question[['科別', '內文擴展']]], ignore_index=True)
    
train_explode()

#%% word segmentate
import jieba
import unicodedata
import re

jieba.set_dictionary('model/dict.txt.big')

def pre_process(text):
    move_chr = r'[^\u4e00-\u9fffA-Za-z,\s]'
    
    # full-width font to half-width
    text = unicodedata.normalize("NFKD", text)
    
    # to lower
    text = str(text).lower()
    
    # Remove URL
    text = re.sub(r'http[s]?://\S+|www\.\S+', '', text)
    
    # Remove non-Chinese characters and English
    text = re.sub(move_chr, '', text)
    
    # segmentate
    text = list(jieba.cut_for_search(text))
    
    # slove spaces problem
    text = ' '.join(text)
    
    # Merge multiple consecutive spaces and line breaks into a single space
    text = re.sub(r'[\s+,]+', ' ', text)
    
    # Remove leading and trailing spaces
    text = text.strip()
    
    # finally remove space
    text = text.split()
    
    return text

train['segmentate'] = train['內文擴展'].apply(pre_process)
test['segmentate'] = test['內文擴展'].apply(pre_process)
# merge train and test data
content_FAQ = pd.concat([train, test], ignore_index=True)

#%% Modeling
# 
# 
# 

#%% Labeled - LDA, prepare corpus
import tomotopy as tp
from tomotopy.utils import Corpus

def creat_corpus():
    global train_corpus, test_corpus
    # creat a corpus
    train_corpus = Corpus()
    test_corpus = Corpus()
    
    # add content and label
    for index, row in train.iterrows():
        train_corpus.add_doc(words=row["segmentate"], labels=[row["科別"]])

    for index, row in test.iterrows():
        test_corpus.add_doc(words=row["segmentate"])

creat_corpus()

#%% train LLDA
# creat a llda model
llda = tp.PLDAModel(tw = tp.TermWeight.IDF, seed = 1) # use tf-idf

# put in the corpus
llda.add_corpus(train_corpus)

# training modeling, set the number of train
llda.train(1000, show_progress = True)

# sum summary
llda.summary()

#%% Evaluation
# 
# 
# 

#%% diagnosis function
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

class diagnosis():
    def __init__(self, real, pred):
        self.real = real # the real label
        self.pred = pred # the predict label
        
        self.name = sorted(set(self.real)) # the label names
        self.confusion_matrix() # confusion matrix
        
    def classification_report(self):
        print(classification_report(self.real, self.pred))
        
    def confusion_matrix(self):
        self.cm = confusion_matrix(self.real, self.pred)
        
    def plt_cm(self, save_dir = None):
        '''
        Plot confusion matrix. 
        '''
        plt.figure(figsize=(19.2, 10.8)) # set figure size

        plt.rc('font', family='Microsoft JhengHei') # set chinese font
        plt.xticks(fontsize=12)  # set x-axe font size
        plt.yticks(fontsize=12)  # set y-axe font size
        
        # confusion matrix set up
        fig, ax = plt.subplots()
        cax = ax.matshow(self.cm, cmap='Blues')
        
        # axe set up 
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        num_classes = self.cm.shape[0] # number of type of lables 
        ax.set_xticks(np.arange(num_classes)) # set number
        ax.set_yticks(np.arange(num_classes)) # set number
        ax.set_xticklabels(self.name) # set text
        ax.set_yticklabels(self.name) # set text
        
        # Set the rotation angle of the x-axis tick labels
        plt.xticks(rotation=45, ha='left')
        
        # add a value to each square
        for (i, j), value in np.ndenumerate(self.cm):
            ax.text(j, i, value, ha='center', va='center', color='black')
        
        # add a color bar
        plt.colorbar(cax)
        
        # save figure
        if save_dir:
            plt.savefig(save_dir, dpi=200, bbox_inches='tight') 
        
        # show figure
        plt.show()

#%% diagnosis LLDA with train accuracy
# make train diagnosis
def train_diagnosis() -> diagnosis:
    # the real label
    real_labels = train["科別"].tolist()
    
    # creat a no-label corpus
    train_corpus_nolabel = Corpus()
    for index, row in train.iterrows():
        train_corpus_nolabel.add_doc(words=row["segmentate"])
    # pred the train label
    topic_dist, ll = llda.infer(train_corpus_nolabel)
    
    # the predicted labels 
    predicted_labels = [doc.get_topics()[0][0] for doc in topic_dist] # in number form
    predicted_labels = [llda.topic_label_dict[i] for i in predicted_labels] # in text form
    
    # diagnosis
    return diagnosis(real_labels, predicted_labels)
    
train_diag = train_diagnosis()
train_diag.classification_report()
train_diag.plt_cm(save_dir = "img/llda train cm")

#%% diagnosis LLDA with test accuracy
# make test diagnosis
def test_diagnosis() -> diagnosis:
    # the real label
    real_labels = test["科別"].tolist()
    
    # pred the train label
    topic_dist, ll = llda.infer(test_corpus)
    
    plt.hist(ll, bins=30, edgecolor='black')
    plt.show()  # 顯示圖形
    
    # the predicted labels 
    predicted_labels = [doc.get_topics()[0][0] for doc in topic_dist] # in number form
    predicted_labels = [llda.topic_label_dict[i] for i in predicted_labels] # in text form
    
    # diagnosis
    return diagnosis(real_labels, predicted_labels)
    
test_diag = test_diagnosis()
test_diag.classification_report()
test_diag.plt_cm(save_dir = "img/llda test cm")

#%% train a full data model 
full_corpus = Corpus()

# add content and label
for index, row in content_FAQ.iterrows():
    full_corpus.add_doc(words=row["segmentate"], labels=[row["科別"]])

# train LLDA
# creat a llda model
llda = tp.PLDAModel(tw = tp.TermWeight.IDF, seed = 1) # use tf-idf

# put in the corpus
llda.add_corpus(full_corpus)

# training modeling, set the number of train
llda.train(1000, show_progress = True)

# sum summary
llda.summary()

# save model
llda.save('model/llda.bin')

#%% SVM
#
#
#

#%% train SVM
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC # C-Support Vector Classification

vectorizer = TfidfVectorizer()

def train_svm():
    global svm_train_x, svm_train_y, svm_test_x, svm_test_y
    # vectorizer
        
    # trian pre-process
    svm_train_x = train["segmentate"].apply(lambda x: ' '.join(x))
    svm_train_x = vectorizer.fit_transform(svm_train_x)
    svm_train_y = train["科別"]
    
    # test pre-process
    svm_test_x = test["segmentate"].apply(lambda x: ' '.join(x))
    svm_test_x = vectorizer.transform(svm_test_x)
    svm_test_y = test["科別"]
    
    # train svm
    svm = SVC(kernel='linear') # kernel: 'linear', 'rbf', 'poly' 
    svm.fit(svm_train_x, svm_train_y)
    
    return svm

svm = train_svm()
    
#%% diagnosis svm
svm_train_diag = diagnosis(real = svm_train_y, pred = svm.predict(svm_train_x))
svm_train_diag.classification_report()
svm_train_diag.plt_cm(save_dir = "img/svm train cm")

svm_test_diag = diagnosis(real = svm_test_y, pred = svm.predict(svm_test_x))
svm_test_diag.classification_report()
svm_test_diag.plt_cm(save_dir = "img/svm test cm")

#%% Multinomial Naive Bayes
#
#
#

#%% train
from sklearn.naive_bayes import MultinomialNB

mnb = MultinomialNB()
mnb.fit(svm_train_x, svm_train_y)

# diagnosis svm
mnb_train_diag = diagnosis(real = svm_train_y, pred = mnb.predict(svm_train_x))
mnb_train_diag.classification_report()
mnb_train_diag.plt_cm(save_dir = "img/svm train cm")

mnb_test_diag = diagnosis(real = svm_test_y, pred = mnb.predict(svm_test_x))
mnb_test_diag.classification_report()
mnb_test_diag.plt_cm(save_dir = "img/svm test cm")

#%% test mnb
def mnb_test(text):
    text = ' '.join(pre_process(text))
    token = vectorizer.transform([text])
    
    prob = mnb.predict_proba(token)
    
    class_labels = mnb.classes_
        
    df = pd.DataFrame(prob, columns=class_labels).T.sort_values(by=0, ascending=False)
    
    print(df)

#text = "心雜音"
#mnb_test(text)

#%% fasttext
#
#
#

#%% prepare data
train_text = content_FAQ["segmentate"].apply(lambda x: ' '.join(x)).tolist()

with open('data/fasttext_train.txt', 'w', encoding='utf-8') as f:
    for line in train_text:
        f.write(line + '\n')

#%% fasttext
'''
if pip install fasttext not work, try
pip install fasttext-wheel
'''
import fasttext

# train model 
w2v = fasttext.train_unsupervised("data/fasttext_train.txt", model='skipgram')

# save model
w2v.save_model('model/fasttext.bin')

#%% try nearest neighbors
w2v.get_nearest_neighbors('眼')  # 獲取最近字
w2v.get_nearest_neighbors('腳')  # 獲取最近字
w2v.get_nearest_neighbors('膝蓋')  # 獲取最近字

#%% key value
# 
# 
# 

#%% key value
word_topic = [llda.get_topic_word_dist(i) for i in range(llda.k)]

# set row and col name
word_topic = pd.DataFrame(word_topic)
word_topic.columns = list(llda.used_vocabs)
word_topic.index = list(llda.topic_label_dict)

# normalize
word_topic = word_topic.div(word_topic.sum(axis=0), axis=1)

def entropy(prob_dist):
    # filter out values ​​with zero probability to avoid log(0)
    prob_dist = np.array(prob_dist)
    prob_dist = prob_dist[prob_dist > 0]
    return np.sum(prob_dist * np.log(prob_dist))

# calculate the entropy of each row
entropies = word_topic.apply(entropy, axis=0)

key_value = np.exp(entropies)

key_value.to_pickle('model/key_value.pkl')

#%% EDA keyvalue
key_value.sort_values().head(10)
key_value.sort_values().tail(10)

#%% keyword to 
key_value = pd.read_pickle("model/key_value.pkl")
w2v = fasttext.load_model("model/fasttext.bin")

def doc_to_key(doc):
    # key value and word vector
    keyvalue = pd.DataFrame([key_value.get(word, 0.1) for word in doc])
    word_vectors = pd.DataFrame([w2v.get_word_vector(word) for word in doc])
     
    # The key value is multiplied by the word vectors row by row
    key_matrix = pd.DataFrame(keyvalue.values * word_vectors.values)
    
    # key vector
    key_vector = key_matrix.sum(axis = 0)
    
    # get unique index
    unique_indices, inverse_indices = np.unique(doc, return_inverse=True)
    
    # add the value with same index 
    aggregated_values = np.bincount(inverse_indices, weights=keyvalue[0])
    
    # merge the result
    key_value_sort = pd.DataFrame(np.column_stack((unique_indices, aggregated_values)))
    
    # the keywords 
    keywords = key_value_sort.sort_values(by=1, ascending=False)
    key_words = keywords.iloc[:10, 0].reset_index(drop=True)
    
    return key_vector, key_words

content['segmentate'] = content["內文擴展"].apply(pre_process)

keys = content["segmentate"].apply(doc_to_key)
key_vec = keys.apply(lambda x: x[0])
key_word = keys.apply(lambda x: x[1])

key_vec.to_pickle('model/key_vec.pkl')
key_word.to_pickle('model/key_word.pkl')

#%% doc similarity
# 
# 
# 

#%% doc similarity
from sklearn.metrics.pairwise import cosine_similarity

# the similarity matrix of words
similarity_matrix = cosine_similarity(key_vec)

def print_doc(index):
    print("index:", index)
    print()
    print(content.loc[index, "問題主旨"])
    print()
    print(content.loc[index, "內文"])
    print("=" * 80)

index = 2
near_index = np.argsort(similarity_matrix[content.index[index]])[::-1]

# hist plot
plt.hist(similarity_matrix[content.index[index]], bins=20)
for near in near_index[:5]:
    print("sim:", similarity_matrix[content.index[index]][near])
    print_doc(near)
