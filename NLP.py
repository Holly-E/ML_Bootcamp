import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import nltk
import string

#opens interactive shell, dwnld stopwords for the most commonly used words
#nltk.download_shell()

#messages = [line.rstrip() for line in open('Data/smsspamcollection/SMSSpamCollection')]
#print(len(messages))

#for ind, message in enumerate(messages[:10]):
#    print(ind, message)

messages = pd.read_csv('Data/smsspamcollection/SMSSpamCollection', sep='\t', names=['label', 'message'])
messages['length']=messages['message'].apply(len)

# text msgs that are spam tend to be longer
# messages.hist(column='length', by='label', bins=60)

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
"""
All the following can be performed with a pipeline (see below): 

# clean messages and convert to a sequence of vectors, 'bag of words'
from nltk.corpus import stopwords
# custom analyzer - note this is for learning and not necessary with the count vectorizer
def text_process(mess):
    no_punc = [char for char in mess if char not in string.punctuation]
    no_punc = ''.join(no_punc)
    return [word for word in no_punc.split() if word.lower() not in stopwords.words('english')]

bow_transformer = CountVectorizer(analyzer= text_process).fit(messages['message'])
#print (len(bow_transformer.vocabulary_))

messages_bow= bow_transformer.transform(messages['message'])
print ('Shape of Sparse Matrix:', messages_bow.shape)

# % non zeros in sparsity matrix
sparsity = (100.0 * messages_bow.nnz / (messages_bow.shape[0] * messages_bow.shape[1]))
print('sparsity: {}'.format((sparsity)))

tfidf_transformer = TfidfTransformer().fit(messages_bow)
messages_tfidf = tfidf_transformer.transform(messages_bow)
print(messages_tfidf.shape)

spam_detect_model = MultinomialNB().fit(messages_tfidf, messages['label'])

#Predict & evaluate model
all_predictions = spam_detect_model.predict(messages_tfidf)
print (classification_report(messages['label'], all_predictions))

"""

from sklearn.model_selection import train_test_split
msg_train, msg_test, label_train, label_test = train_test_split(messages['message'], messages['label'], test_size=0.2)

from sklearn.pipeline import Pipeline


pipeline = Pipeline([
    ('bow', CountVectorizer()),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
])

pipeline.fit(msg_train,label_train)
predictions = pipeline.predict(msg_test)
print(classification_report(predictions,label_test))

plt.show()