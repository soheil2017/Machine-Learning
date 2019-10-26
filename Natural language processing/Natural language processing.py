import nltk
nltk.download('stopwords')
#nltk.download_shell()
messages = [line.rstrip() for line in open('smsspamcollection/SMSSpamCollection')]
print(len(messages))

print(messages[50])

for message_no, message in enumerate(messages[:10]):
    print(message_no, message)
    print('\n')

print(messages[0])

import pandas as pd

messages = pd.read_csv('smsspamcollection/SMSSpamCollection', sep='\t',
                           names=["label", "message"])

print(messages.head())
print("="*40)

print(messages.describe())
print("="*40)

print(messages.groupby('label').describe())
print("="*40)

messages['length'] = messages['message'].apply(len)
print(messages.head())

#Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns

messages['length'].plot.hist(bins=50)
plt.show()

print(messages.length.describe())

messages[messages['length'] == 910]['message'].iloc[0]

messages.hist(column='length', by='label', bins=50,figsize=(12,4))
plt.show()

#Text Pre-processing
#In this section we'll convert the raw messages (sequence of characters) into vectors (sequences of numbers).
import string

mess = 'Sample message! Notice: it has punctuation.'
# Check characters to see if they are in punctuation
nopunc = [char for char in mess if char not in string.punctuation]

# Join the characters again to form the string.
nopunc = ''.join(nopunc)

from nltk.corpus import stopwords
print(stopwords.words('english')[0:10]) # Show some stop words

nopunc.split()

# Now just remove any stopwords
clean_mess = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
print(clean_mess)
print("="*40)

def text_process(mess):
    """
    Takes in a string of text, then performs the following:
    1. Remove all punctuation
    2. Remove all stopwords
    3. Returns a list of the cleaned text
    """
    # Check characters to see if they are in punctuation
    nopunc = [char for char in mess if char not in string.punctuation]

    # Join the characters again to form the string.
    nopunc = ''.join(nopunc)

    # Now just remove any stopwords
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

print(messages.head())
# Check to make sure its working
print(messages['message'].head(5).apply(text_process))
print("="*40)

# Show original dataframe
print(messages.head())
print("="*40)

#Continuing Normalization

#Vectorization
from sklearn.feature_extraction.text import CountVectorizer

# Might take awhile...
bow_transformer = CountVectorizer(analyzer=text_process).fit(messages['message'])

# Print total number of vocab words
print(len(bow_transformer.vocabulary_))
print("="*40)

message4 = messages['message'][3]
print(message4)
print("="*40)

bow4 = bow_transformer.transform([message4])
print(bow4)
print(bow4.shape)
print("="*40)

print(bow_transformer.get_feature_names()[4073])
print(bow_transformer.get_feature_names()[9570])
print("="*40)

messages_bow = bow_transformer.transform(messages['message'])
print('Shape of Sparse Matrix: ', messages_bow.shape)
print('Amount of Non-Zero occurences: ', messages_bow.nnz)
print("="*40)

sparsity = (100.0 * messages_bow.nnz / (messages_bow.shape[0] * messages_bow.shape[1]))
print('sparsity: {}'.format(round(sparsity)))
print("="*40)

from sklearn.feature_extraction.text import TfidfTransformer

# TF-IDF
tfidf_transformer = TfidfTransformer().fit(messages_bow)
tfidf4 = tfidf_transformer.transform(bow4)
print(tfidf4)
print("="*40)

print(tfidf_transformer.idf_[bow_transformer.vocabulary_['u']])
print(tfidf_transformer.idf_[bow_transformer.vocabulary_['university']])
print("="*40)

messages_tfidf = tfidf_transformer.transform(messages_bow)
print(messages_tfidf.shape)
print("="*40)

#Training a model
#With messages represented as vectors, we can finally train our spam/ham classifier.
# Now we can actually use almost any sort of classification algorithms. For a variety of reasons, the Naive Bayes classifier algorithm is a good choice.
from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(messages_tfidf, messages['label'])

print('predicted:', spam_detect_model.predict(tfidf4)[0])
print('expected:', messages.label[3])
print("="*40)

#Model Evaluation
all_predictions = spam_detect_model.predict(messages_tfidf)
print(all_predictions)
print("="*40)

from sklearn.metrics import classification_report
print (classification_report(messages['label'], all_predictions))
print("="*40)

from sklearn.model_selection import train_test_split
msg_train, msg_test, label_train, label_test = \
train_test_split(messages['message'], messages['label'], test_size=0.2)

print(len(msg_train), len(msg_test), len(msg_train) + len(msg_test))
print("="*40)

#Creating a Data Pipline
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer=text_process)),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
])
pipeline.fit(msg_train,label_train)
predictions = pipeline.predict(msg_test)
print(classification_report(predictions,label_test))
print("="*40)
