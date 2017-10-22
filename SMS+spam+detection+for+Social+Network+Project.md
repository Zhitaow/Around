
## Overview

The challenge of the social network project is that a simple keyword-based filtering method is only limited to some sensitive words. This section is about filtering spam messages from the social network platform using machine learning algorithm. To improve the spam message classifier, I obtained a realistic dataset of SMS text messages from the [UCI datasets](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection), which is also already downloaded for you under the same directory.

## Getting the Data


```python
messages = [line.rstrip() for line in open('smsspamcollection/SMSSpamCollection')]
print('Total sample size: ', len(messages), '\n')
# print first 10 messages
for message_no, message in enumerate(messages[:10]):
    print(message_no, message)
    print('\n')
```

    Total sample size:  5574 
    
    0 ham	Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat...
    
    
    1 ham	Ok lar... Joking wif u oni...
    
    
    2 spam	Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's
    
    
    3 ham	U dun say so early hor... U c already then say...
    
    
    4 ham	Nah I don't think he goes to usf, he lives around here though
    
    
    5 spam	FreeMsg Hey there darling it's been 3 week's now and no word back! I'd like some fun you up for it still? Tb ok! XxX std chgs to send, £1.50 to rcv
    
    
    6 ham	Even my brother is not like to speak with me. They treat me like aids patent.
    
    
    7 ham	As per your request 'Melle Melle (Oru Minnaminunginte Nurungu Vettam)' has been set as your callertune for all Callers. Press *9 to copy your friends Callertune
    
    
    8 spam	WINNER!! As a valued network customer you have been selected to receivea £900 prize reward! To claim call 09061701461. Claim code KL341. Valid 12 hours only.
    
    
    9 spam	Had your mobile 11 months or more? U R entitled to Update to the latest colour mobiles with camera for Free! Call The Mobile Update Co FREE on 08002986030
    
    


## Exploring the Data


```python
import pandas as pd
messages = pd.read_csv('smsspamcollection/SMSSpamCollection', sep='\t',
                           names=["label", "message"])
messages.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>label</th>
      <th>message</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ham</td>
      <td>Go until jurong point, crazy.. Available only ...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ham</td>
      <td>Ok lar... Joking wif u oni...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>spam</td>
      <td>Free entry in 2 a wkly comp to win FA Cup fina...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ham</td>
      <td>U dun say so early hor... U c already then say...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ham</td>
      <td>Nah I don't think he goes to usf, he lives aro...</td>
    </tr>
  </tbody>
</table>
</div>




```python
# check how many spammer and hammer
messages.groupby('label').describe()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="4" halign="left">message</th>
    </tr>
    <tr>
      <th></th>
      <th>count</th>
      <th>unique</th>
      <th>top</th>
      <th>freq</th>
    </tr>
    <tr>
      <th>label</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ham</th>
      <td>4825</td>
      <td>4516</td>
      <td>Sorry, I'll call later</td>
      <td>30</td>
    </tr>
    <tr>
      <th>spam</th>
      <td>747</td>
      <td>653</td>
      <td>Please call our customer service representativ...</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>



## Text Pre-processing


```python
import string
from nltk.corpus import stopwords
# Show some stop word examples
stopwords.words('english')[0:10] 
```




    ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your']




```python
def text_process(message):
    """
    Read in a segment of text message, then performs the following pre-processing steps:
    1. Remove all punctuation
    2. Remove all stopwords
    3. Returns a list of the cleaned text
    """
    # Check characters to see if they are in punctuation
    nopunc = [char for char in message if char not in string.punctuation]

    # Join the characters again to form the string.
    nopunc = ''.join(nopunc)
    
    # Now just remove any stopwords
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
```


```python
# test on a small set of samples:
messages['message'].head(5).apply(text_process)
```




    0    [Go, jurong, point, crazy, Available, bugis, n...
    1                       [Ok, lar, Joking, wif, u, oni]
    2    [Free, entry, 2, wkly, comp, win, FA, Cup, fin...
    3        [U, dun, say, early, hor, U, c, already, say]
    4    [Nah, dont, think, goes, usf, lives, around, t...
    Name: message, dtype: object




```python
# split the sample into training set (80%) and cross-validation set (20%)
from sklearn.model_selection import train_test_split
msg_train, msg_test, label_train, label_test = \
train_test_split(messages['message'], messages['label'], test_size=0.2)
print(" Total sample size: ", len(msg_train) + len(msg_test), "\n Total # of training set: ", len(msg_train),\
      "\n Total # of test set: ", len(msg_test))
```

     Total sample size:  5572 
     Total # of training set:  4457 
     Total # of test set:  1115



```python
from sklearn.feature_extraction.text import CountVectorizer
# convert to bag of words
bow_transformer = CountVectorizer(analyzer=text_process).fit(msg_train)
# Print total number of vocab words
#print(len(bow_transformer.vocabulary_))
#print(bow_transformer.vocabulary_.head(10))
```


```python
# Print top 10 word counts
i = 10;
for key, value in bow_transformer.vocabulary_.items():
    print(key, value)
    i = i -1
    if i < 0: break
```

    Networking 2573
    technical 9098
    support 8995
    associate 4214
    long 6930
    applebees 4137
    fucking 5931
    take 9049
    Hows 1965
    queen 8050
    going 6054



```python
# construct the bag-of-word maxtrix using all SMS messages in training set and test set
msg_bow_train = bow_transformer.transform(msg_train)
msg_bow_test = bow_transformer.transform(msg_test)
print("The dimension of bag-of-word matrix for all messages in training set: ", msg_train.shape)
print("The dimension of bag-of-word matrix for all messages in test set: ", msg_test.shape)
```

    The dimension of bag-of-word matrix for all messages in training set:  (4457,)
    The dimension of bag-of-word matrix for all messages in test set:  (1115,)



```python
# let's see what it's doing using one sample message
message2 = msg_train[2]
print("Print one message body: ",message2)
bow2 = bow_transformer.transform([message2])
print("The dimension of bag-of-word vector for No.2 message: ", bow2.shape)
print("Print key-value pairs of BOW for message 2: \n", bow2)
```

    Print one message body:  Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's
    The dimension of bag-of-word vector for No.2 message:  (1, 10077)
    Print key-value pairs of BOW for message 2: 
       (0, 60)	1
      (0, 347)	1
      (0, 354)	1
      (0, 364)	1
      (0, 738)	1
      (0, 1334)	1
      (0, 1577)	2
      (0, 1672)	1
      (0, 2425)	1
      (0, 3449)	1
      (0, 4140)	1
      (0, 4926)	1
      (0, 5558)	2
      (0, 5765)	1
      (0, 8054)	1
      (0, 8090)	1
      (0, 8136)	1
      (0, 9250)	1
      (0, 9406)	1
      (0, 9763)	1
      (0, 9798)	1



```python
# calculate the IDF for traning set and test set
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer_train = TfidfTransformer().fit(msg_bow_train)
tfidf_transformer_test = TfidfTransformer().fit(msg_bow_test)
print("Dimension of inverted document frequency in the traning set: ", tfidf_transformer_train.idf_.shape)
print("Print details: ", tfidf_transformer_train.idf_)
```

    Dimension of inverted document frequency in the traning set:  (10077,)
    Print details:  [ 8.01616115  8.30384323  8.70930833 ...,  8.70930833  7.20523094
      8.70930833]



```python
# now calculate the TF-IDF maxtrix based on the bag-of-word counts (TF) and IDF 
# the dimension should be consistent with messages_bow
# the first index refers to the total number of messages, 
# and the second index the total number of unique words that appear in the sample
msg_tfidf_train = tfidf_transformer_train.transform(msg_bow_train)
msg_tfidf_test = tfidf_transformer_test.transform(msg_bow_test)
print("Dimension of TF-IDF matrix in the traning set: ", msg_tfidf_train.shape)
```

    Dimension of TF-IDF matrix in the traning set:  (4457, 10077)


## Trainning a Spam Classifier


```python
# step 2: fit the model with the training data
from sklearn.naive_bayes import MultinomialNB
spam_classifier = MultinomialNB().fit(msg_tfidf_train, label_train)
```

## Model Evaluation


```python
test_predictions = spam_classifier.predict(msg_tfidf_test)
print(test_predictions)
from sklearn.metrics import classification_report
print (classification_report(label_test, test_predictions))
```

    ['ham' 'ham' 'ham' ..., 'spam' 'ham' 'ham']
                 precision    recall  f1-score   support
    
            ham       0.96      1.00      0.98       957
           spam       1.00      0.75      0.86       158
    
    avg / total       0.97      0.97      0.96      1115
    

