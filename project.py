from keras.datasets import imdb
import collections
import numpy as np
import pandas as pd
# np.random.rand(100)
# data=open('data.csv','w')
#tokeniZer = Tokenizer(num_words=vocabulary_size)
# tokenizer.fit_on_texts(X_train)
# X=pad_sequences(X_train,maxlen=500)
vocabulary_size = 5000
INDEX_FROM=3

#(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words = None,maxlen=None,index_from=INDEX_FROM)
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words = vocabulary_size)
print('Loaded dataset with {} training samples, {} test samples'.format(len(X_train), len(X_test)))

# word2id = imdb.get_word_index()
# word2id = {k:(v+0) for k,v in word2id.items()}

trainCountLabel=collections.Counter(y_train)
print('Train Label Count')
print('ZeroCount::',trainCountLabel.get(0),'OneCount::',trainCountLabel.get(0))

testCountLabel=collections.Counter(y_test)
print('Test Label Count')
print('ZeroCount::',trainCountLabel.get(0),'OneCount::',trainCountLabel.get(0))



# word2id = imdb.get_word_index()
# # word2id = {k:(v+INDEX_FROM) for k,v in word2id.items()}
# # word2id["<PAD>"] = 0
# # word2id["<START>"] = 1
# # word2id["<UNK>"] = 2
# id2word = {i: word for word, i in word2id.items()}
# print('---review with words---')
# # for x in range(len(X_train)):
# #     for i in X_train[x]:
# #         # p=id2word.get(i)
# #         if(str(id2word.get(i))!="<START>"):
# #             data.write(str(id2word.get(i))+' ')
# #     data.write(str(','))
# #     data.write(str(y_train[x])+'\n')
# # data.close()
# print([id2word.get(i, ' ') for i in X_train[6]])
# print('---label---')
# print(y_train[6])
#
#
# print('Maximum review length: {}'.format(
# len(max((X_train + X_test), key=len))))
#
# print('Minimum review length: {}'.format(
# len(min((X_test + X_test), key=len))))


#preprocessing for svn and regression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('data.csv', header=None)
dfTest = pd.read_csv('dataTest.csv', header=None)
reviews_train = []
reviews_test = []
trainTarget = []
testTarget = []
for i in range(df.__len__()):
    reviews_train.append(df.iloc[i, 0].strip())
    # print(reviews_train[i])
    trainTarget.append(df.iat[i, 1])

for i in range(dfTest.__len__()):
    reviews_test.append(dfTest.iloc[i, 0].strip())
    testTarget.append(dfTest.iat[i, 1])

stop_words = ['in', 'of', 'at', 'a', 'the']
ngram_vectorizer = CountVectorizer(binary=False, ngram_range=(1, 1), stop_words=stop_words)
ngram_vectorizer.fit(reviews_train)
X = ngram_vectorizer.transform(reviews_train)
print(X.shape)
X_test2 = ngram_vectorizer.transform(reviews_test)

X_train3, X_val3, y_train3, y_val3 = train_test_split(
    X, trainTarget, train_size = 0.75
)



from keras.preprocessing import sequence

# train = train.sample(frac=1).reset_index(drop=True)
# train['Phrase'] = train['Phrase'].apply(lambda x: x.lower())
# test['Phrase'] = test['Phrase'].apply(lambda x: x.lower())
#
# X = train['Phrase']
# test_X = test['Phrase']
# Y = to_categorical(train['Sentiment'].values)
#
# tokenizer = Tokenizer(num_words=max_features)
# tokenizer.fit_on_texts(list(X))
#
# X = tokenizer.texts_to_sequences(X)

max_words = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_words)
X_test = sequence.pad_sequences(X_test, maxlen=max_words)

from keras import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout

def lstm():
    embedding_size=32
    model=Sequential()
    model.add(Embedding(vocabulary_size, embedding_size, input_length=max_words))
    model.add(LSTM(100,return_sequences=True))
    model.add(LSTM(100))
    model.add(Dense(1, activation='sigmoid'))

    print(model.summary())

    model.compile(loss='binary_crossentropy',
                 optimizer='adam',
                 metrics=['accuracy'])
    batch_size = 64
    num_epochs = 3

    X_valid, y_valid = X_train[:batch_size], y_train[:batch_size]
    X_train2, y_train2 = X_train[batch_size:], y_train[batch_size:]

    # model.fit(X_train2, y_train2, validation_data=(X_valid, y_valid), batch_size=batch_size, epochs=num_epochs)

    model.fit(X_train3, y_train3, validation_data=(X_val3, y_val3), batch_size=batch_size, epochs=num_epochs)


    scores = model.evaluate(X_test, y_test, verbose=0)
    print('Test accuracy:', scores[1])


def regression():

    bestC=0
    bestAcc=-100000
    for c in [0.01, 0.05, 0.25, 0.5, 1]:
        lr = LogisticRegression(C=c)
        lr.fit(X_train3, y_train3)
        acc=accuracy_score(y_val3, lr.predict(X_val3))
        print("Accuracy for C=%s: %s"%(c,acc))
        if(acc>bestAcc):
            bestC=c
            bestAcc=acc
        # print("Accuracy for C=%s: %s"
        #       % (c, accuracy_score(y_valid, lr.predict(X_valid))))

    final_model = LogisticRegression(C=bestC)
    final_model.fit(X, trainTarget)
    print ("Final Accuracy: %s"
           % accuracy_score(testTarget, final_model.predict(X_test2)))
# Final Accuracy: 0.88128
from sklearn.svm import LinearSVC

def svn():


    bestC=0
    bestAcc=-100000
    for c in [0.01, 0.05, 0.25, 0.5, 1]:
        svm = LinearSVC(C=c)
        svm.fit(X_train3, y_train3)
        acc = accuracy_score(y_val3, svm.predict(X_val3))
        print("Accuracy for C=%s: %s"% (c, acc))
        if (acc > bestAcc):
            bestC = c
            bestAcc=acc

    # Accuracy for C=0.01: 0.89104
    # Accuracy for C=0.05: 0.88736
    # Accuracy for C=0.25: 0.8856
    # Accuracy for C=0.5: 0.88608
    # Accuracy for C=1: 0.88592

    final_svm_ngram = LinearSVC(C=bestC)
    final_svm_ngram.fit(X, trainTarget)
    print("Final Accuracy: %s"
          % accuracy_score(testTarget, final_svm_ngram.predict(X_test2)))

modelNo = input("Press 0 for lstm, 1 for svn and 2 for regression\n")
if(modelNo=='0'):
    lstm()
elif (modelNo=='1'):
    svn()
else:
    print("linear")
    regression()