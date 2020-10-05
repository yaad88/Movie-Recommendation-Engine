from keras.datasets import imdb
from keras.preprocessing import sequence
from keras import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout
import numpy as np
np.random.seed(100)

#!/usr/bin/python

import sys

import collections

output=open('output.txt','a')

def lstm(vocabularySize,maxLength,embeddingSize,numOfLayers,batchSize,epoch):
    output.write(":::::::::::::::::::::::LSTM::::::::::::::::::::\n")
    output.write(":::::Input Parameters::::::::\n")
    output.write("Vocabulary Size::"+str(vocabularySize)+"\n")
    output.write("Maximum length of review::" + str(maxLength)+"\n")
    output.write("Embedding Size::" + str(embeddingSize)+"\n")
    output.write("Number of LSTM layers used::" + str(numOfLayers)+"\n")
    output.write("Batch Size::" + str(batchSize)+"\n")
    output.write("Epoch::" + str(epoch)+"\n")
    vocabulary_size = vocabularySize
    INDEX_FROM = 3

    (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocabulary_size)
    print('Loaded dataset with {} training samples, {} test samples'.format(len(X_train), len(X_test)))

    trainCountLabel = collections.Counter(y_train)
    print('Train Label Count')
    print('ZeroCount::', trainCountLabel.get(0), 'OneCount::', trainCountLabel.get(0))

    testCountLabel = collections.Counter(y_test)
    print('Test Label Count')
    print('ZeroCount::', trainCountLabel.get(0), 'OneCount::', trainCountLabel.get(0))

    max_words = maxLength
    X_train = sequence.pad_sequences(X_train, maxlen=max_words)
    X_test = sequence.pad_sequences(X_test, maxlen=max_words)
    embedding_size=embeddingSize
    model=Sequential()
    model.add(Embedding(vocabulary_size, embedding_size, input_length=max_words))
    for i in range(numOfLayers-1):
        model.add(LSTM(100,return_sequences=True))
    model.add(LSTM(100))
    model.add(Dense(1, activation='sigmoid'))

    print(model.summary())

    model.compile(loss='binary_crossentropy',
                 optimizer='adam',
                 metrics=['accuracy'])
    batch_size = batchSize
    num_epochs = epoch

    X_valid, y_valid = X_train[:batch_size], y_train[:batch_size]
    X_train2, y_train2 = X_train[batch_size:], y_train[batch_size:]

    model.fit(X_train2, y_train2, validation_data=(X_valid, y_valid), epochs=num_epochs)


    scores = model.evaluate(X_test, y_test, verbose=0)
    print('Test accuracy:', scores[1])
    output.write("Accuracy on test Data"+str(scores[1])+"\n")
# lstm(vocabularySize=5000,maxLength=500,embeddingSize=32,numOfLayers=2,batchSize=64,epoch=3)

fileName=sys.argv[1]
with open(fileName) as openfileobject:
    ls=[]
    for line in openfileobject:
        ls=line.split()
        print(ls)
        lstm(int(ls[0]),int(ls[1]),int(ls[2]),int(ls[3]),int(ls[4]),int(ls[5]))
output.close()