import collections
import numpy as np
import pandas as pd
#!/usr/bin/python
np.random.seed(100)

import sys

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

output=open('output.txt','a')

def svn(lowNgram,highNgram,listOfC):
    output.write(":::::::::::::::::::::::SVM::::::::::::::::::::\n")
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
    ngram_vectorizer = CountVectorizer(binary=False, ngram_range=(lowNgram, highNgram), stop_words=stop_words)
    ngram_vectorizer.fit(reviews_train)
    X = ngram_vectorizer.transform(reviews_train)
    X_test2 = ngram_vectorizer.transform(reviews_test)

    X_train3, X_val3, y_train3, y_val3 = train_test_split(
        X, trainTarget, train_size=0.75
    )
    print("Pre-Process Done")
    bestC=0
    bestAcc=-100000
    for c in listOfC:
        svm = LinearSVC(C=c)
        svm.fit(X_train3, y_train3)
        acc = accuracy_score(y_val3, svm.predict(X_val3))
        output.write("vallue of C::" + str(c) + " " + str(acc) + "\n")
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
    tacc=accuracy_score(testTarget, final_svm_ngram.predict(X_test2))
    output.write("Best C:: "+str(bestC)+"Final Accuracy on Test data:: "+str(tacc)+"\n")
    print("Final Accuracy: %s"
          % tacc)


fileName=sys.argv[1]
with open(fileName) as openfileobject:
    ls=[]
    for line in openfileobject:
        ls=line.split()
        print(ls)
        cValues=[]
        for i in range(len(ls)-2):
            cValues.append(float(ls[i+2]))
        print(cValues)
        svn(int(ls[0]),int(ls[1]),cValues)
output.close()
