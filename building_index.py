### IMPORTING LIBRARIES

import nltk
import csv
import math
from sklearn.metrics.pairwise import cosine_similarity
from collections import OrderedDict
import numpy as np
import operator
from processing import Preprocessing


idfDict = {}

### CONVERTING DOCUMENTS INTO VECTORS
def index():

    D = {}

    for doc in range(50):
        doc = doc + 1
        mydoc = 'CS317-IR Dataset for A1/ShortStories/' + str(doc) + '.txt'
        D[doc] = open(mydoc).read()

    ## Concatinating all documents' content
    allDoc = ""
    for doc in range(50):
        doc = doc + 1
        allDoc = allDoc + " \n" + D[doc]


    tokens = []

    docToken = {}
    for doc in range(50):
        doc = doc + 1
        docToken[doc] = Preprocessing(D[doc])

    for doc in docToken:
        for word in docToken[doc]:
            tokens.append(word)

    tokens = sorted(list(set(tokens)))


    ### FINDING TF-IDF
    docV = {}

    for doc in range(1, 51):
        docV[doc] = dict.fromkeys(tokens, 0)

    ## Finding term frequency in documents (tf)
    for doc in range(1, 51):
        for word in docToken[doc]:
            docV[doc][word] += 1

    
    tfDocV = {}
    for x in range(1, 51):
        tfDocV[x] = {}
        for word, count in docV[x].items():
            tfDocV[x][word] = count
    
    ## Finding inverse document frequency (idf)
    for x in range(1, 51):
        docToken[x] = set(docToken[x])
        docToken[x] = list(set(docToken[x]))


    wordDcount = dict.fromkeys(tokens, 0)
    for word in tokens:
        for x in range(1, 51):
            if word in docToken[x]:
                wordDcount[word] += 1

    
    global idfDict
    for word in tokens:
        if wordDcount[word] > 0:
            count = wordDcount[word]
            if count > 50:
                count = 50
        idfDict[word] = math.log(50/count)

    

    ## Finding tfidf
    tfidf = {}
    for x in range(1, 51):
        tfidf[x] = {}
        for word in docV[x]:
            tfidf[x][word] = tfDocV[x][word]*idfDict[word]



    # print(tfidf[37]['crowd'])


    values2 = []   

    values = list(idfDict.values())
    
    for val in values:
        values2.append(str(val))


    ## Storing values in file
    values2 = ' '.join(values2)
    f = open('values.txt', 'w')
    f.write(values2)
    f.close()



    ## Storing keys in file
    keys = idfDict.keys()
    keys = ' '.join(keys)
    f = open('keys.txt', 'w')
    f.write(keys)
    f.close()



    for i in range(1,2):
        k = tfidf[i].keys()
        f = open('fileno' + str(i) + '.txt', 'w')
        k = ' '.join(k)
        f.write(k)

    
    for i in range(1, 51):
        v = list(tfidf[i].values())
        f = open('val' + str(i) + '.txt', 'w')
        v2 = [str(val) for val in v]
        v2 = ' '.join(v2)
        f.write(v2)

    

index()