### IMPORTING LIBRARIES
import nltk
import re
import csv
import math
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import OrderedDict
import numpy as np
import operator
from flask import Flask, redirect, render_template, url_for, request
from flask_wtf import FlaskForm
from wtforms import StringField, IntegerField, SubmitField




########################################################
######### VECTOR SPACE MODEL ###########################
########################################################



### FOR LAMMETIZATION
wordnet_lemmatizer = WordNetLemmatizer()


### READING STOP WORDS
stop_words = []
f = open("CS317-IR Dataset for A1/Stopword-List.txt", "r")
txt = f.read().replace('\n', ' ')
f.close()
stop_words = txt.split()


### FUNCTION FOR CONTRACTIONS
def decontract_words(text):

    ## For general words
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'s", " is", text)
    text = re.sub(r"\'t", " not", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'m", " am", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"n\'t", " not", text)
    text = re.sub(r"[0-9] +", "", text)
    text = re.sub(r"[^\w\s]", " ", text)


    ## For specific words
    text = re.sub(r"won\'t", "will not", text)
    text = re.sub(r"can\'t", "can not", text)

    return text


### PREPROCESSING OF DOCUMENTS AND QUERY
def Preprocessing(text):

    text = decontract_words(text)

    tokens = nltk.word_tokenize(text)

    tokens = list(set(tokens))

    ## Removing special characters
    removetable = str.maketrans("", "", "'!@#$%^&*()_=-\|][:';:,<.>/?`~")
    tokens=[x.translate(removetable) for x in tokens]

    ## Converting to lower case
    tokens = [t.lower() for t in tokens]
    
    ## Removing stopwords
    tokens = [t for t in tokens if t.isalnum() and t not in stop_words]

    ## Lemmatization
    lemmatized = []
    for word in tokens:
        lemmatized.append(wordnet_lemmatizer.lemmatize(word))

    return lemmatized





idfDict = {}
'''
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
'''





## Getting tokens
f = open('keys.txt', 'r')
tokens = list(f.read().split(' '))
f.close()


## Fetching keys
f = open('keys.txt', 'r')
keys = list(f.read().split(' '))
f.close()


## Fetching values
f = open('values.txt')
values = list(f.read().split(' '))
f.close()


## Fetching tfidf
tfidf = {}

f = open('fileno1.txt', 'r')
k = list(f.read().split(' '))


for i in range(1, 51):
    f = open('val' + str(i) + '.txt', 'r')
    v = list(f.read().split(' '))
    tfidf[i] = {}
    for index, word in enumerate(k):
        tfidf[i][k[index]] = float(v[index])


for index, key in enumerate(keys):
    idfDict[key] = float(values[index])




### USER QUERY
# query = input("Enter query here: ")
# query = Preprocessing(query)

# qtV = dict.fromkeys(tokens, 0)
# for word in query:
#     if word in tokens:
#         qtV[word] += 1


# ## idf
# for word in qtV:
#     qtV[word] = qtV[word]*idfDict[word]


# ## Finding cosine similarity
# res = {}
# temp = 0

# vec1  = np.array([list(qtV.values())])
# for x in range(1, 51):
#     vec2 = np.array([list(tfidf[x].values())])
#     if cosine_similarity(vec1, vec2) > 0:
#         temp = cosine_similarity(vec1, vec2)[0][0]
#         res[x] = temp


# Results = []
# results = (sorted(res.items(), key=operator.itemgetter(1), reverse=True))
# for res, fre in results:
#     print(res)
#     Results.append(res)





### USER QUERY
# query = input("Enter query here: ")
# query = Preprocessing(query)
query = []



########################################################
############ UI USING FLASK ############################
########################################################


class Form(FlaskForm):
    query = StringField()
    submit = SubmitField("Search")


app = Flask(__name__)
app.config['SECRET_KEY'] = 'noureen'



@app.route('/', methods=['GET', 'POST'])
def index():

    form = Form()

    if form.validate_on_submit():
        global query
        query = request.form.get("input")
        return redirect(url_for("result"))

    return render_template("base.html", form=form)    


@app.route('/Results')
def result():


    ### USER QUERY
    global query
    query = Preprocessing(query)
    qtV = dict.fromkeys(tokens, 0)
    for word in query:
        if word in tokens:
            qtV[word] += 1


    ## idf
    for word in qtV:
        qtV[word] = qtV[word]*idfDict[word]


    ## Finding cosine similarity
    res = {}
    temp = 0

    vec1  = np.array([list(qtV.values())])
    for x in range(1, 51):
        vec2 = np.array([list(tfidf[x].values())])
        if cosine_similarity(vec1, vec2) > 0:
            temp = cosine_similarity(vec1, vec2)[0][0]
            res[x] = temp


    Results = []
    results = (sorted(res.items(), key=operator.itemgetter(1), reverse=True))
    for res, fre in results:
        Results.append(res)

    Results = sorted(Results)

    return render_template("result.html", result=Results, length=len(Results))


if __name__ == "__main__":
    app.run(debug=True)
