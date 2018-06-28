import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import re
import nltk

with open('glove.6B.50d.txt') as f:
    text = [ a.strip('\n').split(' ') for a in f.readlines()]
    
transcript = pd.read_csv('ted-talks/transcripts.csv')


def unique_words(text):
    total_word = []
    for talk in text:
        x = str(talk)
        x=re.sub("[\(\[].*?[\)\]]", "", x)
        x = re.sub("[^a-zA-Z]", " ", x)    
        x = x.lower().split()
        total_word.extend(x)
    return set(total_word)





def build_dict(t):
    word_score_dict = {}
    for item in t:
        if item[0] in all_words:
            array = np.array([float(i) for i in item[1:]])
            word_score_dict[item[0]] = array
    return word_score_dict


all_words = unique_words(transcript.transcript)
word_score = build_dict(text)
whole_array = np.array(list(word_score.values()))

print ('---building kmeans')

kmeans = KMeans(n_clusters=80, random_state=0)
kmeans.fit(whole_array)
words_class = kmeans.predict(whole_array)


def word_classifier(keys, classes):
    word_class = {}
    for key, value in zip(keys, classes):
        word_class[key] = value
    return word_class

print ('---mapping words class')
word_class_dict = word_classifier(word_score.keys(), words_class)

print ('---creating dataframe and saving')
df = pd.DataFrame({'keys':list(word_class_dict.keys()), \
                   'values':list(word_class_dict.values())})
df.to_csv('mapping.csv')
        
        
        
        
        
        
        
        