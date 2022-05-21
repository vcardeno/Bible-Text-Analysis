# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, OrderedDict

from nltk.probability import FreqDist
from nltk.corpus import stopwords
import nltk
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
#nltk.download('stopwords')

import gensim
from gensim.utils import simple_preprocess

from textblob import TextBlob
from textblob import Word

import sklearn

import platform

import re

platform.architecture()


#import data
data = pd.read_csv("C:/Users/victo/Documents/Analytics Projects/Bible Text Analysis/Data/t_web.csv")
key_english = pd.read_csv("C:/Users/victo/Documents/Analytics Projects/Bible Text Analysis/Data/key_english.csv")

#check data
data.head()
key_english.head()

#merge with key
data = data.merge(key_english, on = 'b')

data.columns

#get word count
data['word_count'] = data['t_x'].apply(lambda x: len(str(x).split(" ")))
data[['t_x','word_count']].head()

#get word count distribution of verses by book

books = data['n'].unique()

for elem in books:
    print(elem)
    temp = data.loc[data['n'] == elem]
    sns.boxplot(x=temp['n'], y=temp['word_count'])
    plt.show()

#get word count distribution of verses by section
sections = data['g'].unique()

for elem in sections:
    print(elem)
    temp = data.loc[data['g'] == elem]
    sns.boxplot(x=temp['g'], y=temp['word_count'])
    plt.show()
    

    
#Pre-processing

#remove footnotes
data['t_x'] = data['t_x'].str.replace(r"\{.*\}","")

#turn all words lowercase
data['t_x'] = data['t_x'].apply(lambda x: " ".join(x.lower() for x in x.split()))
data['t_x'].head()

#remove punctuation
data['t_x'] = data['t_x'].str.replace('[^\w\s]','')
data['t_x'].head()

#lemmatize
data['t_x'] = data['t_x'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

#change for respect 
data['t_x'] = data['t_x'].replace('yahweh', 'adonai', regex=True)


#most common words in Bible
all_words = ''.join(data.t_x)
tokenizer = nltk.RegexpTokenizer(r"\w+")
all_words = tokenizer.tokenize(all_words)
counter = Counter(all_words)

#ten most common words
top10 = OrderedDict(counter.most_common(10))
names = list(top10.keys())
values = list(top10.values())

plt.bar(range(len(top10)), values, tick_label=names)
plt.show()

#25 most common words
top25 = OrderedDict(counter.most_common(25))
names = list(top25.keys())
values = list(top25.values())

plt.bar(range(len(top25)), values, tick_label=names)
plt.show()

#get rid of stop words

#remove stopwords
stop = stopwords.words('english')
manual_stop = ['ye', 'unto', 'thou', 'thy', 'thee', 'upon', 'shalt', 'shall', 'hath', 'wa', 'u', 'also', 'therefore', 'ha', 'one', 'dont', 'may']
combined = stop + manual_stop
data['t_x'] = data['t_x'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
data['t_x'] = data['t_x'].apply(lambda x: " ".join(x for x in x.split() if x not in manual_stop))
data['t_x'].head()

#Separating Old Testament and new Testament
ot = data.loc[data['t_y'] == 'OT']
nt = data.loc[data['t_y'] == 'NT']


#most common words in Bible
all_words = ''.join(nt.t_x)
tokenizer = nltk.RegexpTokenizer(r"\w+")
all_words = tokenizer.tokenize(all_words)
counter = Counter(all_words)

#ten most common words
top10 = OrderedDict(counter.most_common(10))
names = list(top10.keys())
values = list(top10.values())

plt.bar(range(len(top10)), values, tick_label=names)
plt.show()

#20 most common words
top20 = OrderedDict(counter.most_common(20))
names = list(top20.keys())
values = list(top20.values())

plt.barh(range(len(top20)), values, tick_label=names)
plt.xlabel('Number of Times Word Appears')
plt.ylabel('Word or Stem')
plt.gca().invert_yaxis()

plt.show()



#split into sections
section_1 = data.loc[data['g'] == 1]
section_2 = data.loc[data['g'] == 2]
section_3 = data.loc[data['g'] == 3]
section_4 = data.loc[data['g'] == 4]
section_5 = data.loc[data['g'] == 5]
section_6 = data.loc[data['g'] == 6]
section_7 = data.loc[data['g'] == 7]
section_8 = data.loc[data['g'] == 8]


#bigrams and trigrams 
from sklearn.feature_extraction.text import CountVectorizer
c_vec = CountVectorizer(stop_words=combined, ngram_range=(2,3))
# matrix of ngrams
ngrams = c_vec.fit_transform(section_7['t_x'])
# count frequency of ngrams
count_values = ngrams.toarray().sum(axis=0)
# list of ngrams
vocab = c_vec.vocabulary_
df_ngram = pd.DataFrame(sorted([(count_values[i],k) for k,i in vocab.items()], reverse=True)
            ).rename(columns={0: 'frequency', 1:'bigram/trigram'})

first_20 = df_ngram.head(20)

plt.barh(range(len(first_20)), first_20['frequency'], tick_label = first_20['bigram/trigram'])
plt.xlabel("Number of Apperances")
plt.ylabel("Bigram / Trigram")
plt.gca().invert_yaxis()
plt.show()


#can we do wordclouds by section?

section_1_words = ' '.join(section_1.t_x)

section_1_wordcloud = WordCloud(collocations=False).generate(section_1_words)
plt.imshow(section_1_wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

section_2_words = ' '.join(section_2.t_x)

section_2_wordcloud = WordCloud(collocations=False).generate(section_2_words)
plt.imshow(section_2_wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

section_3_words = ' '.join(section_3.t_x)

section_3_wordcloud = WordCloud(collocations=False).generate(section_3_words)
plt.imshow(section_3_wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

section_4_words = ' '.join(section_4.t_x)

section_4_wordcloud = WordCloud(collocations=False).generate(section_4_words)
plt.imshow(section_4_wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

section_5_words = ' '.join(section_5.t_x)

section_5_wordcloud = WordCloud(collocations=False).generate(section_5_words)
plt.imshow(section_5_wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

section_6_words = ' '.join(section_6.t_x)

section_6_wordcloud = WordCloud(collocations=False).generate(section_6_words)
plt.imshow(section_6_wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

section_7_words = ' '.join(section_7.t_x)

section_7_wordcloud = WordCloud(collocations=False).generate(section_7_words)
plt.imshow(section_7_wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

section_8_words = ' '.join(section_8.t_x)

section_8_wordcloud = WordCloud(collocations=False).generate(section_8_words)
plt.imshow(section_8_wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# Topic Analysis with LDA 
section_1_list = section_1_words.split(" ")
section_2_list = section_2_words.split(" ")
section_3_list = section_3_words.split(" ")
section_4_list = section_4_words.split(" ")
section_5_list = section_5_words.split(" ")
section_6_list = section_6_words.split(" ")
section_7_list = section_7_words.split(" ")
section_8_list = section_8_words.split(" ")

#all
all = []
all.append(section_1_list)
all.append(section_2_list)
all.append(section_3_list)
all.append(section_4_list)
all.append(section_5_list)
all.append(section_6_list)
all.append(section_7_list)
all.append(section_8_list)


import gensim.corpora as corpora
# Create Dictionary
id2word = corpora.Dictionary(all)
# Create Corpus
texts = all
# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]
# View
print(corpus[:1][0][:30])


from pprint import pprint
# number of topics
num_topics = 4
# Build LDA model
lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                       id2word=id2word,
                                       num_topics=num_topics,
                                       passes = 50)
# Print the Keyword in the 10 topics
#pprint(lda_model.print_topics())
doc_lda = lda_model[corpus]


print(lda_model.print_topics(num_topics = 4, num_words = 10))

count = 0
for i in lda_model[corpus]:
    print("doc : ", count, i)
    count += 1
    
from gensim.test.utils import datapath

# Save model to disk.
temp_file = datapath("model")
lda_model.save(temp_file)


#Sentiment Analysis 
blob1 = TextBlob(section_1_words)

blob1.tags

blob1.sentiment




blob2 = TextBlob(section_2_words)

blob2.tags

blob2.sentiment


blob3 = TextBlob(section_3_words)

blob3.sentiment


blob4 = TextBlob(section_4_words)

blob4.sentiment


blob5 = TextBlob(section_5_words)

blob5.sentiment


blob6 = TextBlob(section_6_words)

blob6.sentiment



blob7 = TextBlob(section_7_words)

blob7.sentiment



blob8 = TextBlob(section_8_words)

blob8.sentiment



#let's go within sections now

#Section 3

#can we do wordclouds by book?
job = data.loc[data['n'] == 'Job']
psalms = data.loc[data['n'] == 'Psalms']
eccl = data.loc[data['n'] == 'Ecclesiastes']
proverbs = data.loc[data['n'] == 'Proverbs']
songs = data.loc[data['n'] == 'Song of Solomon']


job_words = ' '.join(job.t_x)

job_wordcloud = WordCloud(collocations=False).generate(job_words)
plt.imshow(job_wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

psalms_words = ' '.join(psalms.t_x)

psalms_wordcloud = WordCloud(collocations=False).generate(psalms_words)
plt.imshow(psalms_wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

eccl_words = ' '.join(eccl.t_x)

eccl_wordcloud = WordCloud(collocations=False).generate(eccl_words)
plt.imshow(eccl_wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

proverbs_words = ' '.join(proverbs.t_x)

proverbs_wordcloud = WordCloud(collocations=False).generate(proverbs_words)
plt.imshow(proverbs_wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

songs_words = ' '.join(songs.t_x)

songs_wordcloud = WordCloud(collocations=False).generate(songs_words)
plt.imshow(songs_wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()





# Topic Analysis across Books 
job_list = job_words.split(" ")
psalms_list = psalms_words.split(" ")
eccl_list = eccl_words.split(" ")
proverbs_list = proverbs_words.split(" ")
songs_list = songs_words.split(" ")


#all
three = []
three.append(job_list)
three.append(psalms_list)
three.append(eccl_list)
three.append(proverbs_list)
three.append(songs_list)



import gensim.corpora as corpora
# Create Dictionary
id2word = corpora.Dictionary(three)
# Create Corpus
texts = three
# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]
# View
print(corpus[:1][0][:30])


from pprint import pprint
# number of topics
num_topics = 5
# Build LDA model
lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                       id2word=id2word,
                                       num_topics=num_topics)
# Print the Keyword in the 5 topics
pprint(lda_model.print_topics())
doc_lda = lda_model[corpus]


print(lda_model.print_topics(num_topics = 5, num_words = 5))

count = 0
for i in lda_model[corpus]:
    print("doc : ", count, i)
    count += 1
    
    
    
#Sentiment Analysis 
blob1 = TextBlob(job_words)

blob1.tags

blob1.sentiment




blob2 = TextBlob(psalms_words)

blob2.tags

blob2.sentiment


blob3 = TextBlob(eccl_words)

blob3.sentiment


blob4 = TextBlob(proverbs_words)

blob4.sentiment


blob5 = TextBlob(songs_words)

blob5.sentiment





#Section 4 - Prophets
isaiah = data.loc[data['n'] == 'Isaiah']
jeremiah = data.loc[data['n'] == 'Jeremiah']
lament = data.loc[data['n'] == 'Lamentations']
ezekiel = data.loc[data['n'] == 'Ezekiel']
daniel = data.loc[data['n'] == 'Daniel']
hosea = data.loc[data['n'] == 'Hosea']
joel = data.loc[data['n'] == 'Joel']
amos = data.loc[data['n'] == 'Amos']
obadiah = data.loc[data['n'] == 'Obadiah']
jonah = data.loc[data['n'] == 'Jonah']
micah = data.loc[data['n'] == 'Micah']
nahum = data.loc[data['n'] == 'Nahum']
habakkuk = data.loc[data['n'] == 'Habakkuk']
zephaniah = data.loc[data['n'] == 'Zephaniah']
haggai = data.loc[data['n'] == 'Haggai']
zechariah = data.loc[data['n'] == 'Zechariah']
malachi = data.loc[data['n'] == 'Malachi']


#Word Clouds

isaiah_words = ' '.join(isaiah.t_x)

isaiah_wordcloud = WordCloud(collocations=False).generate(isaiah_words)
plt.imshow(isaiah_wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

jeremiah_words = ' '.join(jeremiah.t_x)

jeremiah_wordcloud = WordCloud(collocations=False).generate(jeremiah_words)
plt.imshow(jeremiah_wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

lament_words = ' '.join(lament.t_x)

lament_wordcloud = WordCloud(collocations=False).generate(lament_words)
plt.imshow(lament_wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

ezekiel_words = ' '.join(ezekiel.t_x)

ezekiel_wordcloud = WordCloud(collocations=False).generate(ezekiel_words)
plt.imshow(ezekiel_wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

daniel_words = ' '.join(daniel.t_x)

daniel_wordcloud = WordCloud(collocations=False).generate(daniel_words)
plt.imshow(daniel_wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

hosea_words = ' '.join(hosea.t_x)

hosea_wordcloud = WordCloud(collocations=False).generate(hosea_words)
plt.imshow(hosea_wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

joel_words = ' '.join(joel.t_x)

joel_wordcloud = WordCloud(collocations=False).generate(joel_words)
plt.imshow(joel_wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

amos_words = ' '.join(amos.t_x)

amos_wordcloud = WordCloud(collocations=False).generate(amos_words)
plt.imshow(amos_wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

obadiah_words = ' '.join(obadiah.t_x)

obadiah_wordcloud = WordCloud(collocations=False).generate(obadiah_words)
plt.imshow(obadiah_wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

jonah_words = ' '.join(jonah.t_x)

jonah_wordcloud = WordCloud(collocations=False).generate(jonah_words)
plt.imshow(jonah_wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

micah_words = ' '.join(micah.t_x)

micah_wordcloud = WordCloud(collocations=False).generate(micah_words)
plt.imshow(micah_wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

nahum_words = ' '.join(nahum.t_x)

nahum_wordcloud = WordCloud(collocations=False).generate(nahum_words)
plt.imshow(nahum_wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

habakkuk_words = ' '.join(habakkuk.t_x)

habakkuk_wordcloud = WordCloud(collocations=False).generate(habakkuk_words)
plt.imshow(habakkuk_wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

zephaniah_words = ' '.join(zephaniah.t_x)

zephaniah_wordcloud = WordCloud(collocations=False).generate(zephaniah_words)
plt.imshow(zephaniah_wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

haggai_words = ' '.join(haggai.t_x)

haggai_wordcloud = WordCloud(collocations=False).generate(haggai_words)
plt.imshow(haggai_wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

zechariah_words = ' '.join(zechariah.t_x)

zechariah_wordcloud = WordCloud(collocations=False).generate(zechariah_words)
plt.imshow(zechariah_wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

malachi_words = ' '.join(malachi.t_x)

malachi_wordcloud = WordCloud(collocations=False).generate(malachi_words)
plt.imshow(malachi_wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()



# Topic Analysis across Books 
isaiah_list = isaiah_words.split(" ")
jeremiah_list = jeremiah_words.split(" ")
lament_list = lament_words.split(" ")
ezekiel_list = ezekiel_words.split(" ")
daniel_list = daniel_words.split(" ")
hosea_list = hosea_words.split(" ")
joel_list = joel_words.split(" ")
amos_list = amos_words.split(" ")
obadiah_list = obadiah_words.split(" ")
jonah_list = jonah_words.split(" ")
micah_list = micah_words.split(" ")
nahum_list = nahum_words.split(" ")
habakkuk_list = habakkuk_words.split(" ")
zephaniah_list = zephaniah_words.split(" ")
haggai_list = haggai_words.split(" ")
zechariah_list = zechariah_words.split(" ")
malachi_list = malachi_words.split(" ")


#all
four = []
four.append(isaiah_list)
four.append(jeremiah_list)
four.append(lament_list)
four.append(ezekiel_list)
four.append(daniel_list)
four.append(hosea_list)
four.append(joel_list)
four.append(amos_list)
four.append(obadiah_list)
four.append(jonah_list)
four.append(micah_list)
four.append(nahum_list)
four.append(habakkuk_list)
four.append(zephaniah_list)
four.append(haggai_list)
four.append(zechariah_list)
four.append(malachi_list)




import gensim.corpora as corpora
# Create Dictionary
id2word = corpora.Dictionary(four)
# Create Corpus
texts = four
# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]
# View
print(corpus[:1][0][:30])


from pprint import pprint
# number of topics
num_topics = 5
# Build LDA model
lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                       id2word=id2word,
                                       num_topics=num_topics,
                                       passes = 20)
# Print the Keyword in the 5 topics
pprint(lda_model.print_topics())
doc_lda = lda_model[corpus]



print(lda_model.print_topics(num_topics = 5, num_words = 10))

count = 0
for i in lda_model[corpus]:
    print("doc : ", count, i)
    count += 1
    
    
    
#Sentiment Analysis 
blob1 = TextBlob(isaiah_words)

blob1.tags

blob1.sentiment




blob2 = TextBlob(jeremiah_words)

blob2.tags

blob2.sentiment


blob3 = TextBlob(lament_words)

blob3.sentiment


blob4 = TextBlob(ezekiel_words)

blob4.sentiment


blob5 = TextBlob(daniel_words)

blob5.sentiment



blob6 = TextBlob(hosea_words)

blob6.sentiment


blob7 = TextBlob(amos_words)

blob7.sentiment



blob8 = TextBlob(joel_words)

blob8.sentiment



blob9 = TextBlob(obadiah_words)

blob9.sentiment


blob10 = TextBlob(jonah_words)

blob10.sentiment


blob11 = TextBlob(micah_words)

blob11.sentiment


blob12 = TextBlob(nahum_words)

blob12.sentiment


blob13 = TextBlob(habakkuk_words)

blob13.sentiment


blob14 = TextBlob(zephaniah_words)

blob14.sentiment


blob15 = TextBlob(haggai_words)

blob15.sentiment


blob16 = TextBlob(zechariah_words)

blob16.sentiment


blob17 = TextBlob(malachi_words)

blob17.sentiment





#now let's look at the gospels
matthew = data.loc[data['n'] == 'Matthew']
mark = data.loc[data['n'] == 'Mark']
luke = data.loc[data['n'] == 'Luke']
john = data.loc[data['n'] == 'John']

matthew_words = ' '.join(matthew.t_x)

matthew_wordcloud = WordCloud(collocations=False).generate(matthew_words)
plt.imshow(matthew_wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

mark_words = ' '.join(mark.t_x)

mark_wordcloud = WordCloud(collocations=False).generate(mark_words)
plt.imshow(mark_wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

luke_words = ' '.join(luke.t_x)

luke_wordcloud = WordCloud(collocations=False).generate(luke_words)
plt.imshow(luke_wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

john_words = ' '.join(john.t_x)

john_wordcloud = WordCloud(collocations=False).generate(john_words)
plt.imshow(john_wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

#Graphs of word counts
#Matthew
tokenizer = nltk.RegexpTokenizer(r"\w+")
matthew_words = tokenizer.tokenize(matthew_words)
counter = Counter(matthew_words)

#ten most common words
top10 = OrderedDict(counter.most_common(10))
names = list(top10.keys())
values = list(top10.values())

plt.barh(range(len(top10)), values, tick_label=names)
plt.xlabel('Number of Times Word Appears')
plt.ylabel('Word or Stem')
plt.gca().invert_yaxis()

plt.show()


#Mark
tokenizer = nltk.RegexpTokenizer(r"\w+")
mark_words = tokenizer.tokenize(mark_words)
counter = Counter(mark_words)

#ten most common words
top10 = OrderedDict(counter.most_common(10))
names = list(top10.keys())
values = list(top10.values())

plt.barh(range(len(top10)), values, tick_label=names)
plt.xlabel('Number of Times Word Appears')
plt.ylabel('Word or Stem')
plt.gca().invert_yaxis()

plt.show()

#Luke
tokenizer = nltk.RegexpTokenizer(r"\w+")
luke_words = tokenizer.tokenize(luke_words)
counter = Counter(luke_words)

#ten most common words
top10 = OrderedDict(counter.most_common(10))
names = list(top10.keys())
values = list(top10.values())

plt.barh(range(len(top10)), values, tick_label=names)
plt.xlabel('Number of Times Word Appears')
plt.ylabel('Word or Stem')
plt.gca().invert_yaxis()

plt.show()

#John
tokenizer = nltk.RegexpTokenizer(r"\w+")
john_words = tokenizer.tokenize(john_words)
counter = Counter(john_words)

#ten most common words
top10 = OrderedDict(counter.most_common(10))
names = list(top10.keys())
values = list(top10.values())

plt.barh(range(len(top10)), values, tick_label=names)
plt.xlabel('Number of Times Word Appears')
plt.ylabel('Word or Stem')
plt.gca().invert_yaxis()

plt.show()




# Topic Analysis across Books 
matthew = data.loc[data['n'] == 'Matthew']
mark = data.loc[data['n'] == 'Mark']
luke = data.loc[data['n'] == 'Luke']
john = data.loc[data['n'] == 'John']

matthew_words = ' '.join(matthew.t_x)
mark_words = ' '.join(mark.t_x)
luke_words = ' '.join(luke.t_x)
john_words = ' '.join(john.t_x)


matthew_list = matthew_words.split(" ")
mark_list = mark_words.split(" ")
luke_list = luke_words.split(" ")
john_list = john_words.split(" ")



#all
five = []
five.append(matthew_list)
five.append(mark_list)
five.append(luke_list)
five.append(john_list)




import gensim.corpora as corpora
# Create Dictionary
id2word = corpora.Dictionary(five)
# Create Corpus
texts = five
# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]
# View
print(corpus[:1][0][:30])


from pprint import pprint
# number of topics
num_topics = 2
# Build LDA model
lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                       id2word=id2word,
                                       num_topics=num_topics,
                                       passes = 20)
# Print the Keyword in the 5 topics
pprint(lda_model.print_topics())
doc_lda = lda_model[corpus]


print(lda_model.print_topics(num_topics = 2, num_words = 5))

count = 0
for i in lda_model[corpus]:
    print("doc : ", count, i)
    count += 1
    
    
    
#Sentiment Analysis 
blob1 = TextBlob(matthew_words)

blob1.tags

blob1.sentiment




blob2 = TextBlob(mark_words)

blob2.tags

blob2.sentiment


blob3 = TextBlob(luke_words)

blob3.sentiment


blob4 = TextBlob(john_words)

blob4.sentiment



#Section 7 - Letters 
romans = data.loc[data['n'] == 'Romans']
corinth_1 = data.loc[data['n'] == '1 Corinthians']
corinth_2 = data.loc[data['n'] == '2 Corinthians']
galatians = data.loc[data['n'] == 'Galatians']
ephesians = data.loc[data['n'] == 'Ephesians']
philippians = data.loc[data['n'] == 'Philippians']
colossians = data.loc[data['n'] == 'Colossians']
thess_1 = data.loc[data['n'] == '1 Thessalonians']
thess_2 = data.loc[data['n'] == '2 Thessalonians']
tim_1 = data.loc[data['n'] == '1 Timothy']
tim_2 = data.loc[data['n'] == '2 Timothy']
titus = data.loc[data['n'] == 'Titus']
philemon = data.loc[data['n'] == 'Philemon']
hebrews = data.loc[data['n'] == 'Hebrews']
james = data.loc[data['n'] == 'James']
peter_1 = data.loc[data['n'] == '1 Peter']
peter_2 = data.loc[data['n'] == '2 Peter']
john_1 = data.loc[data['n'] == '1 John']
john_2 = data.loc[data['n'] == '2 John']
john_3 = data.loc[data['n'] == '3 John']
jude = data.loc[data['n'] == 'Jude']


#Word Clouds

romans_words = ' '.join(romans.t_x)

romans_wordcloud = WordCloud(collocations=False).generate(romans_words)
plt.imshow(romans_wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

corinth_1_words = ' '.join(corinth_1.t_x)

corinth_1_wordcloud = WordCloud(collocations=False).generate(corinth_1_words)
plt.imshow(corinth_1_wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

corinth_2_words = ' '.join(corinth_2.t_x)

corinth_2_wordcloud = WordCloud(collocations=False).generate(corinth_2_words)
plt.imshow(corinth_2_wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

galatians_words = ' '.join(galatians.t_x)

galatians_wordcloud = WordCloud(collocations=False).generate(galatians_words)
plt.imshow(galatians_wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

ephesians_words = ' '.join(ephesians.t_x)

ephesians_wordcloud = WordCloud(collocations=False).generate(ephesians_words)
plt.imshow(ephesians_wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

philippians_words = ' '.join(philippians.t_x)

philippians_wordcloud = WordCloud(collocations=False).generate(philippians_words)
plt.imshow(philippians_wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

colossians_words = ' '.join(colossians.t_x)

colossians_wordcloud = WordCloud(collocations=False).generate(colossians_words)
plt.imshow(colossians_wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

thess_1_words = ' '.join(thess_1.t_x)

thess_1_wordcloud = WordCloud(collocations=False).generate(thess_1_words)
plt.imshow(thess_1_wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

thess_2_words = ' '.join(thess_2.t_x)

thess_2_wordcloud = WordCloud(collocations=False).generate(thess_2_words)
plt.imshow(thess_2_wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

tim_1_words = ' '.join(tim_1.t_x)

tim_1_wordcloud = WordCloud(collocations=False).generate(tim_1_words)
plt.imshow(tim_1_wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

tim_2_words = ' '.join(tim_2.t_x)

tim_2_wordcloud = WordCloud(collocations=False).generate(tim_2_words)
plt.imshow(tim_2_wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

titus_words = ' '.join(titus.t_x)

titus_wordcloud = WordCloud(collocations=False).generate(titus_words)
plt.imshow(titus_wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

philemon_words = ' '.join(philemon.t_x)

philemon_wordcloud = WordCloud(collocations=False).generate(philemon_words)
plt.imshow(philemon_wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

hebrews_words = ' '.join(hebrews.t_x)

hebrews_wordcloud = WordCloud(collocations=False).generate(hebrews_words)
plt.imshow(hebrews_wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

james_words = ' '.join(james.t_x)

james_wordcloud = WordCloud(collocations=False).generate(james_words)
plt.imshow(james_wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

peter_1_words = ' '.join(peter_1.t_x)

peter_1_wordcloud = WordCloud(collocations=False).generate(peter_1_words)
plt.imshow(peter_1_wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

peter_2_words = ' '.join(peter_2.t_x)

peter_2_wordcloud = WordCloud(collocations=False).generate(peter_2_words)
plt.imshow(peter_2_wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

john_1_words = ' '.join(john_1.t_x)

john_1_wordcloud = WordCloud(collocations=False).generate(john_1_words)
plt.imshow(john_1_wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

john_2_words = ' '.join(john_2.t_x)

john_2_wordcloud = WordCloud(collocations=False).generate(john_2_words)
plt.imshow(john_2_wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

john_3_words = ' '.join(john_3.t_x)

john_3_wordcloud = WordCloud(collocations=False).generate(john_3_words)
plt.imshow(john_3_wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

jude_words = ' '.join(jude.t_x)

jude_wordcloud = WordCloud(collocations=False).generate(jude_words)
plt.imshow(jude_wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()



# Topic Analysis across Books 
romans_list = romans_words.split(" ")
corinth_1_list = corinth_1_words.split(" ")
corinth_2_list = corinth_2_words.split(" ")
galatians_list = galatians_words.split(" ")
ephesians_list = ephesians_words.split(" ")
philippians_list = philippians_words.split(" ")
colossians_list = colossians_words.split(" ")
thess_1_list = thess_1_words.split(" ")
thess_2_list = thess_2_words.split(" ")
tim_1_list = tim_1_words.split(" ")
tim_2_list = tim_2_words.split(" ")
titus_list = titus_words.split(" ")
philemon_list = philemon_words.split(" ")
hebrews_list = hebrews_words.split(" ")
james_list = james_words.split(" ")
peter_1_list = peter_1_words.split(" ")
peter_2_list = peter_2_words.split(" ")
john_1_list = john_1_words.split(" ")
john_2_list = john_2_words.split(" ")
john_3_list = john_3_words.split(" ")
jude_list = jude_words.split(" ")



#all
seven = []
seven.append(romans_list)
seven.append(corinth_1_list)
seven.append(corinth_2_list)
seven.append(galatians_list)
seven.append(ephesians_list)
seven.append(philippians_list)
seven.append(colossians_list)
seven.append(thess_1_list)
seven.append(thess_2_list)
seven.append(tim_1_list)
seven.append(tim_2_list)
seven.append(titus_list)
seven.append(philemon_list)
seven.append(hebrews_list)
seven.append(james_list)
seven.append(peter_1_list)
seven.append(peter_2_list)
seven.append(john_1_list)
seven.append(john_2_list)
seven.append(john_3_list)
seven.append(jude_list)




import gensim.corpora as corpora
# Create Dictionary
id2word = corpora.Dictionary(seven)
# Create Corpus
texts = seven
# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]
# View
print(corpus[:1][0][:30])


from pprint import pprint
# number of topics
num_topics = 3
# Build LDA model
lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                       id2word=id2word,
                                       num_topics=num_topics,
                                       passes = 100)
# Print the Keyword in the 5 topics
pprint(lda_model.print_topics())
doc_lda = lda_model[corpus]



print(lda_model.print_topics(num_topics = 3, num_words = 10))

count = 0
for i in lda_model[corpus]:
    print("doc : ", count, i)
    count += 1
    
    
    
#Sentiment Analysis 
blob1 = TextBlob(romans_words)

blob1.tags

blob1.sentiment




blob2 = TextBlob(corinth_1_words)

blob2.tags

blob2.sentiment


blob3 = TextBlob(corinth_2_words)

blob3.sentiment


blob4 = TextBlob(galatians_words)

blob4.sentiment


blob5 = TextBlob(ephesians_words)

blob5.sentiment



blob6 = TextBlob(philippians_words)

blob6.sentiment


blob7 = TextBlob(colossians_words)

blob7.sentiment



blob8 = TextBlob(thess_1_words)

blob8.sentiment



blob9 = TextBlob(thess_2_words)

blob9.sentiment


blob10 = TextBlob(tim_1_words)

blob10.sentiment


blob11 = TextBlob(tim_2_words)

blob11.sentiment


blob12 = TextBlob(titus_words)

blob12.sentiment


blob13 = TextBlob(philemon_words)

blob13.sentiment


blob14 = TextBlob(hebrews_words)

blob14.sentiment


blob15 = TextBlob(james_words)

blob15.sentiment


blob16 = TextBlob(peter_1_words)

blob16.sentiment


blob17 = TextBlob(peter_2_words)

blob17.sentiment


blob18 = TextBlob(john_1_words)

blob18.sentiment


blob19 = TextBlob(john_2_words)

blob19.sentiment


blob20 = TextBlob(john_3_words)

blob20.sentiment


blob21 = TextBlob(jude_words)

blob21.sentiment



rev = data.loc[data['n'] == 'Revelation']


#Word Clouds

rev_words = ' '.join(rev.t_x)

rev_wordcloud = WordCloud(collocations=False).generate(rev_words)
plt.imshow(rev_wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

blob22 = TextBlob(rev_words)

blob22.sentiment
    