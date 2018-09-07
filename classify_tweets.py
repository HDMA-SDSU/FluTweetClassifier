from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
import pandas as pd

# set up a classifier pipline: word/n-gram counter -> tfidf transformer -> support vector classifier
classifier = Pipeline([
    ('vectorizer', CountVectorizer(ngram_range=(1,3))),
    ('tfidf', TfidfTransformer()),
    ('clf', OneVsRestClassifier(LinearSVC()))
])

# load excel (the most time-consuming part)
df = pd.read_excel("training_set.xlsx", "Sheet1")

# the tweets are the input
tweets = df.text.astype(str)

# the 'valid' column is the desired output (is this tweet a valid flu tweet?)
Y = df.valid.astype(bool)

# train the classifier
classifier.fit(tweets, Y)


# try it out on some sample messages
texts = [
	'i hate having the flu', 
	'i dont want to get a flu shot!!', 
	'the flu sucks, i keep coughing', 
	'Studies show that flu rates are rising in the nation: http://www.fakeurl.com',
	'RT i have the flu!!! ahhh!!!!',
	'just got a flu shot, but i hope i dont get the flu!',
	'got a flu shot i better not get the flu now!'
]
	
outputs = classifier.predict(texts)
		
for text, output in zip(texts, outputs):
	print '%s --> %s' % (text, str(output))
	
