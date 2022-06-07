import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import nltk

nltk.download('stopwords')
true = pd.read_csv('True.csv')
fake = pd.read_csv('Fake.csv')
true.head(3)
fake.head(3)
# print(true.shape)
# print(fake.shape)
true['label'] = 1
fake['label'] = 0
frames = [true.loc[:5000][:], fake.loc[:5000][:]]
df = pd.concat(frames)

# print(df.shape)
# print(df.tail())

X = df. drop('label', axis=1)
y = df['label']
df = df.dropna()
df2 = df.copy()

df2.reset_index(inplace=True)
# print(df2['title'][2])

ps = PorterStemmer()

corpus = []
for i in range(0, len(df2)):
    review = re.sub('[^a-zA-Z]', ' ', df2['text'][i])
    review = review.lower()
    review = review.split()
    
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)

tfidf_v = TfidfVectorizer(max_features=5000, ngram_range=(1,3))
X = tfidf_v.fit_transform(corpus).toarray()
print(X)
y = df2['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)