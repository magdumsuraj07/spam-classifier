import pandas as pd

# Loading dataset
messages = pd.read_csv(
    "smsspamcollection/SMSSpamCollection", delimiter="\t", names=["label", "message"]
)

# Importing necessary libraries for data cleaning and preprocessing
import re
import nltk

nltk.download("stopwords")

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

# Text preprocessing
corpus = []
for i in range(len(messages)):
    # Removing non-alphabet characters
    text = re.sub("[^a-zA-Z]", " ", messages["message"][i])
    text = text.lower().split()

    # Stemming and removing stopwords
    text = [ps.stem(word) for word in text if word not in stopwords.words("english")]
    processed_message = " ".join(text)
    corpus.append(processed_message)

# Creating Bag of Words (BoW) model
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features=2500)
X = cv.fit_transform(corpus).toarray()

# Encoding labels (spam/ham)
y = pd.get_dummies(messages["label"])
y = y.iloc[:,1].values

# Splitting dataset into training and testing sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Training Naive Bayes classifier
from sklearn.naive_bayes import MultinomialNB

nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)

# Predicting on test data
y_pred = nb_model.predict(X_test)

from sklearn.metrics import confusion_matrix
confusion_m = confusion_matrix(y_test, y_pred)
print(confusion_m)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)
