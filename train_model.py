import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import nltk
from nltk.corpus import stopwords

# Ensure stopwords are downloaded
try:
    stop_words = stopwords.words('english')
except LookupError:
    nltk.download('stopwords')
    stop_words = stopwords.words('english')

# Load dataset
data = pd.read_csv("tweets.csv")
X = data['text']
y = data['sentiment']

# Preprocess and vectorize text (reduce max_features for small data)
vectorizer = TfidfVectorizer(max_features=10, stop_words=stop_words)
X_vec = vectorizer.fit_transform(X)

# Train model on all data (no split for now)
model = LogisticRegression()
model.fit(X_vec, y)

# Save model and vectorizer
joblib.dump(model, 'sentiment_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

# Optional: Predict on training data to check
predictions = model.predict(X_vec)
accuracy = (predictions == y).mean()
print("Training Accuracy:", accuracy)