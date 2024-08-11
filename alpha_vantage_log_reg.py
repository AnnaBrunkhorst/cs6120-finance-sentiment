import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Load data
data = pd.read_csv('./data/cleaned_data_for_modeling.csv')

# Load GloVe embeddings
def load_glove_model(glove_file):
    with open(glove_file, 'r', encoding='utf8') as f:
        model = {}
        for line in f:
            split_line = line.split()
            word = split_line[0]
            embedding = np.array([float(val) for val in split_line[1:]])
            model[word] = embedding
        return model

glove_model = load_glove_model('./data/glove.6B.100d.txt')

# Vectorize text using GloVe embeddings
def document_vector(doc):
    words = doc.split()
    word_vectors = [glove_model[word] for word in words if word in glove_model]
    vector = np.mean(word_vectors, axis=0) if word_vectors else np.zeros(100)
    return vector

data['doc_vector'] = data['cleaned_text'].apply(document_vector)
X = np.array(data['doc_vector'].tolist())
y = data['label'].values

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save the model
joblib.dump(model, 'sentiment_logistic_model.pkl')
