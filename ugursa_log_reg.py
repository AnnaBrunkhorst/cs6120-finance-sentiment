import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

# Load the preprocessed data
data = pd.read_csv('ugursa_pre.csv')

# Prepare the data
X = np.stack(data['wordvec'].apply(lambda x: np.fromstring(x.strip('[]'), sep=' ')))
y = data['label'].values

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save the model
joblib.dump(model, 'ugursa_log_reg.pkl')
