import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
import pickle

df = pd.read_csv('framingham.csv')
df.drop(['education'], inplace = True, axis = 1)
df.rename(columns ={'male':'Sex_male'}, inplace = True)
df.dropna(axis = 0, inplace = True)



X = np.asarray(df[['age', 'Sex_male', 'cigsPerDay','totChol', 'sysBP','BMI','heartRate','glucose']])

y = np.asarray(df['TenYearCHD'])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)


# Make predictions
y_pred = model.predict(X_test)
# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy:.2f}')

pickle.dump(model,open("model.pkl","wb"))            # pkl for pickle extension ---- to read : m = pickle.load(open("model.pkl","rb"))
