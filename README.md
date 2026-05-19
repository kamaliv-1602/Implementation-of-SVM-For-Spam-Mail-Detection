# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. Collect and preprocess email data by cleaning text, removing stop words, and converting messages into numerical features using TF-IDF.

2. Split the dataset into training and testing sets, then train the Support Vector Machine (SVM) model using the training data.

3. Test the trained SVM model with unseen emails to classify them as spam or not spam.

4. Evaluate the model performance using accuracy, precision, recall, and confusion matrix results.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Kamali V
RegisterNumber:  212225240066
*/
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("spam.csv", encoding='latin-1')

data = data[['v1','v2']]
data.columns = ['label','message']

# Convert labels
data['label'] = data['label'].map({'ham':0, 'spam':1})

# Features and target
X = data['message']
y = data['label']
vectorizer = TfidfVectorizer(stop_words='english')
X_vectorized = vectorizer.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y, test_size=0.2, random_state=42)
model = SVC(kernel='linear')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Ham','Spam'],
            yticklabels=['Ham','Spam'])

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix for SVM Spam Detection")
plt.show()
```

## Output:

<img width="585" height="482" alt="image" src="https://github.com/user-attachments/assets/a3680ce1-2b77-4a32-b705-5daa39128bfe" />



## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
