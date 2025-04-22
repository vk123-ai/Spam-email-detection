import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import nltk
from nltk.corpus import stopwords
import re

# Download stopwords
nltk.download('stopwords')

# Load the dataset
# You can use any dataset, for example, a CSV file with columns 'EmailText' and 'Label' (spam=1, not spam=0)
# Example: df = pd.read_csv('spam_data.csv')

# For demonstration, let's create a small example dataset
data = {
    'EmailText': [
        'Hi Vikrant',
        'KIRIT 5.0, the inter-collegiate case-analysis competition hosted by Kirloskar Institute of Management, is now live. Showcase your skills and grab a chance to secure PPIs.',
        'Application Link: https://unstop.com/competitions/kirit-50-kirloskar-institute-of-management-kim-1295120',
        'Exciting Prizes:',
        'Pre-placement interview offers (PPIs) for Winning Team',
        'Cash Prize Pool worth INR 1.75 Lakhs',
        'Participation Certificates',
        'Eligibility: Open to all (Undergraduate courses)',
        'Best Regards',
        'Team Unstop',
    ],
    'Label': [1, 0, 1, 0, 1, 1, 1, 1, 1, 1]  # Make sure this matches the length of 'EmailText'
}
df = pd.DataFrame(data)

# Check the DataFrame
print(df.head())

# Preprocess the text data
def preprocess_text(text):
    # Remove special characters and numbers
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\d', ' ', text)
    # Convert to lower case
    text = text.lower()
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

df['EmailText'] = df['EmailText'].apply(preprocess_text)

# Check the preprocessed text
print(df['EmailText'].head())

# Vectorize the text data using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['EmailText']).toarray()
y = df['Label']

# Check the shapes of X and y
print(X.shape, y.shape)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a Naive Bayes model
model = MultinomialNB()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print(classification_report(y_test, y_pred))

# Example usage
test_email = "Congratulations! You have won a free ticket to Bahamas."
processed_email = preprocess_text(test_email)
vectorized_email = vectorizer.transform([processed_email]).toarray()
prediction = model.predict(vectorized_email)
print("Spam" if prediction[0] == 1 else "Not Spam")
