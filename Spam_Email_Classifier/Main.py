import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Load the SMS Spam dataset
df = pd.read_csv("Spam_Email_Classifier/spam(1).csv", encoding="latin-1")

# Display the first five rows
print(df.head())

# Display the number of rows and columns
print(df.shape)

# Display column names
print(df.columns)

# Keep only the columns containing the message and label
df = df[["Message", "Message"]]

# Rename the columns
df.columns = ["label","message"]

# Check for missing values
print(df.isnull().sum())

# Remove rows containing missing values
df = df.dropna()

# Check the number of spam and ham messages
print(df["label"].value_counts())

# Display the percentage of each class
print(df["label"].value_counts(normalize=True) * 100)

# Visualize the number of spam and ham message
sns.countplot(
    data=df,
    x="label"
)

plt.title("Spam vs Ham Messages")
plt.xlabel("Message Type")
plt.ylabel("Number of Messages")
plt.show()

# Convert spam into 1 and ha into 0
df["label"] = df["label"].map(
    {
        "ham":0,
        "spam":1
    }
)

# Create a function to clean the text
def clean_text(text):
    # Convert text to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(
        r"http\S+|www\S+|https\S+", 
        "",
        text
    )

    # Remove email addresses
    text = re.sub(
        r"\S+@\S+",
        "",
        text
    )

    # Remove numbers
    text = re.sub(
        r"\d+",
        "",
        text
    )

    # Remove punctuation and special characters
    text = re.sub(
        r"[^a-zA-Z\s]",
        "",
        text
    )

    # Remove extra spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text

# Apply the text cleaning 
df["clean_message"] = df["message"].apply(clean_text)

# Display original and cleaned messages
print(
    df[
        [
            "message",
            "clean_message"
        ]
    ].head()
)

# Select the input text
X = df["clean_message"]

# Slect the target variable
y = df["label"]

# Split the dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Create the TF-IDF vectorizer
tfidf = TfidfVectorizer(
    stop_words="english",
    max_features=5000
)

# Learn the vocabulary and transform the training messages
X_train_tfidf = tfidf.fit_transform(X_train)

# Transform the testing messages using the same vocabulary
X_test_tfidf = tfidf.transform(X_test)

# Display the shape of the TF-IDF matrices
print("Training TF-IDf shape:", X_train_tfidf.shape)
print("Testing TF-IDF shape:",X_test_tfidf.shape)

# Create the Naive Bayes model
nb_model = MultinomialNB()

# Train the Naive Bayes model
nb_model.fit(
    X_train_tfidf,
    y_train
)

# Make predictions using Naive Bayes
y_pred_nb = nb_model.predict(
    X_test_tfidf
)

# Calculate Naive Bayes metrics
accuracy_nb = accuracy_score(
    y_test,
    y_pred_nb
)

precision_nb = precision_score(
    y_test,
    y_pred_nb
)

recall_nb = recall_score(
    y_test,
    y_pred_nb
)

f1_nb = f1_score(
    y_test,
    y_pred_nb
)

# Disaply Naive Bayees results
print("Naive Byes")
print("Accuracy: ",accuracy_nb)
print("Precision: ",precision_nb)
print("Recall: ",recall_nb)
print("F1 Score: ",f1_nb)

# Display the Naive Bayes Classification report
print(
    classification_report(
        y_test,
        y_pred_nb,
        target_names=[
            "Ham",
            "Spam"
        ]
    )
)

# Create the Naive Bayes confusion mertix
cm_nb = confusion_matrix(
    y_test,
    y_pred_nb
)

# Display the Naive Bayes Confusion matrix
sns.heatmap(
    cm_nb,
    annot=True,
    fmt="d",
    xticklabels=[
        "Ham",
        "Spam"
    ],
    yticklabels=[
        "Ham",
        "Spam"
    ]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Naive Bayes confusion Matrix")
plt.show()

# create the Logistic Regression model
log_model = LogisticRegression(
    max_iter=1000
)

# Train the Logistic Regression model
log_model.fit(
    X_train_tfidf,
    y_train
)

# Make Prediction using Logistic Regression
y_pred_log = log_model.predict(
    X_test_tfidf
)

# Calculate Logistic Regression metrics
accuracy_log = accuracy_score(
    y_test,
    y_pred_log
)

precision_log = precision_score(
    y_test,
    y_pred_log
)

recall_log = recall_score(
    y_test,
    y_pred_log
)

f1_log = f1_score(
    y_test,
    y_pred_log
)

# Disaply Logistic Regression results
print("Logistic Regression")
print("Accuracy:", accuracy_log)
print("Precision:", precision_log)
print("Recall:", recall_log)
print("F1 Score:", f1_log)

# Display the Logistic Regression clssification report
print(
    classification_report(
        y_test,
        y_pred_log,
        target_names=[
            "Ham",
            "Spam"
        ]
    )
)

# Create the Logistic Regression confusion matrix
cm_log = confusion_matrix(
    y_test,
    y_pred_log
)

# Display the Logistic Regression confusion matix
sns.heatmap(
    cm_log,
    annot=True,
    fmt="d",
    xticklabels=[
        "Ham",
        "Spam"
    ],
    yticklabels=[
        "Ham",
        "Spam"
    ]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Logistic Regression Confusion Matrix")
plt.show()

# Compare both models
comparison = pd.DataFrame(
    {
        "Model" :[
            "Naive Bayes",
            "Logistic Regression"
        ],
        "Accuracy" :[
            accuracy_nb,
            accuracy_log
        ],
        "Precision" :[
            precision_nb,
            precision_log
        ],
        "Recall" :[
            recall_nb,
            recall_log
        ],
        "F1 Score" :[
            f1_nb,
            f1_log
        ]
    }
)

# Display the model comparision
print(comparison)

# Create a new message for prediction
new_messages = [
    "Congratulations! You have won a free prize. Click here to claim it.",
    "Hi, are we meeting tomorrow for the project?"
] 

# Clean the new messages
new_messages_cleaned = [
    clean_text(message)
    for message in new_messages
]

# Convert the new messages into TF-IDF features
new_messages_tfidf = tfidf.transform(
    new_messages_cleaned
)

# Predict using Naive bayed
nb_predictions = nb_model.predict(
    new_messages_tfidf
)

# Predict using Logistic Regression
log_prediction = log_model.predict(
    new_messages_tfidf
)

# Display predictions for each message
for i in range(len(new_messages)):
    print("\nMessage:", new_messages[i])

    if nb_predictions[i] == 1:
        print("Naive Bayes Prediction: Spam")
    else:
        print("Naive Bayes Prediction: Ham")

    if log_prediction[i] == 1:
            print("Logistic Regression Prediction: Spam")
    else:
        print("Logistic Regression Prediction: Ham")