import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the built-in Iris dataset
iris = load_iris()

# Create a DataFrame using the Iris features
df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

# Add the target colun
df["target"] = iris.target

# Add the flower specied name
df["species"] = df["target"].map(
    {
        0: "setosa",
        1: "versicolor",
        2: "virginica"
    }
)

# Display the first five rows
print(df.head())

# Display the number of rows and columns
print(df.shape)

# Display column names
print(df.columns)

# Check data types and missing values
print(df.info())

# check missing values
print(df.isnull().sum())

# Display statistical summary
print(df.describe())

# Display the number of samples in each species
print(df["species"].value_counts())

# Visualize the relationship between sepal length and sepal width
sns.scatterplot(
    data=df,
    x="sepal length (cm)",
    y="sepal width (cm)",
    hue="species"
)

plt.title("Sepal Length VS Sepal Width")
plt.show()

# Visualize the relationship between petal length and petal width
sns.scatterplot(
    data=df,
    x="petal length (cm)",
    y="petal width (cm)",
    hue="species"
)

plt.title("Petal Length vs Petal Width")
plt.show()

# Create a pairplot to visualize relationships between all features
sns.pairplot(
    df,
    hue="species"
)

plt.show()

# Select the input features
X = df[iris.feature_names]

# Select the target variable
y = df["target"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Create a StandardScaler
scaler = StandardScaler()

# Fit the scaler only on the traninig data
X_train_scaled = scaler.fit_transform(X_train)

# Ue the same scaling on the test data
X_test_scaled = scaler.transform(X_test)

# Create the KNN model
knn_model = KNeighborsClassifier(
    n_neighbors=5
)

# Train the KNN model
knn_model.fit(
    X_train_scaled,
    y_train
)

# Make predictions using KNN
y_pred_knn = knn_model.predict(
    X_test_scaled
)

# Calculate KNN accuracy
knn_accuracy = accuracy_score(
    y_test, 
    y_pred_knn
)

print("KNN Accuracy: ",knn_accuracy)

# Display the KNN classification report
print(
    classification_report(
        y_test,
        y_pred_knn,
        target_names= iris.target_names
    )
)

# Create the KNN confusion matrix
cm_knn = confusion_matrix(
    y_test,
    y_pred_knn
)

# Display the KNN confusion matrix
sns.heatmap(
    cm_knn,
    annot=True,
    fmt="d",
    xticklabels=iris.target_names,
    yticklabels=iris.target_names
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("KNN Confusion Matrix")
plt.show()

# Create the SVM model
svm_model = SVC(
    kernel="rbf",
    random_state=42
)

# Train the SVM model
svm_model.fit(
    X_train_scaled,
    y_train
)

# Make predictions using SVM
y_pred_svm = svm_model.predict(
    X_test_scaled
)

# Claculate SVM accuracy
svm_accuracy = accuracy_score(
    y_test,
    y_pred_svm
)

print("SVM Accuracy: ",svm_accuracy)

# Display the SVM classification report
print(
    classification_report(
        y_test,
        y_pred_svm,
        target_names=iris.target_names
    )
)

# Create the SVM confusion matrix
cm_svm = confusion_matrix(
    y_test,
    y_pred_svm
)

# Display the SVM confusion matrix
sns.heatmap(
    cm_svm,
    annot=True,
    fmt="d",
    xticklabels=iris.target_names,
    yticklabels=iris.target_names
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("SVM Confusion Matrix")
plt.show()

# compare the two models
comparison = pd.DataFrame(
    {
        "Model" :[
            "KNN",
            "SVM"
        ],
        "Accuracy":[
            knn_accuracy,
            svm_accuracy
        ]
    }
)

print(comparison)

# Create a new flower sample
new_flower = [
    [
        5.1,
        3.5,
        1.4,
        0.2
    ]
]

# Scale the new flower using the already fitted sacaler
new_flower_scaled = scaler.transform(
    new_flower
)

# Predict the species using KNN
knn_prediction = knn_model.predict(
    new_flower_scaled
)

# Convert the predicted class number into species name
print(
    "KNN Predicted Species: ",
    iris.target_names[knn_prediction[0]]
)

# Predict the species using SVM
svm_prediction = svm_model.predict(
    new_flower_scaled
)

# Convert the predicted class number into species name
print(
    "SVM Predicted Species: ",
    iris.target_names[svm_prediction[0]]
)