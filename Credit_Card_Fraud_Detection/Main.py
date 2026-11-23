import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve
)

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


# Load the dataset
df = pd.read_csv("Credit_Card_Fraud_Detection/Credit_Card_Fraud.csv")

# Display the first five rows
print(df.head())

# Display the number of rows and columns
print(df.shape)

# Display column names
print(df.columns.tolist())

# Display data types and non-null values
print(df.info())

# Check missing values
print(df.isnull().sum())

# Display statistical summary
print(df.describe(include="all"))

# Check the distribution of the target classes
print(df["Class"].value_counts())

# Display class percentages
print(df["Class"].value_counts(normalize=True) * 100)

# Visualize the class distribution
sns.countplot(
    data=df,
    x="Class"
)

plt.title("Normal vs Fraudulent Transactions")
plt.xlabel("Class")
plt.ylabel("Number of Transactions")
plt.show()


# Remove Customer ID because it is an identifier
if "Customer ID" in df.columns:
    df = df.drop(columns=["Customer ID"])


# Separate input features and target
X = df.drop(columns=["Class"])
y = df["Class"]


# Display the unique target values
print("Target values:")
print(y.unique())


# Convert target values to 0 and 1 if they are stored as strings
if y.dtype == "object":

    y = y.astype(str).str.strip().str.lower()

    unique_values = y.unique()

    print("Unique target values after cleaning:")
    print(unique_values)

    if set(unique_values) == {"0", "1"}:
        y = y.map({
            "0": 0,
            "1": 1
        })

    elif set(unique_values) == {"+", "-"}:
        y = y.map({
            "-": 0,
            "+": 1
        })

    elif set(unique_values) == {"normal", "fraud"}:
        y = y.map({
            "normal": 0,
            "fraud": 1
        })


# Convert target to numeric
y = pd.to_numeric(
    y,
    errors="coerce"
)

# Remove rows where target could not be converted
valid_rows = y.notna()

X = X.loc[valid_rows].copy()
y = y.loc[valid_rows].astype(int)


# Display final class distribution
print("Final class distribution:")
print(y.value_counts())


# Identify numerical columns
numerical_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

# Identify categorical columns
categorical_features = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

print("Numerical features:")
print(numerical_features)

print("Categorical features:")
print(categorical_features)


# Split the dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Create preprocessing for numerical features
numerical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)


# Create preprocessing for categorical features
categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)


# Combine numerical and categorical preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        (
            "numerical",
            numerical_pipeline,
            numerical_features
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    ]
)


# Create a baseline Logistic Regression model
logistic_model = ImbPipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            LogisticRegression(
                max_iter=1000
            )
        )
    ]
)


# Train the baseline Logistic Regression model
logistic_model.fit(
    X_train,
    y_train
)


# Make predictions
y_pred_logistic = logistic_model.predict(
    X_test
)

# Get probability predictions
y_prob_logistic = logistic_model.predict_proba(
    X_test
)[:, 1]


# Calculate Logistic Regression metrics
accuracy_logistic = accuracy_score(
    y_test,
    y_pred_logistic
)

precision_logistic = precision_score(
    y_test,
    y_pred_logistic,
    zero_division=0
)

recall_logistic = recall_score(
    y_test,
    y_pred_logistic,
    zero_division=0
)

f1_logistic = f1_score(
    y_test,
    y_pred_logistic,
    zero_division=0
)

roc_auc_logistic = roc_auc_score(
    y_test,
    y_prob_logistic
)


# Display Logistic Regression results
print("\nLogistic Regression")
print("Accuracy:", accuracy_logistic)
print("Precision:", precision_logistic)
print("Recall:", recall_logistic)
print("F1 Score:", f1_logistic)
print("ROC-AUC:", roc_auc_logistic)


# Display classification report
print(
    classification_report(
        y_test,
        y_pred_logistic,
        zero_division=0
    )
)


# Create Logistic Regression confusion matrix
cm_logistic = confusion_matrix(
    y_test,
    y_pred_logistic
)

# Display confusion matrix
sns.heatmap(
    cm_logistic,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Logistic Regression Confusion Matrix")
plt.show()


# Create preprocessing without scaling categorical output manually
smote_preprocessor = ColumnTransformer(
    transformers=[
        (
            "numerical",
            Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="median"
                        )
                    ),
                    (
                        "scaler",
                        StandardScaler()
                    )
                ]
            ),
            numerical_features
        ),
        (
            "categorical",
            Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="most_frequent"
                        )
                    ),
                    (
                        "onehot",
                        OneHotEncoder(
                            handle_unknown="ignore"
                        )
                    )
                ]
            ),
            categorical_features
        )
    ]
)


# Create Logistic Regression with SMOTE
smote_logistic_model = ImbPipeline(
    steps=[
        (
            "preprocessor",
            smote_preprocessor
        ),
        (
            "smote",
            SMOTE(
                random_state=42
            )
        ),
        (
            "model",
            LogisticRegression(
                max_iter=1000
            )
        )
    ]
)


# Train Logistic Regression with SMOTE
smote_logistic_model.fit(
    X_train,
    y_train
)


# Make predictions using SMOTE model
y_pred_smote = smote_logistic_model.predict(
    X_test
)

# Get probability predictions
y_prob_smote = smote_logistic_model.predict_proba(
    X_test
)[:, 1]


# Calculate SMOTE Logistic Regression metrics
accuracy_smote = accuracy_score(
    y_test,
    y_pred_smote
)

precision_smote = precision_score(
    y_test,
    y_pred_smote,
    zero_division=0
)

recall_smote = recall_score(
    y_test,
    y_pred_smote,
    zero_division=0
)

f1_smote = f1_score(
    y_test,
    y_pred_smote,
    zero_division=0
)

roc_auc_smote = roc_auc_score(
    y_test,
    y_prob_smote
)


# Display SMOTE Logistic Regression results
print("\nLogistic Regression + SMOTE")
print("Accuracy:", accuracy_smote)
print("Precision:", precision_smote)
print("Recall:", recall_smote)
print("F1 Score:", f1_smote)
print("ROC-AUC:", roc_auc_smote)


# Display classification report
print(
    classification_report(
        y_test,
        y_pred_smote,
        zero_division=0
    )
)


# Create SMOTE confusion matrix
cm_smote = confusion_matrix(
    y_test,
    y_pred_smote
)

# Display SMOTE confusion matrix
sns.heatmap(
    cm_smote,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Logistic Regression + SMOTE Confusion Matrix")
plt.show()


# Create Random Forest model with SMOTE
random_forest_model = ImbPipeline(
    steps=[
        (
            "preprocessor",
            smote_preprocessor
        ),
        (
            "smote",
            SMOTE(
                random_state=42
            )
        ),
        (
            "model",
            RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)


# Train Random Forest
random_forest_model.fit(
    X_train,
    y_train
)


# Make Random Forest predictions
y_pred_rf = random_forest_model.predict(
    X_test
)

# Get Random Forest probability predictions
y_prob_rf = random_forest_model.predict_proba(
    X_test
)[:, 1]


# Calculate Random Forest metrics
accuracy_rf = accuracy_score(
    y_test,
    y_pred_rf
)

precision_rf = precision_score(
    y_test,
    y_pred_rf,
    zero_division=0
)

recall_rf = recall_score(
    y_test,
    y_pred_rf,
    zero_division=0
)

f1_rf = f1_score(
    y_test,
    y_pred_rf,
    zero_division=0
)

roc_auc_rf = roc_auc_score(
    y_test,
    y_prob_rf
)


# Display Random Forest results
print("\nRandom Forest + SMOTE")
print("Accuracy:", accuracy_rf)
print("Precision:", precision_rf)
print("Recall:", recall_rf)
print("F1 Score:", f1_rf)
print("ROC-AUC:", roc_auc_rf)


# Display Random Forest classification report
print(
    classification_report(
        y_test,
        y_pred_rf,
        zero_division=0
    )
)


# Create Random Forest confusion matrix
cm_rf = confusion_matrix(
    y_test,
    y_pred_rf
)

# Display Random Forest confusion matrix
sns.heatmap(
    cm_rf,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Random Forest + SMOTE Confusion Matrix")
plt.show()


# Calculate ROC curve for Logistic Regression
fpr_logistic, tpr_logistic, _ = roc_curve(
    y_test,
    y_prob_logistic
)

# Calculate ROC curve for SMOTE Logistic Regression
fpr_smote, tpr_smote, _ = roc_curve(
    y_test,
    y_prob_smote
)

# Calculate ROC curve for Random Forest
fpr_rf, tpr_rf, _ = roc_curve(
    y_test,
    y_prob_rf
)


# Plot ROC curves
plt.figure(figsize=(8, 6))

plt.plot(
    fpr_logistic,
    tpr_logistic,
    label=f"Logistic Regression AUC = {roc_auc_logistic:.3f}"
)

plt.plot(
    fpr_smote,
    tpr_smote,
    label=f"Logistic + SMOTE AUC = {roc_auc_smote:.3f}"
)

plt.plot(
    fpr_rf,
    tpr_rf,
    label=f"Random Forest + SMOTE AUC = {roc_auc_rf:.3f}"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC-AUC Comparison")

plt.legend()

plt.show()


# Create a table to compare all models
comparison = pd.DataFrame(
    {
        "Model": [
            "Logistic Regression",
            "Logistic Regression + SMOTE",
            "Random Forest + SMOTE"
        ],
        "Accuracy": [
            accuracy_logistic,
            accuracy_smote,
            accuracy_rf
        ],
        "Precision": [
            precision_logistic,
            precision_smote,
            precision_rf
        ],
        "Recall": [
            recall_logistic,
            recall_smote,
            recall_rf
        ],
        "F1 Score": [
            f1_logistic,
            f1_smote,
            f1_rf
        ],
        "ROC-AUC": [
            roc_auc_logistic,
            roc_auc_smote,
            roc_auc_rf
        ]
    }
)


# Display model comparison
print("\nModel Comparison")
print(comparison)


# Create a new transaction
new_transaction = pd.DataFrame(
    {
        "A1": [1],
        "A2": [30],
        "A3": [5000],
        "A4": [2],
        "A5": [1],
        "A6": [1],
        "A7": [1],
        "A8": [100],
        "A9": [1],
        "A10": [1],
        "A11": [1],
        "A12": [1],
        "A13": [1],
        "A14": [1]
    }
)


# Make sure the new transaction contains the same columns
new_transaction = new_transaction.reindex(
    columns=X.columns
)


# Predict the new transaction using Random Forest
new_prediction = random_forest_model.predict(
    new_transaction
)

# Get fraud probability
new_probability = random_forest_model.predict_proba(
    new_transaction
)[:, 1]


# Display the prediction
if new_prediction[0] == 1:
    print("New Transaction: FRAUD")
else:
    print("New Transaction: NORMAL")

print(
    "Fraud Probability:",
    new_probability[0]
)