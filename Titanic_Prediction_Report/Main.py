import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_auc_score, roc_curve

df = pd.read_csv("train.csv")

print(df.head())
print(df.shape)
print(df.info())
print(df.isnull().sum())

sns.countplot(x="Survived", data=df)
plt.show()

sns.countplot(x="Sex", hue="Survived", data=df)
plt.show()

sns.countplot(x="Pclass", hue="Survived", data=df)
plt.show()

sns.histplot(df["Age"], bins=30, kde=True)
plt.show()

df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
df["IsAlone"] = np.where(df["FamilySize"] == 1, 1, 0)

df["Title"] = df["Name"].str.extract(r",\s*([^.]*)\.", expand=False)

common_titles = ["Mr", "Miss", "Mrs", "Master"]

df["Title"] = df["Title"].apply(
    lambda x: x if x in common_titles else "Rare"
)

features = [
    "Pclass",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Embarked",
    "FamilySize",
    "IsAlone",
    "Title"
]

X = df[features].copy()
y = df["Survived"]

X["Age"] = X["Age"].fillna(X["Age"].median())
X["Embarked"] = X["Embarked"].fillna(X["Embarked"].mode()[0])

X = pd.get_dummies(
    X,
    columns=["Sex", "Embarked", "Title"],
    drop_first=True
)

X = X.astype(float)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

log_model = LogisticRegression(max_iter=1000)

log_model.fit(X_train_scaled, y_train)

y_pred_log = log_model.predict(X_test_scaled)
y_prob_log = log_model.predict_proba(X_test_scaled)[:, 1]

log_accuracy = accuracy_score(y_test, y_pred_log)
log_precision = precision_score(y_test, y_pred_log)
log_recall = recall_score(y_test, y_pred_log)
log_f1 = f1_score(y_test, y_pred_log)
log_roc_auc = roc_auc_score(y_test, y_prob_log)

print("Logistic Regression")
print("Accuracy:", log_accuracy)
print("Precision:", log_precision)
print("Recall:", log_recall)
print("F1 Score:", log_f1)
print("ROC-AUC:", log_roc_auc)
print(classification_report(y_test, y_pred_log))

cm_log = confusion_matrix(y_test, y_pred_log)

sns.heatmap(
    cm_log,
    annot=True,
    fmt="d"
)

plt.title("Logistic Regression Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)
y_prob_rf = rf_model.predict_proba(X_test)[:, 1]

rf_accuracy = accuracy_score(y_test, y_pred_rf)
rf_precision = precision_score(y_test, y_pred_rf)
rf_recall = recall_score(y_test, y_pred_rf)
rf_f1 = f1_score(y_test, y_pred_rf)
rf_roc_auc = roc_auc_score(y_test, y_prob_rf)

print("Random Forest")
print("Accuracy:", rf_accuracy)
print("Precision:", rf_precision)
print("Recall:", rf_recall)
print("F1 Score:", rf_f1)
print("ROC-AUC:", rf_roc_auc)
print(classification_report(y_test, y_pred_rf))

cm_rf = confusion_matrix(y_test, y_pred_rf)

sns.heatmap(
    cm_rf,
    annot=True,
    fmt="d"
)

plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

comparison = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Random Forest"
    ],
    "Accuracy": [
        log_accuracy,
        rf_accuracy
    ],
    "Precision": [
        log_precision,
        rf_precision
    ],
    "Recall": [
        log_recall,
        rf_recall
    ],
    "F1 Score": [
        log_f1,
        rf_f1
    ],
    "ROC-AUC": [
        log_roc_auc,
        rf_roc_auc
    ]
})

print(comparison.round(3))

comparison.set_index("Model")["Accuracy"].plot(kind="bar")

plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.xticks(rotation=0)
plt.show()

fpr_log, tpr_log, _ = roc_curve(
    y_test,
    y_prob_log
)

fpr_rf, tpr_rf, _ = roc_curve(
    y_test,
    y_prob_rf
)

plt.figure(figsize=(8, 6))

plt.plot(
    fpr_log,
    tpr_log,
    label=f"Logistic Regression (AUC = {log_roc_auc:.3f})"
)

plt.plot(
    fpr_rf,
    tpr_rf,
    label=f"Random Forest (AUC = {rf_roc_auc:.3f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf_model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print(feature_importance)

sns.barplot(
    x="Importance",
    y="Feature",
    data=feature_importance
)

plt.title("Random Forest Feature Importance")
plt.show()

new_passenger = pd.DataFrame({
    "Pclass": [3],
    "Sex": ["female"],
    "Age": [25],
    "SibSp": [0],
    "Parch": [0],
    "Fare": [10],
    "Embarked": ["S"],
    "FamilySize": [1],
    "IsAlone": [1],
    "Title": ["Miss"]
})

new_passenger = pd.get_dummies(
    new_passenger,
    columns=["Sex", "Embarked", "Title"],
    drop_first=True
)

new_passenger = new_passenger.reindex(
    columns=X.columns,
    fill_value=0
)

new_passenger = new_passenger.astype(float)

new_passenger_scaled = scaler.transform(new_passenger)

prediction = log_model.predict(new_passenger_scaled)

probability = log_model.predict_proba(
    new_passenger_scaled
)[0][1]

if prediction[0] == 1:
    print("Passenger is predicted to SURVIVE.")
else:
    print("Passenger is predicted NOT to survive.")

print("Survival probability:", round(probability * 100, 2), "%")

best_model = comparison.loc[
    comparison["Accuracy"].idxmax(),
    "Model"
]

best_accuracy = comparison["Accuracy"].max()

print("Best Model:", best_model)
print("Best Accuracy:", round(best_accuracy, 3))