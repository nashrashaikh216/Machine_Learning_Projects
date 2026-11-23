import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# Load the Mall Customers dataset
df = pd.read_csv("Customer_Segmenatation/Mall_Customers.csv")

# Display the first five rows
print(df.head())

# Display the number of rows and columns
print(df.shape)

# Display column names
print(df.columns)

# Check data typees and missing values
print(df.info())

# Check missing values
print(df.isnull().sum())

# Display statistial summary
print(df.describe())

# Display the number of customers by gender
print(df["Gender"].value_counts())

# Visualize he distribution of customers by gender
sns.countplot(
    data=df,
    x="Gender"
)

plt.title("Customer Distrbution by Gender")
plt.xlabel("Gender")
plt.ylabel("Number of Customers")
plt.show()

# Visualize Age distribution
sns.histplot(
    data=df,
    x="Age",
    kde=True
)

plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Number of Customers")
plt.show()

# Visualize Annual Income distribution
sns.histplot(
    data=df,
    x="Annual Income (k$)",
    kde=True
)

plt.title("Annual Income Distribution")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Number of Customers")
plt.show()

# Visualize Spending Score distribution
sns.histplot(
    data=df,
    x="Spending Score (1-100)",
    kde=True
)

plt.title("Spending Score Distribution")
plt.xlabel("Spending Score")
plt.ylabel("Number of Customers")
plt.show()

# Visualize Annual Income vs spending score
sns.scatterplot(
    data=df, 
    x="Annual Income (k$)",
    y="Spending Score (1-100)"
)

plt.title("Annual Income vs Spending Score")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score")
plt.show()

# Select features for clustering
features = [
    "Age",
    "Annual Income (k$)",
    "Spending Score (1-100)"
]

X = df[features].copy()

# Check for missing values in seleted features
print(X.isnull().sum())

# Fill missing numerical values using the mediaan
X = X.fillna(X.median())

# Create the StandardScaler
scaler = StandardScaler()

# Scale the features
X_scaled = scaler.fit_transform(X)

# Store inerta values for different numbers of clusters
inertia = []

# Test K values from 2 to 10
k_values = range(2,11)

for k in k_values:

    # Create the Kmean model
    kmean = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    # Train the model
    kmean.fit(X_scaled)

    # Store the inertia value
    inertia.append(kmean.inertia_)

# Plot the Elbow Method
plt.figure(figsize=(8, 5))

plt.plot(
    k_values,
    inertia,
    marker="o"
)

plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method")

plt.show()

# Calculate silhouette scores for different K values
silhouette_scores = []

for k in k_values:

    # Create the KMeans model
    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    # Generate cluster labels
    cluster_labels = kmeans.fit_predict(X_scaled)

    # Calculate silhouette score
    score = silhouette_score(
        X_scaled,
        cluster_labels
    )

    # Store the score
    silhouette_scores.append(score)

    print(
        "K =",
        k,
        "Silhouette Score =",
        score
    )


# Plot silhouette scores
plt.figure(figsize=(8, 5))

plt.plot(
    k_values,
    silhouette_scores,
    marker="o"
)

plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score for Different K Values")

plt.show()


# Select the number of clusters
# Five clusters is commonly used for this dataset
optimal_k = 5

# Create the final KMeans model
kmeans = KMeans(
    n_clusters=optimal_k,
    random_state=42,
    n_init=10
)

# Generate cluster labels
clusters = kmeans.fit_predict(
    X_scaled
)

# Add cluster labels to the original dataset
df["Cluster"] = clusters

# Display the first rows with cluster labels
print(df.head())

# Display the number of customers in each cluster
print(
    df["Cluster"].value_counts().sort_index()
)


# Calculate the final silhouette score
final_silhouette_score = silhouette_score(
    X_scaled,
    clusters
)

print(
    "Final Silhouette Score:",
    final_silhouette_score
)


# Reduce the scaled data to two dimensions using PCA
pca = PCA(
    n_components=2
)

X_pca = pca.fit_transform(
    X_scaled
)

# Create a DataFrame for PCA results
pca_df = pd.DataFrame(
    X_pca,
    columns=[
        "PC1",
        "PC2"
    ]
)

# Add cluster labels
pca_df["Cluster"] = clusters

# Visualize the clusters
sns.scatterplot(
    data=pca_df,
    x="PC1",
    y="PC2",
    hue="Cluster",
    palette="Set1",
    s=100
)

plt.title("Customer Segments Using KMeans and PCA")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(title="Cluster")

plt.show()


# Visualize clusters using Annual Income and Spending Score
plt.figure(figsize=(8, 6))

sns.scatterplot(
    data=df,
    x="Annual Income (k$)",
    y="Spending Score (1-100)",
    hue="Cluster",
    palette="Set1",
    s=100
)

plt.title("Customer Segments")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend(title="Cluster")

plt.show()


# Calculate the average values for each cluster
cluster_profile = df.groupby(
    "Cluster"
)[
    [
        "Age",
        "Annual Income (k$)",
        "Spending Score (1-100)"
    ]
].mean()

# Display the cluster profiles
print("\nCluster Profile:")
print(cluster_profile)


# Count the number of customers in each cluster
cluster_size = df["Cluster"].value_counts().sort_index()

print("\nCustomers in Each Cluster:")
print(cluster_size)


# Add customer count to the cluster profile
cluster_profile["Customer Count"] = cluster_size

# Display the complete cluster profile
print("\nComplete Cluster Profile:")
print(cluster_profile)


# Save the dataset with cluster labels
df.to_csv(
    "Mall_Customers_Segmented.csv",
    index=False
)

print(
    "\nSegmented dataset saved as Mall_Customers_Segmented.csv"
)