import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.decomposition import PCA
from pyswarm import pso
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

# Load and preprocess the dataset
df = pd.read_csv('amazon_shopping_behavior.csv')

# Assign IDs to customers
df['customer_id'] = range(1, len(df) + 1)

# Define feature columns
numeric_features = [
    'Personalized_Recommendation_Frequency', 'Shopping_Satisfaction'
]

categorical_features = [
    'Purchase_Frequency', 'Purchase_Categories', 'Personalized_Recommendation_Frequency',
    'Browsing_Frequency', 'Product_Search_Method', 'Search_Result_Exploration',
    'Customer_Reviews_Importance', 'Add_to_Cart_Browsing', 'Cart_Completion_Frequency',
    'Cart_Abandonment_Factors', 'Saveforlater_Frequency', 'Review_Left', 'Review_Reliability',
    'Review_Helpfulness', 'Recommendation_Helpfulness', 'Service_Appreciation', 'Improvement_Areas'
]

# Encode categorical features and store mappings
label_encoders = {}
for feature in categorical_features:
    if df[feature].dtype == 'object':
        le = LabelEncoder()
        df[feature] = le.fit_transform(df[feature])
        label_encoders[feature] = le

# Convert 'Purchase_Frequency' to string if necessary
df['Purchase_Frequency'] = df['Purchase_Frequency'].astype(str)

# Handle 'Others' or unexpected labels
df = df[df['Purchase_Categories'] != 'Others']

# Encode Purchase_Categories
le = LabelEncoder()
df['Purchase_Categories'] = le.fit_transform(df['Purchase_Categories'])

# Combine numeric and categorical features
all_features = numeric_features + categorical_features
X = df[all_features].values

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA for dimensionality reduction
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Apply K-means++ clustering
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=0)
kmeans_labels = kmeans.fit_predict(X_pca)
df['kmeans_cluster'] = kmeans_labels

# Calculate silhouette score for K-means++
silhouette_kmeans = silhouette_score(X_pca, kmeans_labels)
print(f'Silhouette Score for K-means++: {silhouette_kmeans}')

# Davies-Bouldin Index for K-means++
dbi_kmeans = davies_bouldin_score(X_pca, kmeans_labels)
print(f'Davies-Bouldin Index for K-means++: {dbi_kmeans}')

# WCSS for K-means++
wcss_kmeans = kmeans.inertia_
print(f'WCSS for K-means++: {wcss_kmeans}')

# Plot K-means Clustering Results
plt.figure(figsize=(10, 7))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_labels, cmap='viridis', s=50, alpha=0.5)
plt.title('K-means++ Clustering Results')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar(label='Cluster Label')
plt.show()

# PSO-K-means clustering function
def pso_kmeans(X, k, max_iter=200):
    n_particles = 50
    dimensions = X.shape[1] * k

    def objective_function(centers):
        centers = centers.reshape((k, X.shape[1]))
        labels = np.argmin(np.linalg.norm(X[:, None] - centers[None, :], axis=2), axis=1)
        return np.sum(np.linalg.norm(X - centers[labels], axis=1))

    lb = np.min(X, axis=0).repeat(k)
    ub = np.max(X, axis=0).repeat(k)

    best_centers, _ = pso(objective_function, lb, ub, swarmsize=n_particles, maxiter=max_iter)
    best_centers = best_centers.reshape((k, X.shape[1]))

    labels = np.argmin(np.linalg.norm(X[:, None] - best_centers[None, :], axis=2), axis=1)
    return labels

# Apply PSO-based clustering
pso_labels = pso_kmeans(X_scaled, k=5)
df['pso_cluster'] = pso_labels

# Calculate silhouette score for PSO-based clustering
silhouette_pso = silhouette_score(X_scaled, pso_labels) if len(set(pso_labels)) > 1 else None
print(f'Silhouette Score for PSO-based Clustering: {silhouette_pso}')

# Davies-Bouldin Index for PSO-based clustering
dbi_pso = davies_bouldin_score(X_scaled, pso_labels) if len(set(pso_labels)) > 1 else None
print(f'Davies-Bouldin Index for PSO-based Clustering: {dbi_pso}')

# Plot PSO Clustering Results
plt.figure(figsize=(10, 7))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=pso_labels, cmap='plasma', s=50, alpha=0.5)
plt.title('PSO Clustering Results')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar(label='Cluster Label')
plt.show()

# Additional clustering methods (e.g., DBSCAN)


# Prepare data for the neural network
y = df['Purchase_Categories'].values
num_classes = len(np.unique(y))

# Neural Network Model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_scaled.shape[1],)),
        tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Setup early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model
history = model.fit(
    X_scaled, y,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping]
)

# Evaluate the model
loss, accuracy = model.evaluate(X_scaled, y)
print(f'Neural Network Model Accuracy: {accuracy}')

# Plot training history
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

