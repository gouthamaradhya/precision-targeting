from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score
from pyswarm import pso
import tensorflow as tf

app = Flask(__name__)
CORS(app)

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
category_mappings = {}
inverse_category_mappings = {}
for feature in categorical_features:
    if df[feature].dtype == 'object':  # Check if the column is of object (string) type
        le = LabelEncoder()
        df[feature] = le.fit_transform(df[feature])
        label_encoders[feature] = le
        if feature == 'Purchase_Categories':
            # Save the mappings for encoding and decoding
            category_mappings = {i: category for i, category in enumerate(le.classes_)}
            inverse_category_mappings = {category: i for i, category in enumerate(le.classes_)}

# Combine numeric and categorical features
all_features = numeric_features + categorical_features

# Extract features and scale them
X = df[all_features].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply K-means clustering
kmeans = KMeans(n_clusters=5, random_state=0)
kmeans_labels = kmeans.fit_predict(X_scaled)
df['kmeans_cluster'] = kmeans_labels

# Calculate silhouette score for K-means
silhouette_kmeans = silhouette_score(X_scaled, kmeans_labels)
print(f'Silhouette Score for K-means: {silhouette_kmeans}')

# PSO-K-means clustering function
def pso_kmeans(X, k, max_iter=100):
    n_particles = 30
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
silhouette_pso = silhouette_score(X_scaled, pso_labels)
print(f'Silhouette Score for PSO-based Clustering: {silhouette_pso}')

# Create and fit the neural network model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_scaled.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(len(category_mappings), activation='softmax')  # Assuming number of categories as output classes
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_scaled, df['Purchase_Categories'], epochs=10, batch_size=32, validation_split=0.2)

# Define mapping from categories to products
category_to_products = {
    'Electronics': ['Smartphone', 'Laptop', 'Headphones', 'Smartwatch', 'Camera'],
    'Clothing': ['T-shirt', 'Jeans', 'Jacket', 'Sneakers', 'Hat'],
    'Books': ['Fiction Novel', 'Non-Fiction', 'Biographies', 'Self-Help', 'Science Fiction'],
    'Home & Kitchen': ['Blender', 'Microwave', 'Coffee Maker', 'Cookware Set', 'Vacuum Cleaner'],
    'Toys': ['Action Figure', 'Board Game', 'Doll', 'Puzzle', 'Building Blocks'],
    'Clothing and Fashion': ['Dress', 'Jacket', 'Shirt', 'Jeans', 'Shoes'],
    'Beauty and Personal Care': ['Face Cream', 'Shampoo', 'Lipstick', 'Moisturizer', 'Sunscreen'],
    'Beauty and Personal Care;Clothing and Fashion': ['Face Cream', 'Shirt', 'Shampoo', 'Dress', 'Lipstick'],
    'Clothing and Fashion;Home and Kitchen': ['Dress', 'Cookware Set', 'Shirt', 'Blender', 'Jacket'],
    'Beauty and Personal Care;Home and Kitchen': ['Face Cream', 'Blender', 'Shampoo', 'Cookware Set', 'Moisturizer'],
    'Others': ['Generic Item 1', 'Generic Item 2', 'Generic Item 3']  # Placeholder for the 'Others' category
}

# Function to recommend products based on customer ID and clustering method
def recommend_products_for_customer(customer_id, method):
    if method not in ['kmeans', 'pso']:
        raise ValueError('Invalid method provided.')

    # Check if customer_id exists
    if customer_id not in df['customer_id'].values:
        raise ValueError('Customer ID not found.')

    # Filter the data based on the method
    if method == 'kmeans':
        customer_cluster = df[df['customer_id'] == customer_id]['kmeans_cluster'].values[0]
        cluster_data = df[df['kmeans_cluster'] == customer_cluster]
    elif method == 'pso':
        customer_cluster = df[df['customer_id'] == customer_id]['pso_cluster'].values[0]
        cluster_data = df[df['pso_cluster'] == customer_cluster]

    print(f'Cluster Data for Customer ID {customer_id} using {method}:')
    print(cluster_data.head())

    # Check if 'Purchase_Categories' is valid
    if 'Purchase_Categories' not in cluster_data.columns:
        raise ValueError('Purchase_Categories column is missing.')

    # Remove "Others" category from the cluster data
    others_data = cluster_data[cluster_data['Purchase_Categories'] == inverse_category_mappings.get('Others', -1)]
    cluster_data = cluster_data[cluster_data['Purchase_Categories'] != inverse_category_mappings.get('Others', -1)]

    # Print data without "Others" category
    print('Cluster Data without "Others" Category:')
    print(cluster_data.head())

    # Get top 5 product categories based on frequency excluding "Others"
    top_categories_indices = cluster_data['Purchase_Categories'].value_counts().index[:5].tolist()

    # Decode category indices back to category names
    top_categories_names = [category_mappings.get(idx, str(idx)) for idx in top_categories_indices]
    print(f'Top Categories: {top_categories_names}')

    # Get top 10 products from each category
    recommendations = []
    for category in top_categories_names:
        products = category_to_products.get(category, [])
        recommendations.extend(products[:10])  # Limit to top 10 products per category

    # Add products from "Others" category as a separate case
    others_products = category_to_products.get('Others', [])
    recommendations.extend(others_products[:10])

    # Ensure unique recommendations
    recommendations = list(set(recommendations))

    # Limit to top 10 overall recommendations
    recommendations = recommendations[:10]

    print(f'Final Recommendations: {recommendations}')
    return recommendations

# API endpoint to recommend products for a specific customer
@app.route('/recommend-products', methods=['POST'])
def recommend_products():
    data = request.json
    customer_id = data.get('customer_id')
    method = data.get('method')

    if customer_id is None:
        return jsonify({'error': 'Customer ID is required.'}), 400
    if method not in ['kmeans', 'pso']:
        return jsonify({'error': 'Invalid method provided.'}), 400

    try:
        recommended_products = recommend_products_for_customer(customer_id, method)
        return jsonify({'recommended_products': recommended_products})
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': 'An error occurred while processing the request: {}'.format(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
