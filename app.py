from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from pyswarm import pso

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
    'Review_Helpfulness', 'Recommendation_Helpfulness', 'Shopping_Satisfaction',
    'Service_Appreciation', 'Improvement_Areas'
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
df['kmeans_cluster'] = kmeans.fit_predict(X_scaled)

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
df['pso_cluster'] = pso_kmeans(X_scaled, k=5)

# Function to recommend products based on customer ID and clustering method
def recommend_products_for_customer(customer_id, method):
    if method not in ['kmeans', 'pso']:
        raise ValueError('Invalid method provided.')

    if method == 'kmeans':
        customer_cluster = df[df['customer_id'] == customer_id]['kmeans_cluster'].values[0]
        cluster_data = df[df['kmeans_cluster'] == customer_cluster]
    elif method == 'pso':
        customer_cluster = df[df['customer_id'] == customer_id]['pso_cluster'].values[0]
        cluster_data = df[df['pso_cluster'] == customer_cluster]

    # Check if 'Purchase_Categories' is valid
    if 'Purchase_Categories' not in cluster_data.columns:
        raise ValueError('Purchase_Categories column is missing.')

    # Get top 5 product categories based on frequency
    top_categories_indices = cluster_data['Purchase_Categories'].value_counts().index[:5].tolist()

    # Decode category indices back to category names
    top_categories_names = [category_mappings.get(idx, str(idx)) for idx in top_categories_indices]

    return top_categories_names

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
