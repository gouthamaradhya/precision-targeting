import axios from 'axios';
import React, { useState } from 'react';

const ProductRecommendation = () => {
    const [customer_id, setCustomer_id] = useState('');
    const [method, setMethod] = useState('');
    const [recommendedProducts, setRecommendedProducts] = useState([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    const handleSubmit = async (e) => {
        e.preventDefault();
        setLoading(true);
        setError(null); // Clear any previous errors

        try {
            const response = await axios.post('http://127.0.0.1:5000/recommend-products',
                { customer_id: parseInt(customer_id, 10), method }, // Ensure customer_id is a number
                { headers: { 'Content-Type': 'application/json' } }  // Headers
            );

            if (response.status === 200) {
                setRecommendedProducts(response.data.recommended_products);
            } else {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
        } catch (error) {
            console.error('Error fetching recommendations:', error.message);
            setError(`Error fetching recommendations: ${error.message}`);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div>
            <h2>Product Recommendation</h2>
            <form onSubmit={handleSubmit}>
                <div>
                    <label>
                        Customer ID:
                        <input
                            type="number"
                            value={customer_id}
                            onChange={(e) => setCustomer_id(e.target.value)}
                            required
                        />
                    </label>
                </div>
                <div>
                    <label>
                        Method:
                        <select value={method} onChange={(e) => setMethod(e.target.value)}>
                            <option value="kmeans">K-means</option>
                            <option value="pso">PSO</option>
                        </select>
                    </label>
                </div>
                <button type="submit" disabled={!customer_id || !method || loading}>
                    {loading ? 'Loading...' : 'Get Recommendations'}
                </button>
            </form>
            {error && <p style={{ color: 'red' }}>{error}</p>}
            {recommendedProducts.length > 0 ? (
                <ul>
                    {recommendedProducts.map((product, index) => (
                        <li key={index}>{product}</li>
                    ))}
                </ul>
            ) : (
                !loading && <p>No recommendations found.</p>
            )}
        </div>
    );
};

export default ProductRecommendation;
