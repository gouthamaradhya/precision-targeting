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
            <div className='flex justify-center items-center bg-slate-900 h-24 w-screen'>
                <div className='font-anton text-zinc-200 text-5xl text-center'>Precision Targeting</div>
            </div>
            <div className="bg-slate-300 w-screen h-screen">
                <div className='flex h-24 w-full items-center justify-center absolute top-24'>
                    <div className='text-3xl font-anton'>Product Recommendation</div>
                </div>
                <div className='absolute w-full top-52 flex justify-center'>
                    <div className='bg-white w-2/3 rounded-md'>
                        <div>
                            <form onSubmit={handleSubmit} className=''>
                                <div className='flex justify-center'>
                                    <label className='font-anton text-xl'>
                                        CUSTOMER ID:
                                        <input
                                            type="number"
                                            value={customer_id}
                                            onChange={(e) => setCustomer_id(e.target.value)}
                                            required
                                            className='border-b border-black  ml-7 w-20'
                                        />
                                    </label>
                                </div>
                                <div className='flex justify-center'>
                                    <label className='font-anton text-xl' >
                                        METHOD:
                                        <select value={method} onChange={(e) => setMethod(e.target.value)} className=' border-black rounded ml-20 mt-14' >
                                            <option value="kmeans">K-means</option>
                                            <option value="pso">PSO</option>
                                        </select>
                                    </label>
                                </div>
                                <div className='flex justify-center'>
                                    <button type="submit" disabled={!customer_id || !method || loading} className='font-anton bg-blue-700 text-white rounded-md w-56 text-xl mt-12 ml-5 '>
                                        {loading ? 'Loading...' : 'Get Recommendations'}
                                    </button>
                                </div>
                            </form>
                            <div>
                                {error && <p style={{ color: 'red' }}>{error}</p>}
                                {recommendedProducts.length > 0 ? (
                                    <ul>
                                        {recommendedProducts.map((product, index) => (
                                            <li key={index} className='mt-2 text-md ml-11'>{product}</li>
                                        ))}
                                    </ul>
                                ) : (
                                    !loading && <p className='mt-10 text-md text-center'>No recommendations found.</p>
                                )}
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default ProductRecommendation;
