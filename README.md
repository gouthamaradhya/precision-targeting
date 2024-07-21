# Precision Targeting

## Project Overview

Precision Targeting is a project aimed at recommending products to customers based on their shopping behavior using clustering algorithms like K-means++ and PSO.

## Features

- Customer segmentation using K-means++ and PSO clustering algorithms
- Product recommendation based on customer segments
- Frontend application built with React

## Prerequisites

- Python 3.x
- Node.js
- npm (Node Package Manager)

## Setting Up the Project

### Backend (Python Flask)

1. **Create and activate a virtual environment:**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

2. **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

3. **Run the Flask app:**

    ```bash
    python app.py
    ```

### Frontend (React)

1. **Navigate to the client directory:**

    ```bash
    cd precision-targeting-client
    ```

2. **Install dependencies:**

    ```bash
    npm install
    ```

3. **Start the React app:**

    ```bash
    npm start
    ```

## Usage

1. Open your browser and go to `http://localhost:3000` to use the frontend application.
2. Use the form to input the customer ID and select the clustering method to get product recommendations.

## Contributing

Feel free to fork this repository and submit pull requests.

## License

This project is licensed under the MIT License.
