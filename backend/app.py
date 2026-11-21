# Import necessary libraries
import numpy as np
import joblib  # For loading the serialized model
import pandas as pd  # For data manipulation
from flask import Flask, request, jsonify  # For creating the Flask API

# Initialize the Flask application
sales_revenue_predictor_api = Flask("SuperKart Sales Revenue Predictor")

# Load the trained machine learning model
model = joblib.load("superkart_model.joblib")

# Define a route for the home page (GET request)
@sales_revenue_predictor_api.get('/')
def home():
    """
    This function handles GET requests to the root URL ('/') of the API.
    It returns a simple welcome message.
    """
    return "Welcome to the SuperKart Sales Revenue Prediction API!"

# Define an endpoint for single sales revenue prediction (POST request)
@sales_revenue_predictor_api.post('/v1/sales')
def predict_sales_revenue():
    """
    This function handles POST requests to the '/v1/sales' endpoint.
    It expects a JSON payload containing product details and returns
    the predicted sales revenue as a JSON response.
    """
    # Get the JSON data from the request body
    product_data = request.get_json()

    # Extract relevant features from the JSON data
    sample = {
        'Product_Weight': product_data['Product_Weight'],
        'Product_Sugar_Content': product_data['Product_Sugar_Content'],
        'Product_Allocated_Area': product_data['Product_Allocated_Area'],
        'Product_MRP': product_data['Product_MRP'],
        'Store_Size': product_data['Store_Size'],
        'Store_Location_City_Type': product_data['Store_Location_City_Type'],
        'Store_Type': product_data['Store_Type'],
        'Product_Id_char': product_data['Product_Id_char'],
        'Store_Age_Years': product_data['Store_Age_Years'],
        'Product_Type_Category': product_data['Product_Type_Category']
    }

    # Convert the extracted data into a Pandas DataFrame
    input_data = pd.DataFrame([sample])

    # Make prediction (get log_price)
    predicted_sales_revenue = model.predict(input_data)[0]


    # Convert predicted_sales_revenue to Python float
    predicted_sales_revenue = round(float(predicted_sales_revenue), 2)

    # Return the actual price
    return jsonify({'Predicted Sales Revenue (in dollars)': predicted_sales_revenue})


# Define an endpoint for batch prediction (POST request)
@sales_revenue_predictor_api.post('/v1/salesbatch')
def predict_sales_revenue_batch():
    """
    This function handles POST requests to the '/v1/salesbatch' endpoint.
    It expects a CSV file containing product details for multiple products
    and returns the predicted sales revenue as a dictionary in the JSON response.
    """
    # Get the uploaded CSV file from the request
    file = request.files['file']

    # Read the CSV file into a Pandas DataFrame
    input_data = pd.read_csv(file)

    # Make predictions for all products in the DataFrame
    predicted_sales_revenue = [round(float(val), 2) for val in model.predict(input_data)]

    # Use 'Product_Id' if present; otherwise generate row-based keys
    if 'Product_Id' in input_data.columns:
        row_keys = input_data['Product_Id'].tolist()
    else:
        row_keys = [f"Record_{i + 1}" for i in range(len(input_data))]

    # Create and return the dictionary of predictions
    output_dict = dict(zip(row_keys, predicted_sales_revenue))
    return jsonify(output_dict)

if __name__ == '__main__':
    sales_revenue_predictor_api.run(debug=True)
