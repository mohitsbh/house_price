from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__, template_folder='../template')

# Load the trained model
model_path = os.path.join(os.path.dirname(__file__), '../model/house_price_model.pkl')
model = None

try:
    model = joblib.load(model_path)
    print("Model loaded successfully!")
except FileNotFoundError:
    print("Model not found. Please train the model first by running model/model.py")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded. Please train the model first.'}), 500
        
        # Get data from form
        data = request.json
        
        # Feature order must match training data
        features = [
            float(data['bedrooms']),
            float(data['bathrooms']),
            float(data['sqft_living']),
            float(data['sqft_lot']),
            float(data['floors']),
            float(data['waterfront']),
            float(data['view']),
            float(data['condition']),
            float(data['grade']),
            float(data['sqft_above']),
            float(data['sqft_basement']),
            float(data['yr_built']),
            float(data['yr_renovated']),
            float(data['zipcode']),
            float(data['lat']),
            float(data['long']),
            float(data['sqft_living15']),
            float(data['sqft_lot15'])
        ]
        
        # Make prediction
        prediction = model.predict([features])[0]
        
        return jsonify({'predicted_price': round(prediction, 2)})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', debug=True, port=port)
