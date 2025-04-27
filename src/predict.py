from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import os
from waitress import serve
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Get the project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class ModelServer:
    def __init__(self):
        self.model = None
        self.imputer = None
        self.encoder = None
        self.scaler = None
        self.load_artifacts()

    def load_artifacts(self):
        """Load model artifacts"""
        try:
            models_dir = os.path.join(PROJECT_ROOT, 'models')
            self.model = joblib.load(os.path.join(models_dir, 'best_model.pkl'))
            self.imputer = joblib.load(os.path.join(models_dir, 'imputer.pkl'))
            self.encoder = joblib.load(os.path.join(models_dir, 'encoder.pkl'))
            self.scaler = joblib.load(os.path.join(models_dir, 'scaler.pkl'))
            logger.info("Model artifacts loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model artifacts: {str(e)}")
            raise

    def predict(self, data):
        """Make prediction"""
        try:
            df = pd.DataFrame([data])
            
            # Preprocess
            df[['bedrooms','bathrooms','size']] = self.imputer.transform(
                df[['bedrooms','bathrooms','size']]
            )
            cat = self.encoder.transform(df[['region']])
            cat_cols = self.encoder.get_feature_names_out(['region'])
            
            df = pd.concat([
                df.drop(columns=['region']), 
                pd.DataFrame(cat, columns=cat_cols, index=df.index)
            ], axis=1)
            
            df[['bedrooms','bathrooms','size']] = self.scaler.transform(
                df[['bedrooms','bathrooms','size']]
            )

            pred = self.model.predict(df)[0]
            return float(pred)
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise

# Initialize Flask app
app = Flask(__name__, 
    template_folder=os.path.join(PROJECT_ROOT, 'src', 'dashboard', 'templates'),
    static_folder=os.path.join(PROJECT_ROOT, 'src', 'dashboard', 'static')
)

# Initialize model server
model_server = ModelServer()

@app.route('/')
def home():
    """Serve dashboard"""
    try:
        return render_template('dashboard.html')
    except Exception as e:
        logger.error(f"Failed to render dashboard: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No input data provided'}), 400

        pred = model_server.predict(data)
        return jsonify({'predicted_price': pred})
    except Exception as e:
        logger.error(f"Prediction request failed: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Get configuration from environment variables
    host = os.getenv('FLASK_HOST', '0.0.0.0')
    port = int(os.getenv('FLASK_PORT', 5000))
    env = os.getenv('FLASK_ENV', 'production')

    logger.info(f"Starting server in {env} mode on {host}:{port}")
    
    if env == 'development':
        # Use Flask's development server
        app.run(host=host, port=port, debug=True)
    else:
        # Use Waitress production server
        serve(app, host=host, port=port, threads=4)