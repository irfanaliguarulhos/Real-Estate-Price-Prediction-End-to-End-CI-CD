from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__, template_folder='src/dashboard/templates', static_folder='src/dashboard/static')

# Load artifacts
model = joblib.load('models/best_model.pkl')
imputer = joblib.load('models/imputer.pkl')
encoder = joblib.load('models/encoder.pkl')
scaler = joblib.load('models/scaler.pkl')

@app.route('/')
def home():
    return render_template('dashboard.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    df = pd.DataFrame([data])

    # Preprocess
    df[['bedrooms','bathrooms','size']] = imputer.transform(df[['bedrooms','bathrooms','size']])
    cat = encoder.transform(df[['region']])
    cat_cols = encoder.get_feature_names_out(['region'])
    df = pd.concat([df.drop(columns=['region']), pd.DataFrame(cat, columns=cat_cols)], axis=1)
    df[['bedrooms','bathrooms','size']] = scaler.transform(df[['bedrooms','bathrooms','size']])

    pred = model.predict(df)[0]
    return jsonify({'predicted_price': float(pred)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)