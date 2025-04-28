# Real-Estate-Price-Prediction-End-to-End-CI-CD

# Real Estate Price Prediction

An end-to-end data science pipeline for predicting real estate prices using free APIs.

## Components

- **ETL**: Fetch and store raw data (`src/etl.py`).
- **Feature Engineering**: Clean and create features (`src/feature_engineering.py`).
- **Modeling**: Train and log model with MLflow (`src/model.py`).
- **Evaluation**: Compute metrics (`src/evaluate.py`).
- **Dashboard**: Interactive dark-themed Flask app (`src/dashboard/`).
- **CI/CD**: Automated tests, build, and Docker deployment (`.github/workflows/ci-cd.yml`).

## Getting Started

1. Clone repository
2. Set `API_KEY` environment variable
3. Install dependencies: `pip install -r requirements.txt`
4. Run pipeline:
   ```bash
   python src/etl.py
   python src/feature_engineering.py
   python src/model.py
