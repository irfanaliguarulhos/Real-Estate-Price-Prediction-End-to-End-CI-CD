# Real-Estate-Price-Prediction-End-to-End-CI-CD

# Real Estate Price Prediction

An end-to-end data science pipeline for predicting real estate prices using free APIs.

## 1. Flowchart
```text
+-----------------+    +---------+    +----------------+    +--------------------------+
|  Data Source    | -> |  ETL    | -> | Data Warehouse | -> | Feature Engineering &    |
| (Free Real      |    | (src/   |    | (data/raw &    |    | EDA (src/feature_       |
| Estate API)     |    | etl.py) |    |  processed)    |    | engineering.py)         |
+-----------------+    +---------+    +----------------+    +-----------+--------------+
                                                                            |
                                                                            v
                                                                     +--------------+   +----------------+
                                                                     | Model Training|->| Evaluation &    |
                                                                     | (src/model.py)|   | Testing (src/   |
                                                                     +--------------+   |evaluate.py)    |
                                                                                         +----------------+
                                                                                                  |
                                                                                                  v
                                                                                     +---------------------+   +---------------------+
                                                                                     | MLflow Registry &   |->| Deployment &        |
                                                                                     | Model Packaging     |   | Dashboard (src/     |
                                                                                     | (models/, mlflow)   |   | dashboard/)         |
                                                                                     +---------------------+   +---------------------+
                                                                                                  |
                                                                                                  v
                                                                                     +---------------------+
                                                                                     | CI/CD (GitHub       |
                                                                                     | Actions, Docker)    |
                                                                                     +---------------------+
````

## 2. Directory Structure

````text
├── data/
│   ├── raw/                # Raw API extracts
│   └── processed/          # Cleaned & feature-engineered data
├── src/
│   ├── etl.py              # API extraction pipeline
│   ├── feature_engineering.py
│   ├── model.py            # Training & MLflow logging
│   ├── evaluate.py         # Model evaluation scripts
│   └── dashboard/
│       ├── app.py          # Flask app for predictions & dashboard
│       └── templates/
│           └── index.html  # Dark-theme HTML dashboard
├── mlflow/                 # MLflow tracking server config (optional)
├── tests/
│   ├── test_etl.py
│   └── test_model.py
├── .github/
│   └── workflows/
│       └── ci-cd.yml       # CI/CD pipeline definition
├── Dockerfile              # Containerize app & API
├── docker-compose.yml      # Local orchestration
├── requirements.txt        # Python dependencies
└── README.md               # Project overview & instructions
````



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

