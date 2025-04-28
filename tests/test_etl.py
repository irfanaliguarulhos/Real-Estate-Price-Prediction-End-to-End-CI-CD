import os
import sys
import pandas as pd
from pathlib import Path
import pytest

# Dynamically add the `src` directory to the Python path
project_root = Path(__file__).resolve().parents[1]
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.append(str(src_path))

try:
    from src.etl import run_etl
except ImportError as e:
    raise ImportError(f"Error importing 'run_etl' from 'src.etl': {e}")

@pytest.fixture
def mock_api_response(monkeypatch):
    """Mock the API response for the ETL process."""
    class MockResponse:
        @staticmethod
        def json():
            return {
                "properties": [
                    {
                        "price": 100000,
                        "area_sqft": 500,
                        "bedrooms": 2,
                        "bathrooms": 1,
                        "features": "pool"
                    }
                ]
            }

        def raise_for_status(self):
            pass

    monkeypatch.setattr('requests.get', lambda *args, **kwargs: MockResponse())
    monkeypatch.setenv("API_KEY", "testkey")

def test_etl_creates_csv(tmp_path, mock_api_response):
    """
    Test that the ETL process creates a CSV file with the expected data.
    """
    # Change to a temporary directory
    cwd = tmp_path
    os.chdir(cwd)

    # Run the ETL process
    run_etl()

    # Check if the CSV file was created
    csv_path = cwd / "data/raw/properties.csv"
    assert csv_path.exists(), f"CSV file not found at {csv_path}"

    # Load the CSV file and check its contents
    df = pd.read_csv(csv_path)
    assert not df.empty, "The CSV file is empty"
    assert "price" in df.columns, "Expected column 'price' not found in the CSV"
    assert "area_sqft" in df.columns, "Expected column 'area_sqft' not found in the CSV"
    assert "bedrooms" in df.columns, "Expected column 'bedrooms' not found in the CSV"
    assert "bathrooms" in df.columns, "Expected column 'bathrooms' not found in the CSV"
    assert "features" in df.columns, "Expected column 'features' not found in the CSV"