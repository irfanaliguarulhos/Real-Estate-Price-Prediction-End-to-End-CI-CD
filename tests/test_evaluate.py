import os
import subprocess

def test_evaluate_script():
    """
    Test the evaluate.py script with a sample model and test dataset.
    """
    # Define paths
    model_path = "/workspaces/Real-Estate-Price-Prediction-End-to-End-CI-CD/models/best_model.pkl"
    test_data_path = "/workspaces/Real-Estate-Price-Prediction-End-to-End-CI-CD/src/processed/NY-House-Features.csv"

    # Check if the model file exists
    assert os.path.isfile(model_path), f"Model file not found at {model_path}"

    # Check if the test dataset exists
    assert os.path.isfile(test_data_path), f"Test dataset not found at {test_data_path}"

    # Run the evaluate.py script
    result = subprocess.run(
        [
            "python",
            "/workspaces/Real-Estate-Price-Prediction-End-to-End-CI-CD/src/evaluate.py",
            "--model-path", model_path,
            "--test-data", test_data_path
        ],
        capture_output=True,
        text=True
    )

    # Check if the script ran successfully
    assert result.returncode == 0, f"Script failed with error: {result.stderr}"

    # Check for expected output
    assert "RMSE:" in result.stdout, "RMSE not found in script output"
    assert "R2 Score:" in result.stdout, "R2 Score not found in script output"

    print("Test passed successfully!")

if __name__ == "__main__":
    test_evaluate_script()