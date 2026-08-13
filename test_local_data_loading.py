import json
import os

def test_data_loading():
    """Verify that we can load the local ARC-AGI-2 data from the ./data symlink."""
    train_path = './data/arc-agi_training_challenges.json'
    assert os.path.exists(train_path), f"File not found: {train_path}"

    with open(train_path, 'r') as f:
        data = json.load(f)

    assert isinstance(data, dict), "Data should be a dictionary."
    assert len(data) > 0, "Data dictionary should not be empty."
    print("Local data loading validated successfully.")

if __name__ == '__main__':
    test_data_loading()
