import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_preprocess import load_data as _load_data

# Dataset path
DATASET_PATH = "/Users/gurudev/Desktop/VS Code/MyProjects/AIML/human+activity+recognition+using+smartphones/UCI HAR Dataset/"

def load_data():
    """Load UCI-HAR dataset with predefined path"""
    return _load_data(DATASET_PATH)
