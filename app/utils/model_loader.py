import os
import pickle
import json
import pandas as pd
from typing import List

# Path configuration
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'model', 'model.pkl')
FEATURES_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'model', 'model_features.json')
DEMOGRAPHICS_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'zipcode_demographics.csv')

def load_artifacts():
    """
    Loads the model, features, and demographic data.
    Returns a tuple containing the model, the complete list of features,
    the demographic data, and the list of bonus features.
    """
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        
        with open(FEATURES_PATH, 'r') as f:
            required_features_all = json.load(f)
        
        demographics_df = pd.read_csv(DEMOGRAPHICS_PATH)
        
        # Define the list of features for the bonus endpoint explicitly,
        # using only the non-demographic columns.
        bonus_features = [
            "bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors",
            "waterfront", "view", "condition", "grade", "sqft_above",
            "sqft_basement", "yr_built", "yr_renovated", "zipcode",
            "lat", "long", "sqft_living15", "sqft_lot15"
        ]
        # Filter the complete features to ensure the bonus model
        # uses only the features defined in the list above.
        bonus_features_ordered = [f for f in required_features_all if f in bonus_features]
        print(f"Features for the bonus endpoint: {bonus_features_ordered}")

        return model, required_features_all, demographics_df, bonus_features_ordered

    except FileNotFoundError as e:
        print(f"Error loading model files: {e}")
        raise RuntimeError("Fatal error: Could not load model artifacts or demographic data.")

# Load artifacts when the module is first imported.
model, required_features_all, demographics_df, bonus_features_ordered = load_artifacts()
