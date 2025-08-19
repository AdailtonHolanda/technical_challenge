import pandas as pd
import requests
import json
import os


API_URL = "http://localhost:8000/predict_all"
DATA_PATH = "data/future_unseen_examples.csv"

def run_test():
    """
    Load sample data, send a random row to the API, and print the response.
    """

    print("Starting the test script...")

    # Try to load the CSV file.
    df = pd.read_csv(DATA_PATH)
    print(f"File '{DATA_PATH}' successfully loaded. Total of {len(df)} examples.")

    random_row = df.sample(n=1).iloc[0]
    payload = random_row.to_dict()

    # Clean and convert the data to match the Pydantic model
    cleaned_payload = {}
    for key, value in payload.items():
        if pd.isna(value):
            cleaned_payload[key] = None
        else:
            try:
                if key in ['id', 'bedrooms', 'sqft_living', 'sqft_lot', 'waterfront', 'view', 'condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode', 'sqft_living15', 'sqft_lot15']:
                    cleaned_payload[key] = int(value)
                elif key in ['bathrooms', 'floors', 'lat', 'long']:
                    cleaned_payload[key] = float(value)
                else:
                    cleaned_payload[key] = value
            except (ValueError, TypeError):
                cleaned_payload[key] = value

    print("Processing a random example...")
    print(json.dumps(cleaned_payload, indent=2))

    try:
        response = requests.post(API_URL, json=cleaned_payload)
        response.raise_for_status()

        print("\nAPI Response (status", response.status_code, "):")
        print(json.dumps(response.json(), indent=2))

    except requests.exceptions.RequestException as e:
        if response and response.status_code == 422:
            print(f"\nValidation error (code {response.status_code}):")
            print(json.dumps(response.json(), indent=2))
        else:
            print("\nError sending the request:", e)

    print("Tests completed.")


if __name__ == "__main__":
    run_test()