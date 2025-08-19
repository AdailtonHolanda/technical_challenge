import pandas as pd
import requests
import json


API_URL_BONUS = "http://localhost:8000/predict_bonus"
DATA_PATH = "data/future_unseen_examples.csv"


def run_test():
    """
    Load sample data, send a random row to the API, and print the response.
    """
    print("Starting the test script for the bonus route...") 

    df = pd.read_csv(DATA_PATH)
    print(f"File '{DATA_PATH}' successfully loaded. Total of {len(df)} examples.")

    with open("model/model_features.json", "r") as f:
        required_features = json.load(f)
    print(f"Required features for the bonus route: {required_features}")

    random_row = df.sample(n=1).iloc[0]

    # 4. Create the payload with the required features that are present in the CSV
    # Filter the required features to include only those that exist in the test DataFrame.
    features_in_csv = [f for f in required_features if f in random_row.index]

    # Prepare the payload by converting the data types to what the model expects
    payload = {}
    for feature in features_in_csv:
        value = random_row[feature]
        if pd.notna(value) and value == int(value):
            payload[feature] = int(value)
        elif pd.notna(value):
            payload[feature] = float(value)
        else:
            payload[feature] = None 

    print("Sending payload (only required features found in the CSV):")
    print(json.dumps(payload, indent=2))

    try:
        response = requests.post(API_URL_BONUS, json=payload)
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